"""
_avi_reader.py - AVI video file reader module

This module handles reading AVI video files,
making them compatible with the HDF5 analysis pipeline.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print(
        "Warning: opencv-python not available. Install with: pip install opencv-python"
    )


class AVIVideoReader:
    """Reader for AVI video files with metadata support."""

    def __init__(self, video_path: str):
        """
        Initialize AVI video reader.

        Args:
            video_path: Path to .avi file
        """
        if not CV2_AVAILABLE:
            raise ImportError(
                "opencv-python is required for AVI support. Install with: pip install opencv-python"
            )

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

        # Simple metadata from video properties
        self.metadata = {
            "fps": self.fps,
            "frame_interval": 5.0,  # Default: 5 seconds between frames (like HDF5)
            "resolution": {"width": self.width, "height": self.height},
            "duration": self.duration,
            "frame_count": self.frame_count,
            "source": "avi",
        }

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """
        Get a specific frame from the video.

        Args:
            frame_index: Frame number to retrieve (0-indexed)

        Returns:
            Frame as numpy array (grayscale) or None if failed
        """
        if frame_index < 0 or frame_index >= self.frame_count:
            return None

        # Set frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()

        if not ret:
            return None

        # Convert to grayscale
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

    def get_frames_sampled(self, sampling_rate: int = 1) -> np.ndarray:
        """
        Get all frames with sampling (compatible with MATLAB frameRateOffline).

        Args:
            sampling_rate: Process 1 in N frames (e.g., 5 = every 5th frame)

        Returns:
            4D array: (n_frames, height, width, 1)
        """
        frames = []

        # Sample frames according to sampling rate
        for frame_idx in range(0, self.frame_count, sampling_rate):
            frame = self.get_frame(frame_idx)
            if frame is not None:
                frames.append(frame)

        if not frames:
            return np.array([])

        # Stack into 4D array (compatible with napari)
        frames_array = np.stack(frames, axis=0)
        frames_array = np.expand_dims(frames_array, axis=-1)  # Add channel dimension

        return frames_array

    def extract_led_data(self) -> Optional[Dict[str, List]]:
        """
        AVI files don't contain LED data.
        LED data is only available in HDF5 files.

        Returns:
            None (AVIs don't have LED data)
        """
        return None

    def get_metadata_dict(self) -> Dict[str, Any]:
        """Get metadata dictionary compatible with HDF5 format."""
        return {
            "fps": self.fps,
            "frame_interval": 1.0 / self.fps if self.fps > 0 else 0.2,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "resolution": {"width": self.width, "height": self.height},
            "source_type": "avi",
            "source_path": self.video_path,
            **self.metadata,
        }

    def close(self):
        """Release video capture."""
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor."""
        self.close()


def load_avi_with_metadata(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load AVI file and return frames with metadata (napari-compatible).

    Args:
        path: Path to AVI file

    Returns:
        Tuple of (frames array, metadata dict)
    """
    with AVIVideoReader(path) as reader:
        # Get sampling rate from metadata
        sampling_rate = reader.metadata.get("sampling_rate", 1)

        # Load frames with sampling
        frames = reader.get_frames_sampled(sampling_rate)

        # Get metadata
        metadata = reader.get_metadata_dict()

        return frames, metadata


def load_avi_batch_timeseries(
    video_paths: List[str], target_frame_interval: float = 5.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load multiple AVI files as continuous timeseries with proper time concatenation.

    This function ensures that multiple AVIs are processed with the same frame interval
    as HDF5 files, by sampling frames at the target interval and concatenating them
    temporally.

    Args:
        video_paths: List of AVI file paths in temporal order
        target_frame_interval: Target time interval between processed frames (seconds)
                              Default: 5.0s = 0.2 FPS effective sampling (1 frame per 5 seconds)

    Returns:
        Tuple of (concatenated frames array, combined metadata dict)

    Example:
        Video 1: 0s - 600s
        Video 2: 600s - 1200s
        Video 3: 1200s - 1800s
        All sampled at 0.2s intervals
    """
    all_frames = []
    all_timestamps = []
    combined_metadata = {
        "videos": [],
        "total_duration": 0.0,
        "target_frame_interval": target_frame_interval,
        "source_type": "avi_batch",
    }

    current_time_offset = 0.0

    for video_idx, video_path in enumerate(video_paths):
        print(
            f"Processing video {video_idx + 1}/{len(video_paths)}: {Path(video_path).name}"
        )

        with AVIVideoReader(video_path) as reader:
            # Calculate how many frames to skip to achieve target interval
            video_fps = reader.fps
            frames_per_sample = max(1, int(video_fps * target_frame_interval))

            print(f"  Video FPS: {video_fps}")
            print(f"  Target interval: {target_frame_interval}s")
            print(f"  Sampling every {frames_per_sample} frames")

            # Sample frames at target interval
            video_frames = []
            video_timestamps = []

            for frame_idx in range(0, reader.frame_count, frames_per_sample):
                frame = reader.get_frame(frame_idx)
                if frame is not None:
                    video_frames.append(frame)

                    # Calculate actual timestamp
                    frame_time = (frame_idx / video_fps) + current_time_offset
                    video_timestamps.append(frame_time)

            if video_frames:
                all_frames.extend(video_frames)
                all_timestamps.extend(video_timestamps)

                # Update time offset for next video
                video_duration = reader.duration
                current_time_offset += video_duration

                # Store video metadata
                combined_metadata["videos"].append(
                    {
                        "path": video_path,
                        "index": video_idx,
                        "fps": video_fps,
                        "frame_count": reader.frame_count,
                        "duration": video_duration,
                        "sampled_frames": len(video_frames),
                        "time_start": video_timestamps[0] if video_timestamps else 0,
                        "time_end": video_timestamps[-1] if video_timestamps else 0,
                        "frames_per_sample": frames_per_sample,
                    }
                )

                print(
                    f"  Sampled {len(video_frames)} frames from {reader.frame_count} total"
                )
                print(
                    f"  Time range: {video_timestamps[0]:.1f}s - {video_timestamps[-1]:.1f}s"
                )

    if not all_frames:
        raise ValueError("No frames could be loaded from any video")

    # Convert to numpy array
    frames_array = np.stack(all_frames, axis=0)
    frames_array = np.expand_dims(frames_array, axis=-1)  # Add channel dimension

    # Complete metadata
    combined_metadata["total_duration"] = current_time_offset
    combined_metadata["total_frames"] = len(all_frames)
    combined_metadata["timestamps"] = all_timestamps
    combined_metadata["effective_fps"] = 1.0 / target_frame_interval
    combined_metadata["frame_interval"] = target_frame_interval

    print("\nBatch processing complete:")
    print(f"  Total videos: {len(video_paths)}")
    print(f"  Total frames: {len(all_frames)}")
    print(
        f"  Total duration: {current_time_offset:.1f}s ({current_time_offset/60:.1f} min)"
    )
    print(f"  Effective FPS: {combined_metadata['effective_fps']:.2f}")

    return frames_array, combined_metadata


def create_metadata_template(output_path: str, experiment_name: str = "experiment"):
    """
    Create a template metadata JSON file for AVI videos.

    Args:
        output_path: Where to save the template
        experiment_name: Name of the experiment
    """
    template = {
        "experiment_name": experiment_name,
        "fps": 5,
        "frame_interval": 0.2,
        "sampling_rate": 5,
        "resolution": {"width": 1920, "height": 1080},
        "videos": [
            {
                "filename": "video_001.avi",
                "excluded_animals": [],
                "duration_minutes": 60,
            }
        ],
        "led_schedule": {
            "type": "custom",
            "light_periods": [{"start_hour": 7, "end_hour": 19}],
        },
        "timeseries": {
            "comment": "Optional: Include actual LED timeseries data here",
            "timestamps": [],
            "led_white_power_percent": [],
            "led_ir_power_percent": [],
        },
    }

    with open(output_path, "w") as f:
        json.dump(template, f, indent=2)

    print(f"Created metadata template: {output_path}")


# Compatibility function for existing MATLAB workflow
def convert_matlab_roi_to_napari(roi_mat_path: str) -> List[Dict]:
    """
    Convert MATLAB ROI.mat file to napari-compatible format.

    Args:
        roi_mat_path: Path to MATLAB .mat file with ROI data

    Returns:
        List of ROI dictionaries
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError(
            "scipy is required for MATLAB file conversion. Install with: pip install scipy"
        )

    mat_data = loadmat(roi_mat_path)
    parametersROI = mat_data.get("parametersROI", [])

    rois = []
    for i, roi in enumerate(parametersROI[0]):
        roi_dict = {
            "index": i,
            "type": "ellipse",  # Assume ellipse for now
            "center": roi[0]["Center"][0][0].tolist(),
            "semi_axes": roi[0]["SemiAxes"][0][0].tolist(),
            "rotation_angle": float(roi[0]["RotationAngle"][0][0]),
        }
        rois.append(roi_dict)

    return rois
