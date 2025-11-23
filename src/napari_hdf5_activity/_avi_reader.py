"""
_avi_reader.py - AVI video file reader module

This module handles reading AVI video files,
making them compatible with the HDF5 analysis pipeline.
"""

import json
import os
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _scan_video_metadata(
    video_idx: int,
    video_path: str,
    target_frame_interval: float,
) -> Tuple[int, Optional[Dict], str]:
    """
    Scan video metadata without loading frames (fast).

    Returns:
        Tuple of (video_idx, metadata_dict, error_message)
    """
    video_name = Path(video_path).name

    try:
        with AVIVideoReader(video_path) as reader:
            video_fps = reader.fps
            frames_per_sample = max(1, int(video_fps * target_frame_interval))

            # Calculate how many frames we'll sample
            sampled_frame_count = len(range(0, reader.frame_count, frames_per_sample))

            if sampled_frame_count == 0:
                return video_idx, None, f"No frames to sample from {video_name}"

            metadata = {
                "path": video_path,
                "name": video_name,
                "index": video_idx,
                "fps": video_fps,
                "frame_count": reader.frame_count,
                "duration": reader.duration,
                "sampled_frames": sampled_frame_count,
                "frames_per_sample": frames_per_sample,
                "height": reader.height,
                "width": reader.width,
            }

            return video_idx, metadata, ""

    except Exception as e:
        return video_idx, None, f"Error scanning {video_name}: {str(e)}"


def _load_single_video_for_batch(
    video_idx: int,
    video_path: str,
    target_frame_interval: float,
) -> Tuple[int, Optional[Dict], Optional[List], str]:
    """
    Load a single video for batch processing (used in parallel).

    Args:
        video_idx: Index of this video in the batch
        video_path: Path to the video file
        target_frame_interval: Target time interval between frames

    Returns:
        Tuple of (video_idx, metadata_dict, frames_list, error_message)
        If error occurs, metadata and frames will be None and error_message will be set.
    """
    video_name = Path(video_path).name

    try:
        with AVIVideoReader(video_path) as reader:
            video_fps = reader.fps
            frames_per_sample = max(1, int(video_fps * target_frame_interval))

            # Sample frames at target interval
            video_frames = []
            for frame_idx in range(0, reader.frame_count, frames_per_sample):
                frame = reader.get_frame(frame_idx)
                if frame is not None:
                    video_frames.append((frame_idx, frame))

            if not video_frames:
                return (
                    video_idx,
                    None,
                    None,
                    f"No frames could be read from {video_name}",
                )

            # Return metadata needed for timestamp calculation
            metadata = {
                "path": video_path,
                "name": video_name,
                "index": video_idx,
                "fps": video_fps,
                "frame_count": reader.frame_count,
                "duration": reader.duration,
                "sampled_frames": len(video_frames),
                "frames_per_sample": frames_per_sample,
            }

            return video_idx, metadata, video_frames, ""

    except Exception as e:
        return video_idx, None, None, f"Error loading {video_name}: {str(e)}"


def load_avi_batch_timeseries(
    video_paths: List[str],
    target_frame_interval: float = 5.0,
    progress_callback=None,
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
        progress_callback: Optional callback function(percent, message) for progress updates

    Returns:
        Tuple of (concatenated frames array, combined metadata dict)

    Example:
        Video 1: 0s - 600s
        Video 2: 600s - 1200s
        Video 3: 1200s - 1800s
        All sampled at 0.2s intervals
    """
    all_timestamps = []
    combined_metadata = {
        "videos": [],
        "total_duration": 0.0,
        "target_frame_interval": target_frame_interval,
        "source_type": "avi_batch",
    }

    total_videos = len(video_paths)
    current_time_offset = 0.0

    # Determine number of parallel workers and chunk size
    num_workers = min(6, os.cpu_count() or 4)
    # Process videos in chunks to avoid memory overflow
    # Chunk size of 10 videos for counting/metadata phase
    chunk_size = 10

    print(
        f"\nPhase 1: Scanning {total_videos} videos to determine total frame count..."
    )
    print(f"Using chunks of {chunk_size} with {num_workers} parallel workers...")

    # Phase 1: Quick scan to get metadata and total frame count
    print("Scanning video metadata (fast, no frame loading)...")
    scan_results = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_scan_video_metadata, idx, path, target_frame_interval): idx
            for idx, path in enumerate(video_paths)
        }

        for future in as_completed(futures):
            video_idx = futures[future]
            idx_result, metadata, error_msg = future.result(timeout=60)

            if progress_callback:
                percent = ((idx_result + 1) / total_videos) * 20  # Scanning is 0-20%
                progress_callback(
                    percent, f"Scanned {idx_result + 1}/{total_videos} videos"
                )

            scan_results[idx_result] = (metadata, error_msg)

            if error_msg:
                print(f"  ✗ Video {idx_result + 1}: {error_msg}")
            else:
                print(
                    f"  ✓ Video {idx_result + 1}: {metadata['name']} ({metadata['sampled_frames']} frames)"
                )

    # Calculate total frame count and dimensions
    total_frame_count = 0
    frame_height = None
    frame_width = None
    all_video_metadata = []

    for video_idx in sorted(scan_results.keys()):
        metadata, error_msg = scan_results[video_idx]

        # Stop at first error
        if error_msg or metadata is None:
            if error_msg:
                print(f"\nERROR at video {video_idx + 1}: {error_msg}")
            print(f"STOPPING: Will process only first {video_idx} videos")
            break

        total_frame_count += metadata["sampled_frames"]
        if frame_height is None:
            frame_height = metadata["height"]
            frame_width = metadata["width"]

        all_video_metadata.append(metadata)

    if not all_video_metadata:
        raise ValueError("No valid videos found")

    print("\nScan complete:")
    print(f"  Videos to process: {len(all_video_metadata)}")
    print(f"  Total frames: {total_frame_count}")
    print(f"  Frame size: {frame_height} × {frame_width}")
    print(
        f"  Memory required: ~{(total_frame_count * frame_height * frame_width) / (1024**3):.1f} GB"
    )

    # Create memory-mapped array on disk
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"napari_avi_batch_{os.getpid()}.dat")
    print(f"\nCreating memory-mapped array: {temp_file}")

    frames_array = np.memmap(
        temp_file,
        dtype="uint8",
        mode="w+",
        shape=(total_frame_count, frame_height, frame_width, 1),
    )

    # Phase 2: Load frames into memory-mapped array (chunked)
    print("\nPhase 2: Loading frames into memory-mapped array...")
    print(f"Processing in chunks of {chunk_size} videos...")

    current_frame_idx = 0

    for chunk_start in range(0, len(all_video_metadata), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(all_video_metadata))
        chunk_metadata = all_video_metadata[chunk_start:chunk_end]

        print(f"\n--- Chunk: videos {chunk_start + 1}-{chunk_end} ---")

        # Load chunk videos in parallel
        chunk_results = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for meta in chunk_metadata:
                future = executor.submit(
                    _load_single_video_for_batch,
                    meta["index"],
                    meta["path"],
                    target_frame_interval,
                )
                futures[future] = meta["index"]

            for future in as_completed(futures):
                video_idx = futures[future]
                idx_result, metadata, frames_with_idx, error_msg = future.result(
                    timeout=120
                )

                if progress_callback:
                    overall_progress = (
                        20 + ((video_idx + 1) / total_videos) * 70
                    )  # Loading is 20-90%
                    progress_callback(
                        overall_progress, f"Loading {video_idx + 1}/{total_videos}"
                    )

                chunk_results[idx_result] = (metadata, frames_with_idx, error_msg)

                if error_msg:
                    print(f"  ✗ Video {idx_result + 1}: {error_msg}")
                else:
                    print(f"  ✓ Video {idx_result + 1}: {metadata['name']}")

        # Write frames to memory-mapped array in order
        for video_idx in sorted(chunk_results.keys()):
            metadata, frames_with_idx, error_msg = chunk_results[video_idx]

            if error_msg:
                print(f"ERROR loading video {video_idx + 1}, stopping")
                break

            video_fps = metadata["fps"]
            video_duration = metadata["duration"]

            # Write frames to mmap array
            for frame_idx, frame in frames_with_idx:
                frames_array[current_frame_idx, :, :, 0] = frame

                # Calculate timestamp
                frame_time = (frame_idx / video_fps) + current_time_offset
                all_timestamps.append(frame_time)

                current_frame_idx += 1

            # Update time offset
            current_time_offset += video_duration

            # Store video metadata
            combined_metadata["videos"].append(
                {
                    "path": metadata["path"],
                    "index": video_idx,
                    "fps": video_fps,
                    "frame_count": metadata["frame_count"],
                    "duration": video_duration,
                    "sampled_frames": metadata["sampled_frames"],
                    "time_start": all_timestamps[
                        current_frame_idx - metadata["sampled_frames"]
                    ],
                    "time_end": all_timestamps[current_frame_idx - 1],
                    "frames_per_sample": metadata["frames_per_sample"],
                }
            )

        del chunk_results  # Free memory

        print(
            f"Chunk complete. Frames written: {current_frame_idx}/{total_frame_count}"
        )

    if current_frame_idx == 0:
        raise ValueError("No frames could be loaded from any video")

    # Ensure memory-mapped array is fully written to disk
    frames_array.flush()

    # Complete metadata
    combined_metadata["total_duration"] = current_time_offset
    combined_metadata["total_frames"] = current_frame_idx
    combined_metadata["timestamps"] = all_timestamps
    combined_metadata["effective_fps"] = 1.0 / target_frame_interval
    combined_metadata["frame_interval"] = target_frame_interval
    combined_metadata["mmap_file"] = temp_file  # Store temp file path for cleanup

    if progress_callback:
        progress_callback(100, "Loading complete")

    print("\nBatch processing complete:")
    print(f"  Total videos processed: {len(combined_metadata['videos'])}")
    print(f"  Total frames: {current_frame_idx}")
    print(
        f"  Total duration: {current_time_offset:.1f}s ({current_time_offset/60:.1f} min)"
    )
    print(f"  Effective FPS: {combined_metadata['effective_fps']:.2f}")
    print(f"  Memory-mapped file: {temp_file}")
    print("  Note: Array is memory-mapped to disk (low RAM usage)")

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
