# import os
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import time
# from datetime import datetime, timedelta
# from typing import Dict, List, Tuple, Optional, Any, Union, Callable
# import numpy as np

# import logging

# # Enhanced import handling with fallbacks
# try:
#     import cv2
#     CV2_AVAILABLE = True
# except ImportError:
#     CV2_AVAILABLE = False
#     print("Warning: OpenCV not available. Circle detection will be disabled.")

# try:
#     import h5py
#     H5PY_AVAILABLE = True
# except ImportError:
#     H5PY_AVAILABLE = False
#     raise ImportError("h5py is required for HDF5 file processing")

# try:
#     import numpy as np
#     NUMPY_AVAILABLE = True
# except ImportError:
#     NUMPY_AVAILABLE = False
#     raise ImportError("numpy is required for data processing")

# try:
#     import matplotlib.pyplot as plt
#     MATPLOTLIB_AVAILABLE = True
# except ImportError:
#     MATPLOTLIB_AVAILABLE = False
#     print("Warning: matplotlib not available. Color generation will use fallback.")

# try:
#     import psutil
#     PSUTIL_AVAILABLE = True
# except ImportError:
#     PSUTIL_AVAILABLE = False
#     print("Warning: psutil not available. Using conservative memory estimates.")

# logger = logging.getLogger(__name__)

# # Memory management constants
# MAX_CHUNK_SIZE_MB = 512  # Maximum chunk size in MB
# MIN_CHUNK_SIZE = 10      # Minimum number of frames per chunk
# MAX_CHUNK_SIZE = 200     # Maximum number of frames per chunk
# MEMORY_SAFETY_FACTOR = 0.25  # Use only 25% of available memory

# def get_available_memory() -> int:
#     """Get available system memory in bytes."""
#     if PSUTIL_AVAILABLE:
#         try:
#             return psutil.virtual_memory().available
#         except Exception:
#             pass
#     # Fallback to conservative estimate
#     return 2 * 1024 * 1024 * 1024  # 2 GB

# def calculate_optimal_chunk_size(frame_shape: Tuple[int, ...], dtype_size: int = 2) -> int:
#     """
#     Calculate optimal chunk size based on available memory and frame size.

#     Args:
#         frame_shape: Shape of a single frame (height, width) or (height, width, channels)
#         dtype_size: Size of data type in bytes (2 for uint16, 1 for uint8, etc.)

#     Returns:
#         Optimal number of frames per chunk
#     """
#     available_memory = get_available_memory()
#     memory_budget = available_memory * MEMORY_SAFETY_FACTOR

#     # Calculate frame size in bytes
#     frame_size = np.prod(frame_shape) * dtype_size

#     # Add overhead for processing (diff operations, masks, etc.)
#     processing_overhead = 4  # Python-native processing overhead
#     effective_frame_size = frame_size * processing_overhead

#     # Calculate number of frames that fit in memory budget
#     frames_in_budget = int(memory_budget / effective_frame_size)

#     # Return a reasonable chunk size
#     optimal_size = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, frames_in_budget))

#     logger.info(f"Memory calculation: {memory_budget/(1024**2):.1f}MB budget, "
#                 f"{frame_size/(1024**2):.1f}MB per frame, optimal chunk: {optimal_size} frames")

#     return optimal_size

# # =============================================================================
# # HDF5 STRUCTURE DETECTION AND READING
# # =============================================================================

# def detect_hdf5_structure_type(file_path: str) -> Dict[str, Any]:
#     """
#     Detect the structure type of an HDF5 file.

#     Returns:
#         Dictionary with structure information
#     """
#     try:
#         with h5py.File(file_path, 'r') as h5_file:
#             root_keys = list(h5_file.keys())

#             # Structure 1: Stacked frames dataset
#             if 'frames' in h5_file:
#                 frames_dataset = h5_file['frames']
#                 if len(frames_dataset) > 0:
#                     return {
#                         'type': 'stacked_frames',
#                         'dataset_name': 'frames',
#                         'frame_count': len(frames_dataset),
#                         'frame_shape': frames_dataset[0].shape,
#                         'dtype': frames_dataset.dtype,
#                         'dtype_size': frames_dataset.dtype.itemsize,
#                         'data_location': 'frames'
#                     }

#             # Structure 2: Individual frames in images/ group
#             if 'images' in h5_file:
#                 images_group = h5_file['images']
#                 if len(images_group.keys()) > 0:
#                     first_key = sorted(images_group.keys())[0]
#                     first_image = images_group[first_key]
#                     return {
#                         'type': 'individual_frames',
#                         'group_name': 'images',
#                         'frame_count': len(images_group.keys()),
#                         'frame_shape': first_image.shape,
#                         'dtype': first_image.dtype,
#                         'dtype_size': first_image.dtype.itemsize,
#                         'data_location': 'images',
#                         'frame_keys': sorted(images_group.keys())
#                     }

#             # Structure 3: Look for other potential datasets
#             for key in root_keys:
#                 if isinstance(h5_file[key], h5py.Dataset):
#                     dataset = h5_file[key]
#                     # Check if this could be image data (3+ dimensions)
#                     if len(dataset.shape) >= 3 and dataset.size > 1000:
#                         return {
#                             'type': 'alternative_dataset',
#                             'dataset_name': key,
#                             'frame_count': dataset.shape[0],
#                             'frame_shape': dataset.shape[1:],
#                             'dtype': dataset.dtype,
#                             'dtype_size': dataset.dtype.itemsize,
#                             'data_location': key
#                         }

#             return {
#                 'type': 'unknown',
#                 'error': f'No suitable image data found in {root_keys}',
#                 'available_keys': root_keys
#             }

#     except Exception as e:
#         return {
#             'type': 'error',
#             'error': str(e)
#         }

# def get_first_frame_enhanced(file_path: str, driver: Optional[str] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
#     """
#     Enhanced first frame getter with automatic structure detection.

#     Returns:
#         Tuple of (first_frame, structure_info)
#     """
#     if not H5PY_AVAILABLE:
#         return None, {'type': 'error', 'error': 'h5py not available'}

#     try:
#         if not os.path.exists(file_path):
#             logger.error(f"File does not exist: {file_path}")
#             return None, {'type': 'error', 'error': 'File not found'}

#         structure_info = detect_hdf5_structure_type(file_path)

#         if structure_info['type'] == 'error':
#             logger.error(f"Structure detection failed: {structure_info['error']}")
#             return None, structure_info

#         with h5py.File(file_path, 'r', **(dict(driver=driver) if driver else {})) as f:
#             logger.info(f"HDF5 file structure: {list(f.keys())}")

#             if structure_info['type'] == 'stacked_frames':
#                 # Handle stacked frames (original structure)
#                 frames_dataset = f['frames']
#                 first_frame = frames_dataset[0].copy()
#                 logger.info(f"Loaded first frame from stacked dataset: {first_frame.shape}")

#             elif structure_info['type'] == 'individual_frames':
#                 # Handle individual frames in images/ group
#                 images_group = f['images']
#                 frame_keys = structure_info['frame_keys']
#                 first_key = frame_keys[0]
#                 first_frame = images_group[first_key][...].copy()
#                 logger.info(f"Loaded first frame from images group: key={first_key}, shape={first_frame.shape}")

#             elif structure_info['type'] == 'alternative_dataset':
#                 # Handle alternative dataset structure
#                 dataset_name = structure_info['dataset_name']
#                 dataset = f[dataset_name]
#                 first_frame = dataset[0].copy()
#                 logger.info(f"Loaded first frame from alternative dataset '{dataset_name}': {first_frame.shape}")

#             else:
#                 logger.error(f"Unknown structure type: {structure_info['type']}")
#                 return None, structure_info

#             return first_frame, structure_info

#     except Exception as e:
#         logger.error(f"Error reading first frame: {e}")
#         return None, {'type': 'error', 'error': str(e)}

# def read_chunk_data_dual_structure(file_path: str, start_idx: int, end_idx: int,
#                                  structure_info: Dict[str, Any], driver: Optional[str] = None) -> Optional[np.ndarray]:
#     """
#     Read a chunk of data handling both stacked and individual frame structures.
#     """
#     try:
#         with h5py.File(file_path, 'r', **(dict(driver=driver) if driver else {})) as f:

#             if structure_info['type'] == 'stacked_frames':
#                 # Standard stacked frames approach
#                 frames_dataset = f['frames']
#                 chunk_data = frames_dataset[start_idx:end_idx].copy()

#             elif structure_info['type'] == 'individual_frames':
#                 # Read individual frames from images/ group
#                 images_group = f['images']
#                 frame_keys = structure_info['frame_keys']

#                 # Get the keys for this chunk
#                 chunk_keys = frame_keys[start_idx:end_idx]

#                 # Read each frame individually and stack them
#                 frames = []
#                 for key in chunk_keys:
#                     frame = images_group[key][...].copy()
#                     frames.append(frame)

#                 # Stack into array format expected by processing
#                 if frames:
#                     chunk_data = np.stack(frames, axis=0)
#                 else:
#                     return None

#             elif structure_info['type'] == 'alternative_dataset':
#                 # Handle alternative dataset
#                 dataset = f[structure_info['dataset_name']]
#                 chunk_data = dataset[start_idx:end_idx].copy()

#             else:
#                 logger.error(f"Cannot read chunk for structure type: {structure_info['type']}")
#                 return None

#             logger.debug(f"Read chunk {start_idx}:{end_idx}, shape: {chunk_data.shape}")
#             return chunk_data

#     except Exception as e:
#         logger.error(f"Error reading chunk {start_idx}:{end_idx}: {e}")
#         return None

# # =============================================================================
# # PYTHON-NATIVE NORMALIZATION FUNCTIONS
# # =============================================================================

# def normalize_image_to_float32(image: np.ndarray,
#                               target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
#     """
#     Python-native image normalization.

#     Args:
#         image: Input image (uint8, uint16, or float)
#         target_range: Target range for normalized values

#     Returns:
#         Normalized float32 array
#     """
#     # Convert to float32 for processing
#     if image.dtype == np.uint8:
#         # Standard 8-bit images: 0-255 → 0.0-1.0
#         normalized = image.astype(np.float32) / 255.0
#     elif image.dtype == np.uint16:
#         # 16-bit images: 0-65535 → 0.0-1.0
#         normalized = image.astype(np.float32) / 65535.0
#     elif image.dtype in [np.int16, np.int32]:
#         # Signed integers: normalize to actual range
#         img_float = image.astype(np.float32)
#         img_min, img_max = img_float.min(), img_float.max()
#         if img_max > img_min:
#             normalized = (img_float - img_min) / (img_max - img_min)
#         else:
#             normalized = np.zeros_like(img_float)
#     elif image.dtype in [np.float32, np.float64]:
#         # Already float - check if needs rescaling
#         img_min, img_max = image.min(), image.max()
#         if img_max > 1.0 or img_min < 0.0:
#             # Rescale to 0-1 if outside standard range
#             if img_max > img_min:
#                 normalized = ((image.astype(np.float32) - img_min) / (img_max - img_min))
#             else:
#                 normalized = np.zeros_like(image, dtype=np.float32)
#         else:
#             normalized = image.astype(np.float32)
#     else:
#         raise ValueError(f"Unsupported image dtype: {image.dtype}")

#     # Adjust to target range if different from [0,1]
#     if target_range != (0.0, 1.0):
#         target_min, target_max = target_range
#         normalized = normalized * (target_max - target_min) + target_min

#     return normalized

# def validate_data_ranges(data: np.ndarray, stage: str, roi_idx: Optional[int] = None):
#     """
#     Validate data ranges at different processing stages.
#     """
#     roi_info = f" ROI {roi_idx}" if roi_idx is not None else ""
#     logger.debug(f"=== DATA VALIDATION{roi_info}: {stage} ===")
#     logger.debug(f"  Dtype: {data.dtype}")
#     logger.debug(f"  Shape: {data.shape}")
#     logger.debug(f"  Range: {data.min():.6f} to {data.max():.6f}")
#     logger.debug(f"  Mean: {data.mean():.6f}")
#     logger.debug(f"  Std: {data.std():.6f}")

#     # Validation checks
#     if stage == "Raw_Input":
#         if data.dtype == np.uint8 and (data.min() < 0 or data.max() > 255):
#             logger.warning(f"  ❌ uint8 values outside expected range!")
#         elif data.dtype == np.uint16 and (data.min() < 0 or data.max() > 65535):
#             logger.warning(f"  ❌ uint16 values outside expected range!")
#         else:
#             logger.debug(f"  ✅ Raw input range looks correct")

#     elif stage == "Normalized_Output":
#         if data.dtype != np.float32:
#             logger.warning(f"  ❌ Expected float32 output, got {data.dtype}!")
#         elif data.min() < -0.1 or data.max() > 1.1:
#             logger.warning(f"  ❌ Normalized values outside [0,1] range!")
#         else:
#             logger.debug(f"  ✅ Normalization successful")

# # =============================================================================
# # NAPARI READER FUNCTIONS
# # =============================================================================

# def napari_get_reader(path: Union[str, List[str]]) -> Optional[Callable]:
#     """
#     Returns a reader function if `path` is a valid HDF5 file or a
#     directory containing HDF5 files. Otherwise returns None.
#     """
#     if not H5PY_AVAILABLE:
#         logger.error("h5py not available. Cannot read HDF5 files.")
#         return None

#     if os.path.isdir(path):
#         return reader_directory_function
#     elif (
#         (isinstance(path, str) and path.lower().endswith(('.h5', '.hdf5')))
#         or (
#             isinstance(path, list)
#             and all(isinstance(p, str) and p.lower().endswith(('.h5', '.hdf5')) for p in path)
#         )
#     ):
#         return reader_function_dual_structure

#     return None

# def reader_function_dual_structure(
#     path: Union[str, List[str]],
#     driver: Optional[str] = None
# ) -> List[Tuple]:
#     """
#     Enhanced reader function with automatic HDF5 structure detection.
#     Handles both stacked frames and individual frames in images/ group.
#     """
#     if not H5PY_AVAILABLE:
#         logger.error("h5py not available")
#         return []

#     # Handle list input
#     if isinstance(path, list) and len(path) == 1:
#         path = path[0]
#     if isinstance(path, list):
#         return reader_directory_function(os.path.dirname(path[0]), filenames=path, driver=driver)

#     try:
#         # Check file exists and is readable
#         if not os.path.exists(path):
#             logger.error(f"File does not exist: {path}")
#             return []

#         if not os.access(path, os.R_OK):
#             logger.error(f"File is not readable: {path}")
#             return []

#         # Get first frame and detect structure
#         first_frame, structure_info = get_first_frame_enhanced(path, driver=driver)
#         if first_frame is None:
#             logger.error(f"Could not read first frame from {path}")
#             if structure_info.get('error'):
#                 logger.error(f"Structure detection error: {structure_info['error']}")
#             return []

#         logger.info(f"Successfully loaded {structure_info['type']} structure from {path}")

#     except Exception as e:
#         logger.error(f"Error reading HDF5 file: {e}")
#         return []

#     layers: List[Tuple] = []

#     # Add the raw first frame with structure information
#     layers.append(
#         (
#             first_frame,
#             {
#                 "name": os.path.basename(path),
#                 "metadata": {
#                     "path": path,
#                     "structure_type": structure_info['type'],
#                     "frame_count": structure_info['frame_count'],
#                     "frame_shape": structure_info['frame_shape'],
#                     "dtype_size": structure_info['dtype_size'],
#                     "data_location": structure_info['data_location'],
#                     "structure_info": structure_info,  # Pass complete info for processing
#                     "is_hdf5": True,
#                     "hdf5_file_path": path,
#                     "optimal_chunk_size": calculate_optimal_chunk_size(
#                         structure_info['frame_shape'],
#                         structure_info['dtype_size']
#                     ),
#                     "processing_style": "python_native",
#                 },
#             },
#             "image",
#         )
#     )

#     return layers

# # LEGACY FUNCTIONS FOR BACKWARD COMPATIBILITY

# def get_first_frame(path: str, driver: Optional[str] = None) -> Optional[np.ndarray]:
#     """
#     Legacy function - now uses enhanced detection.
#     """
#     first_frame, structure_info = get_first_frame_enhanced(path, driver)
#     return first_frame

# def reader_directory_function(
#     path: str,
#     filenames: Optional[List[str]] = None,
#     driver: Optional[str] = None
# ) -> List[Tuple]:
#     """
#     Enhanced directory reader with dual structure support.
#     """
#     if not H5PY_AVAILABLE:
#         logger.error("h5py not available")
#         return []

#     if filenames is None:
#         try:
#             if not os.path.exists(path):
#                 logger.error(f"Directory does not exist: {path}")
#                 return []

#             filenames = [
#                 os.path.join(path, f)
#                 for f in os.listdir(path)
#                 if f.lower().endswith(('.h5', '.hdf5')) and os.path.isfile(os.path.join(path, f))
#             ]
#         except OSError as e:
#             logger.error(f"Error reading directory {path}: {e}")
#             return []

#     if not filenames:
#         logger.warning(f"No HDF5 files found in {path}")
#         return []

#     layers: List[Tuple] = []
#     first_file_path = filenames[0]

#     # Use enhanced structure detection
#     try:
#         first_frame, structure_info = get_first_frame_enhanced(first_file_path, driver)

#         if first_frame is None:
#             logger.error(f"Could not read first frame from {first_file_path}")
#             return []

#         logger.info(f"Directory processing: detected {structure_info['type']} structure")

#         # Get metadata for memory management
#         frame_shape = structure_info['frame_shape']
#         dtype_size = structure_info['dtype_size']

#     except Exception as e:
#         logger.error(f"Error reading first HDF5 file: {e}")
#         return []

#     # Detect ROIs only if OpenCV is available
#     if CV2_AVAILABLE:
#         masks, labeled_frame = detect_circles_and_create_masks(
#             first_frame,
#             min_radius=80,
#             max_radius=150,
#             dp=0.5,
#             min_dist=150,
#             param1=40,
#             param2=40,
#         )
#     else:
#         logger.warning("OpenCV not available. Skipping ROI detection.")
#         masks = []
#         labeled_frame = first_frame

#     # Add overview image
#     layers.append(
#         (
#             labeled_frame,
#             {
#                 "name": "Detected ROIs" if CV2_AVAILABLE else "First Frame",
#                 "metadata": {
#                     "path": first_file_path,
#                     "detected_rois": len(masks),
#                     "total_files": len(filenames),
#                     "roi_detection_available": CV2_AVAILABLE,
#                     "frame_shape": frame_shape,
#                     "dtype_size": dtype_size,
#                     "structure_type": structure_info['type'],
#                     "optimal_chunk_size": calculate_optimal_chunk_size(frame_shape, dtype_size),
#                     "processing_style": "python_native",
#                 },
#             },
#             "image",
#         )
#     )

#     # Add preview of each file with memory-aware processing
#     for filename in filenames[:10]:  # Limit previews to prevent memory issues
#         try:
#             preview_frame, preview_structure = get_first_frame_enhanced(filename, driver)

#             if preview_frame is not None:
#                 layers.append(
#                     (
#                         preview_frame,
#                         {
#                             "name": os.path.basename(filename),
#                             "metadata": {
#                                 "path": filename,
#                                 "frame_count": preview_structure['frame_count'],
#                                 "structure_type": preview_structure['type'],
#                                 "is_hdf5": True,
#                                 "processing_style": "python_native",
#                             },
#                         },
#                         "image",
#                     )
#                 )
#             else:
#                 logger.warning(f"No frames found in {filename}")
#         except Exception as e:
#             logger.error(f"Error reading HDF5 file {filename}: {e}")

#     # Add ROI mask layers if detection was successful
#     if CV2_AVAILABLE and masks:
#         for i, mask in enumerate(masks):
#             layers.append((mask, {"name": f"ROI {i+1} Mask", "visible": False}, "labels"))

#     return layers

# def sort_circles_left_to_right(circles: np.ndarray) -> np.ndarray:
#     """
#     Sort detected circles from left to right based on x-coordinate only.
#     """
#     if circles is None or len(circles) == 0:
#         return np.array([])

#     # Extract circle data
#     circles_list = [(circle[0], circle[1], circle[2]) for circle in circles[0]]

#     # Sort based on x-coordinate only (left to right)
#     sorted_circles = sorted(circles_list, key=lambda circle: circle[0])

#     # Convert back to numpy array format
#     return np.array(sorted_circles)

# def sort_circles_meandering_auto(circles: np.ndarray) -> np.ndarray:
#     """
#     Automatically sort circles in meandering pattern based on detected count.
#     Supports 6-well (2x3), 12-well (3x4), and 24-well (4x6) plates.

#     Pattern:
#     Row 1: 1→2→3→4
#     Row 2: 8←7←6←5
#     Row 3: 9→10→11→12
#     """
#     if circles is None or len(circles) == 0:
#         return circles

#     # Remove extra dimension from HoughCircles output
#     if len(circles.shape) == 3:
#         circles = circles[0]

#     num_circles = len(circles)

#     # Auto-detect plate layout based on circle count
#     if num_circles == 6:
#         rows, cols = 2, 3  # 6-well plate
#     elif num_circles == 12:
#         rows, cols = 3, 4  # 12-well plate
#     elif num_circles == 24:
#         rows, cols = 4, 6  # 24-well plate
#     elif num_circles == 4:
#         rows, cols = 2, 2  # 4-well
#     elif num_circles == 8:
#         rows, cols = 2, 4  # 8-well
#     elif num_circles == 16:
#         rows, cols = 4, 4  # 16-well
#     else:
#         # For other counts, fall back to simple left-to-right sorting
#         return sort_circles_left_to_right_simple(circles)

#     # Group circles into rows based on Y coordinates
#     circle_rows = _group_into_rows(circles, rows)

#     # Apply meandering pattern
#     sorted_circles = []
#     for row_idx, row_circles in enumerate(circle_rows):
#         if len(row_circles) == 0:
#             continue

#         # Sort current row by X coordinate (left to right)
#         row_sorted = sorted(row_circles, key=lambda c: c[0])

#         # Reverse every odd row (0-indexed) for meandering pattern
#         if row_idx % 2 == 1:
#             row_sorted = row_sorted[::-1]

#         sorted_circles.extend(row_sorted)

#     return np.array(sorted_circles, dtype=np.uint16)


# def _group_into_rows(circles: np.ndarray, expected_rows: int) -> list:
#     """Group circles into rows based on Y coordinates."""
#     if len(circles) == 0:
#         return []

#     # Sort all circles by Y coordinate
#     y_sorted_indices = np.argsort(circles[:, 1])
#     y_sorted_circles = circles[y_sorted_indices]

#     if expected_rows == 1:
#         return [y_sorted_circles.tolist()]

#     # Divide circles into expected number of rows
#     circles_per_row = len(circles) // expected_rows
#     rows = []

#     for i in range(expected_rows):
#         start_idx = i * circles_per_row
#         end_idx = start_idx + circles_per_row

#         # Handle last row (include remaining circles)
#         if i == expected_rows - 1:
#             end_idx = len(y_sorted_circles)

#         row_circles = y_sorted_circles[start_idx:end_idx].tolist()
#         rows.append(row_circles)

#     return rows


# def sort_circles_left_to_right_simple(circles: np.ndarray) -> np.ndarray:
#     """Simple left-to-right sorting fallback."""
#     if circles is None or len(circles) == 0:
#         return circles

#     if len(circles.shape) == 3:
#         circles = circles[0]

#     # Sort by X coordinate only
#     sorted_indices = np.argsort(circles[:, 0])
#     return circles[sorted_indices]


# def detect_circles_and_create_masks(frame: np.ndarray,
#                                   min_radius: int = 80,
#                                   max_radius: int = 150,
#                                   dp: float = 0.5,
#                                   min_dist: int = 150,
#                                   param1: int = 40,
#                                   param2: int = 40) -> Tuple[List[np.ndarray], np.ndarray]:
#     """
#     Enhanced circle detection with automatic meandering sorting for multi-well plates.
#     """
#     if not CV2_AVAILABLE:
#         logger.error("OpenCV not available for circle detection")
#         return [], frame if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)

#     if frame is None:
#         logger.error("Input frame is None")
#         return [], np.zeros((100, 100, 3), dtype=np.uint8)

#     try:
#         # Validate frame
#         if frame.size == 0:
#             logger.error("Input frame is empty")
#             return [], frame

#         # Convert to grayscale if needed (memory efficient)
#         if len(frame.shape) == 3:
#             # Use memory-efficient conversion
#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#         else:
#             gray_frame = frame.copy()

#         # Validate grayscale frame
#         if gray_frame.size == 0:
#             logger.error("Grayscale conversion resulted in empty frame")
#             return [], frame

#         # Apply CLAHE enhancement with error handling
#         try:
#             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#             enhanced_frame = clahe.apply(gray_frame)
#         except Exception as e:
#             logger.warning(f"CLAHE enhancement failed: {e}. Using original frame.")
#             enhanced_frame = gray_frame

#         # Detect circles with validation
#         circles = cv2.HoughCircles(
#             enhanced_frame,
#             cv2.HOUGH_GRADIENT,
#             dp=dp,
#             minDist=min_dist,
#             param1=param1,
#             param2=param2,
#             minRadius=min_radius,
#             maxRadius=max_radius
#         )

#         masks = []

#         # Create labeled frame for visualization
#         if len(frame.shape) == 3:
#             labeled_frame = frame.copy()
#         else:
#             labeled_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

#         if circles is not None and len(circles[0]) > 0:
#             circles = np.uint16(np.around(circles))

#             # NEW: Apply automatic meandering sorting
#             sorted_circles = sort_circles_meandering_auto(circles)

#             # Log the sorting pattern used
#             num_circles = len(sorted_circles)
#             if num_circles in [6, 12, 24, 4, 8, 16]:
#                 if num_circles == 6:
#                     pattern_info = "6-well plate (2x3) meandering"
#                 elif num_circles == 12:
#                     pattern_info = "12-well plate (3x4) meandering"
#                 elif num_circles == 24:
#                     pattern_info = "24-well plate (4x6) meandering"
#                 elif num_circles == 4:
#                     pattern_info = "4-well (2x2) meandering"
#                 elif num_circles == 8:
#                     pattern_info = "8-well (2x4) meandering"
#                 elif num_circles == 16:
#                     pattern_info = "16-well (4x4) meandering"
#                 logger.info(f"Applied {pattern_info} sorting to {num_circles} ROIs")
#             else:
#                 logger.info(f"Applied simple left-to-right sorting to {num_circles} ROIs")

#             for idx, circle in enumerate(sorted_circles):
#                 try:
#                     # Validate circle parameters
#                     x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
#                     if x < 0 or y < 0 or r <= 0:
#                         logger.warning(f"Invalid circle parameters: ({x}, {y}, {r})")
#                         continue

#                     # Check if circle is within frame bounds
#                     h, w = gray_frame.shape
#                     if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
#                         logger.warning(f"Circle {idx} extends beyond frame bounds")
#                         # Adjust circle to fit within bounds
#                         x = max(r, min(w - r - 1, x))
#                         y = max(r, min(h - r - 1, y))

#                     # Create mask for this ROI
#                     mask = np.zeros(gray_frame.shape, dtype=np.uint8)
#                     cv2.circle(mask, (x, y), r, 255, thickness=-1)
#                     masks.append(mask)

#                     # Draw circle on the labeled frame with enhanced visibility
#                     cv2.circle(labeled_frame, (x, y), r, (0, 255, 0), 2)

#                     # Add ROI number with better visibility
#                     text = str(idx + 1)
#                     font_scale = 1.5
#                     thickness = 3

#                     # Get text size for centering
#                     (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

#                     # Add white background for better readability
#                     cv2.rectangle(labeled_frame,
#                                  (x - text_width//2 - 5, y - text_height//2 - 5),
#                                  (x + text_width//2 + 5, y + text_height//2 + 5),
#                                  (255, 255, 255), -1)

#                     # Add black text
#                     cv2.putText(labeled_frame, text,
#                                 (x - text_width//2, y + text_height//2),
#                                 cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

#                 except Exception as e:
#                     logger.error(f"Error processing circle {idx}: {e}")
#                     continue
#         else:
#             logger.warning("No circles detected with current parameters")

#     except Exception as e:
#         logger.error(f"Error in circle detection: {e}")
#         labeled_frame = frame.copy() if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)

#     return masks, labeled_frame
# # def detect_circles_and_create_masks(frame: np.ndarray,
# #                                   min_radius: int = 80,
# #                                   max_radius: int = 150,
# #                                   dp: float = 0.5,
# #                                   min_dist: int = 150,
# #                                   param1: int = 40,
# #                                   param2: int = 40) -> Tuple[List[np.ndarray], np.ndarray]:
# #     """
# #     Enhanced circle detection with better error handling and memory management.
# #     """
# #     if not CV2_AVAILABLE:
# #         logger.error("OpenCV not available for circle detection")
# #         return [], frame if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)

# #     if frame is None:
# #         logger.error("Input frame is None")
# #         return [], np.zeros((100, 100, 3), dtype=np.uint8)

# #     try:
# #         # Validate frame
# #         if frame.size == 0:
# #             logger.error("Input frame is empty")
# #             return [], frame

# #         # Convert to grayscale if needed (memory efficient)
# #         if len(frame.shape) == 3:
# #             # Use memory-efficient conversion
# #             gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
# #         else:
# #             gray_frame = frame.copy()

# #         # Validate grayscale frame
# #         if gray_frame.size == 0:
# #             logger.error("Grayscale conversion resulted in empty frame")
# #             return [], frame

# #         # Apply CLAHE enhancement with error handling
# #         try:
# #             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# #             enhanced_frame = clahe.apply(gray_frame)
# #         except Exception as e:
# #             logger.warning(f"CLAHE enhancement failed: {e}. Using original frame.")
# #             enhanced_frame = gray_frame

# #         # Detect circles with validation
# #         circles = cv2.HoughCircles(
# #             enhanced_frame,
# #             cv2.HOUGH_GRADIENT,
# #             dp=dp,
# #             minDist=min_dist,
# #             param1=param1,
# #             param2=param2,
# #             minRadius=min_radius,
# #             maxRadius=max_radius
# #         )

# #         masks = []

# #         # Create labeled frame for visualization
# #         if len(frame.shape) == 3:
# #             labeled_frame = frame.copy()
# #         else:
# #             labeled_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

# #         if circles is not None and len(circles[0]) > 0:
# #             circles = np.uint16(np.around(circles))
# #             sorted_circles = sort_circles_left_to_right(circles)

# #             logger.info(f"Detected {len(sorted_circles)} circles")

# #             for idx, circle in enumerate(sorted_circles):
# #                 try:
# #                     # Validate circle parameters
# #                     x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
# #                     if x < 0 or y < 0 or r <= 0:
# #                         logger.warning(f"Invalid circle parameters: ({x}, {y}, {r})")
# #                         continue

# #                     # Check if circle is within frame bounds
# #                     h, w = gray_frame.shape
# #                     if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
# #                         logger.warning(f"Circle {idx} extends beyond frame bounds")
# #                         # Adjust circle to fit within bounds
# #                         x = max(r, min(w - r - 1, x))
# #                         y = max(r, min(h - r - 1, y))

# #                     # Create mask for this ROI
# #                     mask = np.zeros(gray_frame.shape, dtype=np.uint8)
# #                     cv2.circle(mask, (x, y), r, 255, thickness=-1)
# #                     masks.append(mask)

# #                     # Draw circle on the labeled frame
# #                     cv2.circle(labeled_frame, (x, y), r, (0, 255, 0), 2)

# #                     # Add ROI number
# #                     cv2.putText(labeled_frame, f"{idx + 1}",
# #                                 (x - 20, y + 10),
# #                                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
# #                 except Exception as e:
# #                     logger.error(f"Error processing circle {idx}: {e}")
# #                     continue
# #         else:
# #             logger.warning("No circles detected with current parameters")

# #     except Exception as e:
# #         logger.error(f"Error in circle detection: {e}")
# #         labeled_frame = frame.copy() if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)

# #     return masks, labeled_frame

# # =============================================================================
# # CHUNK PROCESSING FUNCTIONS
# # =============================================================================

# def process_chunk(chunk_data: np.ndarray, masks: List[np.ndarray],
#                  start_time: float, frame_interval: float = 5) -> Dict[int, List[Tuple[float, float]]]:
#     """
#     Enhanced chunk processing that handles different HDF5 structures.
#     """
#     roi_changes = {roi_idx + 1: [] for roi_idx in range(len(masks))}

#     if len(chunk_data) < 2:
#         return roi_changes

#     try:
#         logger.debug(f"Processing chunk shape: {chunk_data.shape}, dtype: {chunk_data.dtype}")
#         validate_data_ranges(chunk_data[0], "Raw_Input")

#         # === STEP 1: Grayscale Conversion ===
#         if len(chunk_data.shape) == 4 and chunk_data.shape[-1] == 3:
#             # RGB to grayscale conversion
#             batch_size = min(10, len(chunk_data))
#             gray_frames = []

#             for i in range(0, len(chunk_data), batch_size):
#                 end_idx = min(i + batch_size, len(chunk_data))
#                 batch = chunk_data[i:end_idx]

#                 if CV2_AVAILABLE:
#                     gray_batch = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in batch])
#                 else:
#                     weights = np.array([0.299, 0.587, 0.114])
#                     gray_batch = np.dot(batch, weights).astype(np.uint8)

#                 gray_frames.append(gray_batch)

#             gray_frames = np.concatenate(gray_frames, axis=0)
#         else:
#             gray_frames = chunk_data.copy()

#         # Continue with rest of processing logic...
#         logger.debug(f"Pre-normalization: dtype={gray_frames.dtype}, range={gray_frames.min()}-{gray_frames.max()}")

#         normalized_frames = normalize_image_to_float32(gray_frames, target_range=(0.0, 1.0))

#         logger.debug(f"Post-normalization: dtype={normalized_frames.dtype}, range={normalized_frames.min():.6f}-{normalized_frames.max():.6f}")
#         validate_data_ranges(normalized_frames[0], "Normalized_Output")

#         # === STEP 3: ROI Processing ===
#         for roi_idx, mask in enumerate(masks, start=1):
#             try:
#                 if mask.size == 0:
#                     logger.warning(f"Empty mask for ROI {roi_idx}")
#                     continue

#                 mask_bool = mask > 0

#                 if mask_bool.shape != normalized_frames.shape[1:]:
#                     logger.error(f"Mask shape {mask_bool.shape} != frame shape {normalized_frames.shape[1:]}")
#                     continue

#                 roi_intensities = []

#                 # Memory-efficient processing
#                 if normalized_frames.nbytes > 500 * 1024 * 1024:  # 500MB threshold
#                     batch_size = max(1, min(50, len(normalized_frames) - 1))

#                     for batch_start in range(0, len(normalized_frames) - 1, batch_size):
#                         batch_end = min(batch_start + batch_size + 1, len(normalized_frames))
#                         batch_frames = normalized_frames[batch_start:batch_end]

#                         for i in range(len(batch_frames) - 1):
#                             frame_curr = batch_frames[i]
#                             frame_next = batch_frames[i + 1]

#                             diff_masked = np.abs(frame_next[mask_bool] - frame_curr[mask_bool])
#                             total_intensity = np.sum(diff_masked)
#                             roi_intensities.append(total_intensity)
#                 else:
#                     for i in range(len(normalized_frames) - 1):
#                         frame_curr = normalized_frames[i]
#                         frame_next = normalized_frames[i + 1]

#                         diff_masked = np.abs(frame_next[mask_bool] - frame_curr[mask_bool])
#                         total_intensity = np.sum(diff_masked)
#                         roi_intensities.append(total_intensity)

#                 time_array = start_time + frame_interval * np.arange(len(roi_intensities))
#                 roi_changes[roi_idx] = list(zip(time_array, roi_intensities))

#                 if roi_idx == 1 and roi_intensities:
#                     roi_area = np.sum(mask_bool)
#                     logger.info(f"Enhanced ROI {roi_idx}: area={roi_area:,} pixels, "
#                               f"intensity_range={np.min(roi_intensities):.6f}-{np.max(roi_intensities):.6f}")

#             except Exception as e:
#                 logger.error(f"Error processing ROI {roi_idx}: {e}")
#                 continue

#     except Exception as e:
#         logger.error(f"Error processing chunk: {e}")

#     return roi_changes

# # =============================================================================
# # PARALLEL PROCESSING FUNCTIONS WITH DUAL STRUCTURE SUPPORT
# # =============================================================================

# def _process_single_chunk_dual_structure(args: Tuple) -> Tuple[int, Dict[int, List[Tuple[float, float]]]]:
#     """
#     Enhanced chunk processing that handles both HDF5 structures.
#     """
#     try:
#         file_path, masks, start_idx, end_idx, frame_interval, structure_info = args

#         # Read chunk using dual structure reader
#         chunk = read_chunk_data_dual_structure(file_path, start_idx, end_idx, structure_info)

#         if chunk is None:
#             logger.error(f"Failed to read chunk {start_idx}:{end_idx}")
#             return start_idx, {roi_idx + 1: [] for roi_idx in range(len(masks))}

#         # Process chunk with existing logic
#         chunk_results = process_chunk(chunk, masks, start_idx * frame_interval, frame_interval)
#         return start_idx, chunk_results

#     except Exception as e:
#         logger.error(f"Error processing dual-structure chunk {start_idx}-{end_idx}: {e}")
#         return start_idx, {roi_idx + 1: [] for roi_idx in range(len(masks))}

# def process_single_file_in_parallel_dual_structure(file_path: str, masks: List[np.ndarray],
#                                                   chunk_size: int = 50,
#                                                   progress_callback: Optional[Callable] = None,
#                                                   frame_interval: float = 5,
#                                                   num_processes: Optional[int] = None) -> Tuple[str, Dict, float]:
#     """
#     Process a single large HDF5 file using dual-structure support.
#     """
#     start_all = time.time()

#     # Detect structure
#     structure_info = detect_hdf5_structure_type(file_path)
#     if structure_info['type'] == 'error':
#         logger.error(f"Cannot process file: {structure_info['error']}")
#         return file_path, {}, 0.0

#     logger.info(f"Processing {structure_info['type']} structure: {file_path}")
#     logger.info(f"Frame count: {structure_info['frame_count']}, Shape: {structure_info['frame_shape']}")

#     num_frames = structure_info['frame_count']
#     if num_frames == 0:
#         logger.error(f"No frames found in {file_path}")
#         return file_path, {}, 0.0

#     # Calculate optimal chunk size
#     frame_shape = structure_info['frame_shape']
#     dtype_size = structure_info['dtype_size']
#     optimal_chunk_size = calculate_optimal_chunk_size(frame_shape, dtype_size)
#     actual_chunk_size = max(MIN_CHUNK_SIZE, min(chunk_size, optimal_chunk_size))

#     logger.info(f"Using chunk size: {actual_chunk_size} frames (structure: {structure_info['type']})")

#     total_chunks = (num_frames + actual_chunk_size - 1) // actual_chunk_size
#     roi_changes = {roi_idx + 1: [] for roi_idx in range(len(masks))}

#     if num_processes is None:
#         if os.name == 'nt':  # Windows
#             num_processes = max(1, min(4, int(os.cpu_count() * 0.75)))
#         else:
#             num_processes = max(1, min(6, int(os.cpu_count() * 0.85)))

#     # Prepare tasks with structure information
#     tasks = []
#     for start_idx in range(0, num_frames, actual_chunk_size):
#         end_idx = min(start_idx + actual_chunk_size, num_frames)
#         tasks.append((file_path, masks, start_idx, end_idx, frame_interval, structure_info))

#     completed = 0

#     try:
#         with ProcessPoolExecutor(max_workers=num_processes) as executor:
#             futures = [executor.submit(_process_single_chunk_dual_structure, task) for task in tasks]

#             for future in as_completed(futures):
#                 try:
#                     start_idx, chunk_res = future.result(timeout=600)

#                     # Merge results
#                     for roi_idx in chunk_res:
#                         roi_changes[roi_idx].extend(chunk_res[roi_idx])

#                     completed += 1

#                     if progress_callback:
#                         percent = (completed / total_chunks) * 100
#                         msg = f"Dual-structure chunk {completed}/{total_chunks} for {os.path.basename(file_path)}"
#                         progress_callback(percent, msg)

#                 except Exception as e:
#                     logger.error(f"Error processing dual-structure chunk: {e}")
#                     completed += 1

#     except Exception as e:
#         logger.error(f"ProcessPoolExecutor error (dual-structure): {e}")
#         # Fallback to single-threaded processing
#         logger.info("Falling back to single-threaded dual-structure processing")
#         return process_hdf5_file_dual_structure(file_path, masks, actual_chunk_size, progress_callback, frame_interval)

#     # Sort results by time
#     for roi_idx in roi_changes:
#         roi_changes[roi_idx].sort(key=lambda x: x[0])

#     total_duration = (num_frames - 1) * frame_interval
#     proc_time = time.time() - start_all
#     logger.info(f"Dual-structure parallel processing complete: {file_path} processed in {proc_time:.2f}s")

#     if progress_callback:
#         progress_callback(100, f"Completed dual-structure processing {os.path.basename(file_path)}")

#     return file_path, roi_changes, total_duration

# def process_hdf5_file_dual_structure(file_path: str, masks: List[np.ndarray],
#                                     chunk_size: int = 50,
#                                     progress_callback: Optional[Callable] = None,
#                                     frame_interval: float = 5) -> Tuple[str, Dict, float]:
#     """
#     Process a single HDF5 file with dual structure support (single-threaded).
#     """
#     start_all = time.time()

#     # Detect structure
#     structure_info = detect_hdf5_structure_type(file_path)
#     if structure_info['type'] == 'error':
#         logger.error(f"Cannot process file: {structure_info['error']}")
#         return file_path, {}, 0.0

#     logger.info(f"Processing {structure_info['type']} structure: {file_path}")

#     num_frames = structure_info['frame_count']
#     if num_frames == 0:
#         logger.error(f"No frames found in {file_path}")
#         return file_path, {}, 0.0

#     # Calculate optimal chunk size
#     frame_shape = structure_info['frame_shape']
#     dtype_size = structure_info['dtype_size']
#     optimal_chunk_size = calculate_optimal_chunk_size(frame_shape, dtype_size)
#     actual_chunk_size = max(MIN_CHUNK_SIZE, min(chunk_size, optimal_chunk_size))

#     total_chunks = (num_frames + actual_chunk_size - 1) // actual_chunk_size
#     roi_changes = {roi_idx + 1: [] for roi_idx in range(len(masks))}

#     last_frame = None

#     for chunk_idx, start_idx in enumerate(range(0, num_frames, actual_chunk_size)):
#         chunk_start = time.time()
#         end_idx = min(start_idx + actual_chunk_size, num_frames)

#         # Define msg outside the if block so it's always available
#         msg = f"Dual-structure processing frames {start_idx} to {end_idx} ({chunk_idx+1}/{total_chunks} chunks)"

#         if progress_callback:
#             percent = (chunk_idx / total_chunks) * 100
#             progress_callback(percent, msg)

#         logger.info(msg)  # Now msg is always defined

#         # Read chunk using dual structure method
#         try:
#             this_chunk = read_chunk_data_dual_structure(file_path, start_idx, end_idx, structure_info)

#             if this_chunk is None:
#                 logger.error(f"Failed to read chunk {chunk_idx}")
#                 continue

#         except Exception as e:
#             logger.error(f"Error reading dual-structure chunk {chunk_idx}: {e}")
#             continue

#         # Prepend last frame for boundary diff if available
#         if last_frame is not None:
#             try:
#                 chunk = np.concatenate([last_frame[np.newaxis], this_chunk], axis=0)
#                 chunk_start_time = (start_idx - 1) * frame_interval
#             except Exception as e:
#                 logger.warning(f"Error combining chunks: {e}")
#                 chunk = this_chunk
#                 chunk_start_time = start_idx * frame_interval
#         else:
#             chunk = this_chunk
#             chunk_start_time = start_idx * frame_interval

#         # Process chunk
#         try:
#             chunk_res = process_chunk(chunk, masks, chunk_start_time, frame_interval)
#             for roi_idx in chunk_res:
#                 roi_changes[roi_idx].extend(chunk_res[roi_idx])
#         except Exception as e:
#             logger.error(f"Error processing dual-structure chunk {chunk_idx}: {e}")
#             continue

#         # Update last_frame for next iteration
#         try:
#             last_frame = this_chunk[-1].copy()
#         except Exception as e:
#             logger.warning(f"Error saving last frame: {e}")
#             last_frame = None

#         # Log chunk performance
#         chunk_time = time.time() - chunk_start
#         fps = (end_idx - start_idx) / chunk_time if chunk_time > 0 else 0
#         logger.info(f"Dual-structure chunk {chunk_idx} processed at {fps:.2f} fps")

#     # Sort results by timestamp
#     for roi_idx in roi_changes:
#         roi_changes[roi_idx].sort(key=lambda x: x[0])

#     total_duration = (num_frames - 1) * frame_interval
#     total_proc = time.time() - start_all
#     logger.info(f"Dual-structure file processed in {total_proc:.2f} seconds")

#     if progress_callback:
#         progress_callback(100, f"Completed dual-structure processing {os.path.basename(file_path)}")

#     return file_path, roi_changes, total_duration

# # =============================================================================
# # ENHANCED PROCESSING FUNCTIONS
# # =============================================================================

# def process_hdf5_files(directory: str,
#                       masks: Optional[List[np.ndarray]] = None,
#                       num_processes: Optional[int] = None,
#                       chunk_size: int = 50,
#                       min_radius: int = 80,
#                       max_radius: int = 150,
#                       progress_callback: Optional[Callable] = None,
#                       frame_interval: float = 5,
#                       dp: float = 0.5,
#                       min_dist: int = 150,
#                       param1: int = 40,
#                       param2: int = 40) -> Tuple[Dict, Dict, List[np.ndarray], Optional[np.ndarray]]:
#     """
#     Enhanced HDF5 processing with automatic structure detection.
#     """
#     start_all = time.time()

#     try:
#         h5_files = sorted([
#             os.path.join(directory, f)
#             for f in os.listdir(directory)
#             if f.lower().endswith(('.h5', '.hdf5')) and os.path.isfile(os.path.join(directory, f))
#         ])
#     except OSError as e:
#         logger.error(f"Error reading directory {directory}: {e}")
#         if progress_callback:
#             progress_callback(0, f"Error reading directory: {e}")
#         return {}, {}, [], None

#     if not h5_files:
#         logger.warning("No HDF5 files found in the directory.")
#         if progress_callback:
#             progress_callback(0, "No HDF5 files found.")
#         return {}, {}, [], None

#     # Enhanced: Detect structure of first file
#     if masks is None:
#         first_file = h5_files[0]
#         first_frame, structure_info = get_first_frame_enhanced(first_file)

#         if first_frame is None:
#             if progress_callback:
#                 progress_callback(0, f"Error: Could not read the first frame of {first_file}")
#             logger.error(f"Structure detection failed: {structure_info.get('error', 'Unknown error')}")
#             return {}, {}, [], None

#         logger.info(f"Detected HDF5 structure: {structure_info['type']}")
#         logger.info(f"Frame count: {structure_info['frame_count']}, Data location: {structure_info['data_location']}")

#         # Use enhanced detection with widget-compatible parameters
#         if CV2_AVAILABLE:
#             masks, labeled_frame = detect_circles_and_create_masks(
#                 first_frame, min_radius, max_radius, dp, min_dist, param1, param2
#             )
#         else:
#             logger.warning("OpenCV not available. Cannot detect ROIs.")
#             masks = []
#             labeled_frame = first_frame

#         if not masks:
#             if progress_callback:
#                 progress_callback(0, f"No ROIs detected in the first frame of {first_file}")
#             logger.warning(f"No ROIs detected in the first frame of {first_file}")
#             return {}, {}, [], labeled_frame
#     else:
#         # Create labeled frame from existing masks
#         first_file = h5_files[0]
#         first_frame, structure_info = get_first_frame_enhanced(first_file)
#         if first_frame is None:
#             labeled_frame = None
#         else:
#             if len(first_frame.shape) == 3:
#                 labeled_frame = first_frame.copy()
#             else:
#                 labeled_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2RGB) if CV2_AVAILABLE else np.stack([first_frame]*3, axis=-1)

#             # Draw existing masks on labeled frame
#             for idx, mask in enumerate(masks):
#                 try:
#                     if CV2_AVAILABLE:
#                         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                         if contours:
#                             (x, y), radius = cv2.minEnclosingCircle(contours[0])
#                             center = (int(x), int(y))
#                             radius = int(radius)
#                             cv2.circle(labeled_frame, center, radius, (0, 255, 0), 2)
#                             cv2.putText(labeled_frame, f"{idx+1}", (center[0]-20, center[1]+10),
#                                         cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
#                 except Exception as e:
#                     logger.warning(f"Error drawing mask {idx}: {e}")

#     # Memory and performance optimization
#     if num_processes is None:
#         available_memory_gb = get_available_memory() / (1024**3)
#         if available_memory_gb < 4:  # Less than 4GB RAM
#             num_processes = 1
#         elif available_memory_gb < 8:  # Less than 8GB RAM
#             num_processes = max(1, min(3, int(os.cpu_count() * 0.6)))
#         else:
#             num_processes = max(1, min(6, int(os.cpu_count() * 0.8)))

#         logger.info(f"Auto-selected {num_processes} processes based on {available_memory_gb:.1f}GB available memory and {len(masks)} ROIs")

#     # Get optimal chunk size from first file
#     try:
#         frame_shape = structure_info['frame_shape']
#         dtype_size = structure_info['dtype_size']
#         optimal_chunk_size = calculate_optimal_chunk_size(frame_shape, dtype_size)
#         actual_chunk_size = max(MIN_CHUNK_SIZE, min(chunk_size, optimal_chunk_size))
#         logger.info(f"Adjusted chunk size to {actual_chunk_size} based on memory constraints and {len(masks)} ROIs")
#     except Exception as e:
#         logger.warning(f"Could not determine optimal chunk size: {e}")
#         actual_chunk_size = max(MIN_CHUNK_SIZE, chunk_size)

#     # Processing strategy based on number of files and memory
#     if len(h5_files) == 1:
#         if progress_callback:
#             progress_callback(0, f"Single file found; using dual-structure chunk-level parallelism with {num_processes} processes.")
#         file_path = h5_files[0]
#         file_path, roi_changes, total_duration = process_single_file_in_parallel_dual_structure(
#             file_path, masks, chunk_size=actual_chunk_size, progress_callback=progress_callback,
#             frame_interval=frame_interval, num_processes=num_processes
#         )
#         results = {file_path: roi_changes}
#         durations = {file_path: total_duration}
#         processed_files = 1
#     else:
#         if progress_callback:
#             progress_callback(0, f"Multiple files found; processing with dual-structure method using {num_processes} processes.")
#         results = {}
#         durations = {}
#         processed_files = 0

#         def file_progress_callback(file_idx: int, total_files: int, filename: str):
#             def callback(percent: float, message: str):
#                 overall = ((file_idx - 1) + (percent / 100)) / total_files * 100
#                 if progress_callback:
#                     progress_callback(overall, f"File {file_idx}/{total_files}: {message}")
#             return callback

#         if num_processes == 1:
#             # Single-threaded processing
#             for file_idx, file_path in enumerate(h5_files, 1):
#                 fp_callback = file_progress_callback(file_idx, len(h5_files), file_path)
#                 file_path, roi_changes, tot_dur = process_hdf5_file_dual_structure(
#                     file_path, masks, actual_chunk_size, fp_callback, frame_interval
#                 )
#                 if roi_changes:
#                     processed_files += 1
#                     results[file_path] = roi_changes
#                     durations[file_path] = tot_dur
#         else:
#             # Multi-threaded processing with memory management
#             try:
#                 with ProcessPoolExecutor(max_workers=num_processes) as executor:
#                     futures = []
#                     for idx, file_path in enumerate(h5_files, 1):
#                         future = executor.submit(process_hdf5_file_dual_structure, file_path, masks, actual_chunk_size, None, frame_interval)
#                         futures.append((future, file_path, idx))

#                     for future, file_path, file_idx in futures:
#                         try:
#                             file_path, roi_changes, tot_dur = future.result(timeout=3600)  # 1 hour timeout
#                             if roi_changes:
#                                 processed_files += 1
#                                 results[file_path] = roi_changes
#                                 durations[file_path] = tot_dur
#                             if progress_callback:
#                                 fraction = (file_idx / len(h5_files)) * 100
#                                 progress_callback(fraction, f"Completed {file_idx}/{len(h5_files)}: {os.path.basename(file_path)}")
#                         except Exception as e:
#                             logger.error(f"Error processing {file_path}: {e}")
#                             if progress_callback:
#                                 fraction = (file_idx / len(h5_files)) * 100
#                                 progress_callback(fraction, f"Error processing {os.path.basename(file_path)}: {e}")
#             except Exception as e:
#                 logger.error(f"Multi-processing failed: {e}. Falling back to single-threaded processing.")
#                 # Fallback to single-threaded
#                 for file_idx, file_path in enumerate(h5_files, 1):
#                     fp_callback = file_progress_callback(file_idx, len(h5_files), file_path)
#                     try:
#                         file_path, roi_changes, tot_dur = process_hdf5_file_dual_structure(
#                             file_path, masks, actual_chunk_size, fp_callback, frame_interval
#                         )
#                         if roi_changes:
#                             processed_files += 1
#                             results[file_path] = roi_changes
#                             durations[file_path] = tot_dur
#                     except Exception as e:
#                         logger.error(f"Error processing {file_path}: {e}")

#     elapsed_all = time.time() - start_all
#     logger.info(f"Finished processing in {elapsed_all:.2f}s, processed {processed_files}/{len(h5_files)} files total.")
#     if progress_callback:
#         progress_callback(100, f"Analysis complete. Processed {processed_files}/{len(h5_files)} files.")

#     return results, durations, masks, labeled_frame

# # =============================================================================
# # UTILITY FUNCTIONS
# # =============================================================================

# def merge_results(results: Dict[str, Dict], durations: Dict[str, float]) -> Dict[int, List[Tuple[float, float]]]:
#     """
#     Merge results from multiple files into a continuous timeline with memory optimization.
#     """
#     merged_results = {}
#     cumulative_time = 0.0
#     sorted_paths = sorted(results.keys())

#     for path in sorted_paths:
#         roi_changes = results[path]
#         for roi, pairs in roi_changes.items():
#             if roi not in merged_results:
#                 merged_results[roi] = []

#             # Memory-efficient merging for large datasets
#             if len(pairs) > 10000:  # Large dataset threshold
#                 # Process in batches to avoid memory spikes
#                 batch_size = 5000
#                 for i in range(0, len(pairs), batch_size):
#                     batch = pairs[i:i + batch_size]
#                     adjusted_batch = [(t_sec + cumulative_time, val) for (t_sec, val) in batch]
#                     merged_results[roi].extend(adjusted_batch)
#             else:
#                 # Standard processing for smaller datasets
#                 for (t_sec, val) in pairs:
#                     merged_results[roi].append((t_sec + cumulative_time, val))

#         cumulative_time += durations[path]

#     return merged_results

# def get_roi_colors(rois: List[int]) -> Dict[int, str]:
#     """
#     Generate consistent colors for ROIs with fallback when matplotlib is not available.
#     """
#     roi_colors = {}

#     if MATPLOTLIB_AVAILABLE:
#         try:
#             color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
#         except Exception:
#             # Fallback color cycle
#             color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
#     else:
#         # Default color cycle when matplotlib is not available
#         color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

#     for i, roi_id in enumerate(rois):
#         roi_colors[roi_id] = color_cycle[i % len(color_cycle)]

#     return roi_colors

# def get_memory_usage() -> Dict[str, float]:
#     """Get current memory usage statistics."""
#     if PSUTIL_AVAILABLE:
#         try:
#             memory = psutil.virtual_memory()
#             return {
#                 'total_gb': memory.total / (1024**3),
#                 'available_gb': memory.available / (1024**3),
#                 'used_gb': memory.used / (1024**3),
#                 'percent_used': memory.percent
#             }
#         except Exception:
#             pass

#     return {
#         'total_gb': 'unknown',
#         'available_gb': 'unknown',
#         'used_gb': 'unknown',
#         'percent_used': 'unknown'
#     }

# def log_memory_usage(context: str = ""):
#     """Log current memory usage for debugging."""
#     memory_info = get_memory_usage()
#     if memory_info['percent_used'] != 'unknown':
#         logger.info(f"Memory usage {context}: {memory_info['used_gb']:.1f}GB/{memory_info['total_gb']:.1f}GB "
#                    f"({memory_info['percent_used']:.1f}% used, {memory_info['available_gb']:.1f}GB available)")
#     else:
#         logger.info(f"Memory usage {context}: unable to determine")

# def cleanup_large_arrays(*arrays):
#     """Explicitly cleanup large numpy arrays to help with memory management."""
#     for arr in arrays:
#         if hasattr(arr, 'nbytes') and arr.nbytes > 100 * 1024 * 1024:  # 100MB threshold
#             logger.debug(f"Cleaning up large array: {arr.nbytes / (1024*1024):.1f}MB")
#         del arr

# # =============================================================================
# # DIAGNOSTIC FUNCTIONS
# # =============================================================================

# def diagnose_preprocessing_impact(chunk_data: np.ndarray, masks: List[np.ndarray]):
#     """
#     Diagnostic function for preprocessing impact analysis.
#     """
#     if len(chunk_data) < 2 or len(masks) == 0:
#         return

#     print("=== PREPROCESSING DIAGNOSIS ===")

#     try:
#         # Convert to grayscale if needed
#         if len(chunk_data.shape) == 4 and chunk_data.shape[-1] == 3:
#             if CV2_AVAILABLE:
#                 gray_frames = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in chunk_data[:2]])
#             else:
#                 weights = np.array([0.299, 0.587, 0.114])
#                 gray_frames = np.dot(chunk_data[:2], weights).astype(np.uint8)
#         else:
#             gray_frames = chunk_data[:2].copy()

#         print(f"Original frame dtype: {gray_frames.dtype}")
#         print(f"Original frame range: {np.min(gray_frames)} to {np.max(gray_frames)}")

#         # Normalization
#         normalized_frames = normalize_image_to_float32(gray_frames)

#         print(f"After normalization range: {np.min(normalized_frames):.6f} to {np.max(normalized_frames):.6f}")

#         # Test processing on first ROI
#         mask = masks[0]
#         mask_bool = mask > 0

#         # Calculate difference
#         frame_curr = normalized_frames[0]
#         frame_next = normalized_frames[1]
#         diff_masked = np.abs(frame_next[mask_bool] - frame_curr[mask_bool])
#         total_intensity = np.sum(diff_masked)

#         roi_area = np.sum(mask_bool)

#         print(f"\nROI Analysis (first ROI):")
#         print(f"ROI area: {roi_area:,} pixels")
#         print(f"Total intensity change: {total_intensity:.6f}")
#         print(f"Per-pixel change: {total_intensity/roi_area:.8f}")
#         print(f"Expected value range: 0.001 - 10 (compatible with analysis pipeline)")

#     except Exception as e:
#         print(f"Error in preprocessing diagnosis: {e}")

# # =============================================================================
# # LEGACY FUNCTIONS FOR BACKWARD COMPATIBILITY
# # =============================================================================

# def _process_single_chunk(args: Tuple) -> Tuple[int, Dict[int, List[Tuple[float, float]]]]:
#     """
#     Legacy chunk processing function - redirects to dual structure version.
#     """
#     try:
#         if len(args) == 6:
#             # New format with structure_info
#             return _process_single_chunk_dual_structure(args)
#         else:
#             # Old format - need to detect structure
#             file_path, masks, start_idx, end_idx, frame_interval = args
#             structure_info = detect_hdf5_structure_type(file_path)
#             new_args = (file_path, masks, start_idx, end_idx, frame_interval, structure_info)
#             return _process_single_chunk_dual_structure(new_args)

#     except Exception as e:
#         logger.error(f"Error in legacy chunk processing: {e}")
#         return 0, {}

# def process_single_file_in_parallel(file_path: str, masks: List[np.ndarray],
#                                    chunk_size: int = 50,
#                                    progress_callback: Optional[Callable] = None,
#                                    frame_interval: float = 5,
#                                    num_processes: Optional[int] = None) -> Tuple[str, Dict, float]:
#     """
#     Legacy function - redirects to dual structure version.
#     """
#     return process_single_file_in_parallel_dual_structure(
#         file_path, masks, chunk_size, progress_callback, frame_interval, num_processes
#     )

# def process_hdf5_file(file_path: str, masks: List[np.ndarray],
#                      chunk_size: int = 50,
#                      progress_callback: Optional[Callable] = None,
#                      frame_interval: float = 5) -> Tuple[str, Dict, float]:
#     """
#     Legacy function - redirects to dual structure version.
#     """
#     return process_hdf5_file_dual_structure(
#         file_path, masks, chunk_size, progress_callback, frame_interval
#     )
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np

import logging

# Enhanced import handling with fallbacks
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Circle detection will be disabled.")

try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    raise ImportError("h5py is required for HDF5 file processing")

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    raise ImportError("numpy is required for data processing")

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Color generation will use fallback.")

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Using conservative memory estimates.")

logger = logging.getLogger(__name__)

# Memory management constants
MAX_CHUNK_SIZE_MB = 512  # Maximum chunk size in MB
MIN_CHUNK_SIZE = 10  # Minimum number of frames per chunk
MAX_CHUNK_SIZE = 200  # Maximum number of frames per chunk
MEMORY_SAFETY_FACTOR = 0.25  # Use only 25% of available memory


def get_available_memory() -> int:
    """Get available system memory in bytes."""
    if PSUTIL_AVAILABLE:
        try:
            return psutil.virtual_memory().available
        except Exception:
            pass
    # Fallback to conservative estimate
    return 2 * 1024 * 1024 * 1024  # 2 GB


def calculate_optimal_chunk_size(
    frame_shape: Tuple[int, ...], dtype_size: int = 2
) -> int:
    """
    Calculate optimal chunk size based on available memory and frame size.

    Args:
        frame_shape: Shape of a single frame (height, width) or (height, width, channels)
        dtype_size: Size of data type in bytes (2 for uint16, 1 for uint8, etc.)

    Returns:
        Optimal number of frames per chunk
    """
    available_memory = get_available_memory()
    memory_budget = available_memory * MEMORY_SAFETY_FACTOR

    # Calculate frame size in bytes
    frame_size = np.prod(frame_shape) * dtype_size

    # Add overhead for processing (diff operations, masks, etc.)
    processing_overhead = 4  # Python-native processing overhead
    effective_frame_size = frame_size * processing_overhead

    # Calculate number of frames that fit in memory budget
    frames_in_budget = int(memory_budget / effective_frame_size)

    # Return a reasonable chunk size
    optimal_size = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, frames_in_budget))

    logger.info(
        f"Memory calculation: {memory_budget/(1024**2):.1f}MB budget, "
        f"{frame_size/(1024**2):.1f}MB per frame, optimal chunk: {optimal_size} frames"
    )

    return optimal_size


# =============================================================================
# IMAGE FORMAT CONVERSION UTILITIES
# =============================================================================


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB/RGBA image to grayscale using consistent method.

    Args:
        image: Input image array

    Returns:
        Grayscale image array
    """
    if len(image.shape) == 2:
        # Already grayscale
        return image.copy()
    elif len(image.shape) == 3:
        if image.shape[-1] == 3:  # RGB
            if CV2_AVAILABLE:
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                # Use standard luminance weights
                weights = np.array([0.299, 0.587, 0.114])
                return np.dot(image, weights).astype(image.dtype)
        elif image.shape[-1] == 4:  # RGBA
            if CV2_AVAILABLE:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                # Use RGB channels only, ignore alpha
                weights = np.array([0.299, 0.587, 0.114])
                return np.dot(image[:, :, :3], weights).astype(image.dtype)
        else:
            logger.warning(f"Unexpected number of channels: {image.shape[-1]}")
            return image[:, :, 0].copy()  # Take first channel
    else:
        logger.error(f"Unexpected image shape: {image.shape}")
        return image


def convert_to_rgb_for_display(image: np.ndarray) -> np.ndarray:
    """
    Convert grayscale image to RGB for napari display.

    Args:
        image: Input image array (grayscale)

    Returns:
        RGB image array for display
    """
    if len(image.shape) == 2:
        # Grayscale to RGB - stack 3 times
        return np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        if image.shape[-1] == 1:
            # Single channel to RGB
            return np.concatenate([image, image, image], axis=-1)
        elif image.shape[-1] == 3:
            # Already RGB
            return image.copy()
        elif image.shape[-1] == 4:
            # RGBA to RGB - drop alpha channel
            return image[:, :, :3].copy()
        else:
            logger.warning(f"Unexpected number of channels: {image.shape[-1]}")
            return np.stack([image[:, :, 0], image[:, :, 0], image[:, :, 0]], axis=-1)
    else:
        logger.error(f"Unexpected image shape: {image.shape}")
        # Fallback - create a basic RGB image
        if image.size > 0:
            flat_image = image.flatten()[: image.shape[0] * image.shape[1]].reshape(
                image.shape[:2]
            )
            return np.stack([flat_image, flat_image, flat_image], axis=-1)
        else:
            return np.zeros((100, 100, 3), dtype=image.dtype)


def preprocess_image_for_processing(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess image for both display and processing.

    Args:
        image: Raw image from HDF5 file

    Returns:
        Tuple of (display_image_rgb, processing_image_grayscale)
    """
    # Always convert to grayscale for processing
    grayscale_image = convert_to_grayscale(image)

    # Always convert to RGB for display
    rgb_image = convert_to_rgb_for_display(grayscale_image)

    logger.debug(
        f"Image preprocessing: input shape {image.shape} -> "
        f"display RGB {rgb_image.shape}, processing grayscale {grayscale_image.shape}"
    )

    return rgb_image, grayscale_image


def preprocess_image_stack_for_processing(image_stack: np.ndarray) -> np.ndarray:
    """
    Preprocess an entire image stack for processing (convert to grayscale).

    Args:
        image_stack: Stack of images from HDF5 file

    Returns:
        Grayscale image stack for processing
    """
    if len(image_stack.shape) == 3:
        # Already grayscale stack (frames, height, width)
        return image_stack.copy()
    elif len(image_stack.shape) == 4:
        # RGB/RGBA stack (frames, height, width, channels)
        if image_stack.shape[-1] == 3:  # RGB
            if CV2_AVAILABLE:
                # Batch convert to grayscale
                grayscale_stack = np.array(
                    [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in image_stack]
                )
            else:
                # Use vectorized conversion
                weights = np.array([0.299, 0.587, 0.114])
                grayscale_stack = np.dot(image_stack, weights).astype(image_stack.dtype)
        elif image_stack.shape[-1] == 4:  # RGBA
            if CV2_AVAILABLE:
                grayscale_stack = np.array(
                    [cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY) for frame in image_stack]
                )
            else:
                # Use RGB channels only
                weights = np.array([0.299, 0.587, 0.114])
                grayscale_stack = np.dot(image_stack[:, :, :, :3], weights).astype(
                    image_stack.dtype
                )
        else:
            logger.warning(f"Unexpected number of channels: {image_stack.shape[-1]}")
            grayscale_stack = image_stack[:, :, :, 0].copy()  # Take first channel

        logger.info(
            f"Converted RGB stack {image_stack.shape} to grayscale {grayscale_stack.shape}"
        )
        return grayscale_stack
    else:
        logger.error(f"Unexpected image stack shape: {image_stack.shape}")
        return image_stack


# =============================================================================
# HDF5 STRUCTURE DETECTION AND READING
# =============================================================================


def detect_hdf5_structure_type(file_path: str) -> Dict[str, Any]:
    """
    Detect the structure type of an HDF5 file.

    Returns:
        Dictionary with structure information
    """
    try:
        with h5py.File(file_path, "r") as h5_file:
            root_keys = list(h5_file.keys())

            # Structure 1: Stacked frames dataset
            if "frames" in h5_file:
                frames_dataset = h5_file["frames"]
                if len(frames_dataset) > 0:
                    return {
                        "type": "stacked_frames",
                        "dataset_name": "frames",
                        "frame_count": len(frames_dataset),
                        "frame_shape": frames_dataset[0].shape,
                        "dtype": frames_dataset.dtype,
                        "dtype_size": frames_dataset.dtype.itemsize,
                        "data_location": "frames",
                    }

            # Structure 2: Individual frames in images/ group
            if "images" in h5_file:
                images_group = h5_file["images"]
                if len(images_group.keys()) > 0:
                    first_key = sorted(images_group.keys())[0]
                    first_image = images_group[first_key]
                    return {
                        "type": "individual_frames",
                        "group_name": "images",
                        "frame_count": len(images_group.keys()),
                        "frame_shape": first_image.shape,
                        "dtype": first_image.dtype,
                        "dtype_size": first_image.dtype.itemsize,
                        "data_location": "images",
                        "frame_keys": sorted(images_group.keys()),
                    }

            # Structure 3: Look for other potential datasets
            for key in root_keys:
                if isinstance(h5_file[key], h5py.Dataset):
                    dataset = h5_file[key]
                    # Check if this could be image data (3+ dimensions)
                    if len(dataset.shape) >= 3 and dataset.size > 1000:
                        return {
                            "type": "alternative_dataset",
                            "dataset_name": key,
                            "frame_count": dataset.shape[0],
                            "frame_shape": dataset.shape[1:],
                            "dtype": dataset.dtype,
                            "dtype_size": dataset.dtype.itemsize,
                            "data_location": key,
                        }

            return {
                "type": "unknown",
                "error": f"No suitable image data found in {root_keys}",
                "available_keys": root_keys,
            }

    except Exception as e:
        return {"type": "error", "error": str(e)}


def get_first_frame_enhanced(
    file_path: str, driver: Optional[str] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Enhanced first frame getter with automatic structure detection and format conversion.

    Returns:
        Tuple of (display_frame_rgb, processing_frame_grayscale, structure_info)
    """
    if not H5PY_AVAILABLE:
        return None, None, {"type": "error", "error": "h5py not available"}

    try:
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return None, None, {"type": "error", "error": "File not found"}

        structure_info = detect_hdf5_structure_type(file_path)

        if structure_info["type"] == "error":
            logger.error(f"Structure detection failed: {structure_info['error']}")
            return None, None, structure_info

        with h5py.File(file_path, "r", **(dict(driver=driver) if driver else {})) as f:
            logger.info(f"HDF5 file structure: {list(f.keys())}")

            if structure_info["type"] == "stacked_frames":
                # Handle stacked frames (original structure)
                frames_dataset = f["frames"]
                raw_frame = frames_dataset[0].copy()
                logger.info(
                    f"Loaded first frame from stacked dataset: {raw_frame.shape}"
                )

            elif structure_info["type"] == "individual_frames":
                # Handle individual frames in images/ group
                images_group = f["images"]
                frame_keys = structure_info["frame_keys"]
                first_key = frame_keys[0]
                raw_frame = images_group[first_key][...].copy()
                logger.info(
                    f"Loaded first frame from images group: key={first_key}, shape={raw_frame.shape}"
                )

            elif structure_info["type"] == "alternative_dataset":
                # Handle alternative dataset structure
                dataset_name = structure_info["dataset_name"]
                dataset = f[dataset_name]
                raw_frame = dataset[0].copy()
                logger.info(
                    f"Loaded first frame from alternative dataset '{dataset_name}': {raw_frame.shape}"
                )

            else:
                logger.error(f"Unknown structure type: {structure_info['type']}")
                return None, None, structure_info

            # Preprocess for both display and processing
            display_frame, processing_frame = preprocess_image_for_processing(raw_frame)

            logger.info(
                f"Frame preprocessing complete: "
                f"display={display_frame.shape}, processing={processing_frame.shape}"
            )

            return display_frame, processing_frame, structure_info

    except Exception as e:
        logger.error(f"Error reading first frame: {e}")
        return None, None, {"type": "error", "error": str(e)}


def read_chunk_data_dual_structure(
    file_path: str,
    start_idx: int,
    end_idx: int,
    structure_info: Dict[str, Any],
    driver: Optional[str] = None,
    for_processing: bool = True,
) -> Optional[np.ndarray]:
    """
    Read a chunk of data handling both stacked and individual frame structures.

    Args:
        file_path: Path to HDF5 file
        start_idx: Start frame index
        end_idx: End frame index
        structure_info: Structure information from detection
        driver: HDF5 driver
        for_processing: If True, return grayscale data for processing. If False, return raw data.

    Returns:
        Chunk data (grayscale if for_processing=True, raw if for_processing=False)
    """
    try:
        with h5py.File(file_path, "r", **(dict(driver=driver) if driver else {})) as f:

            if structure_info["type"] == "stacked_frames":
                # Standard stacked frames approach
                frames_dataset = f["frames"]
                chunk_data = frames_dataset[start_idx:end_idx].copy()

            elif structure_info["type"] == "individual_frames":
                # Read individual frames from images/ group
                images_group = f["images"]
                frame_keys = structure_info["frame_keys"]

                # Get the keys for this chunk
                chunk_keys = frame_keys[start_idx:end_idx]

                # Read each frame individually and stack them
                frames = []
                for key in chunk_keys:
                    frame = images_group[key][...].copy()
                    frames.append(frame)

                # Stack into array format expected by processing
                if frames:
                    chunk_data = np.stack(frames, axis=0)
                else:
                    return None

            elif structure_info["type"] == "alternative_dataset":
                # Handle alternative dataset
                dataset = f[structure_info["dataset_name"]]
                chunk_data = dataset[start_idx:end_idx].copy()

            else:
                logger.error(
                    f"Cannot read chunk for structure type: {structure_info['type']}"
                )
                return None

            # Convert to grayscale for processing if requested
            if for_processing:
                chunk_data = preprocess_image_stack_for_processing(chunk_data)

            logger.debug(
                f"Read chunk {start_idx}:{end_idx}, shape: {chunk_data.shape}, "
                f"for_processing: {for_processing}"
            )
            return chunk_data

    except Exception as e:
        logger.error(f"Error reading chunk {start_idx}:{end_idx}: {e}")
        return None


# =============================================================================
# PYTHON-NATIVE NORMALIZATION FUNCTIONS
# =============================================================================


def normalize_image_to_float32(
    image: np.ndarray, target_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Python-native image normalization.

    Args:
        image: Input image (uint8, uint16, or float)
        target_range: Target range for normalized values

    Returns:
        Normalized float32 array
    """
    # Convert to float32 for processing
    if image.dtype == np.uint8:
        # Standard 8-bit images: 0-255 → 0.0-1.0
        normalized = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        # 16-bit images: 0-65535 → 0.0-1.0
        normalized = image.astype(np.float32) / 65535.0
    elif image.dtype in [np.int16, np.int32]:
        # Signed integers: normalize to actual range
        img_float = image.astype(np.float32)
        img_min, img_max = img_float.min(), img_float.max()
        if img_max > img_min:
            normalized = (img_float - img_min) / (img_max - img_min)
        else:
            normalized = np.zeros_like(img_float)
    elif image.dtype in [np.float32, np.float64]:
        # Already float - check if needs rescaling
        img_min, img_max = image.min(), image.max()
        if img_max > 1.0 or img_min < 0.0:
            # Rescale to 0-1 if outside standard range
            if img_max > img_min:
                normalized = (image.astype(np.float32) - img_min) / (img_max - img_min)
            else:
                normalized = np.zeros_like(image, dtype=np.float32)
        else:
            normalized = image.astype(np.float32)
    else:
        raise ValueError(f"Unsupported image dtype: {image.dtype}")

    # Adjust to target range if different from [0,1]
    if target_range != (0.0, 1.0):
        target_min, target_max = target_range
        normalized = normalized * (target_max - target_min) + target_min

    return normalized


def validate_data_ranges(data: np.ndarray, stage: str, roi_idx: Optional[int] = None):
    """
    Validate data ranges at different processing stages.
    """
    roi_info = f" ROI {roi_idx}" if roi_idx is not None else ""
    logger.debug(f"=== DATA VALIDATION{roi_info}: {stage} ===")
    logger.debug(f"  Dtype: {data.dtype}")
    logger.debug(f"  Shape: {data.shape}")
    logger.debug(f"  Range: {data.min():.6f} to {data.max():.6f}")
    logger.debug(f"  Mean: {data.mean():.6f}")
    logger.debug(f"  Std: {data.std():.6f}")

    # Validation checks
    if stage == "Raw_Input":
        if data.dtype == np.uint8 and (data.min() < 0 or data.max() > 255):
            logger.warning(f"  ❌ uint8 values outside expected range!")
        elif data.dtype == np.uint16 and (data.min() < 0 or data.max() > 65535):
            logger.warning(f"  ❌ uint16 values outside expected range!")
        else:
            logger.debug(f"  ✅ Raw input range looks correct")

    elif stage == "Normalized_Output":
        if data.dtype != np.float32:
            logger.warning(f"  ❌ Expected float32 output, got {data.dtype}!")
        elif data.min() < -0.1 or data.max() > 1.1:
            logger.warning(f"  ❌ Normalized values outside [0,1] range!")
        else:
            logger.debug(f"  ✅ Normalization successful")


# =============================================================================
# NAPARI READER FUNCTIONS
# =============================================================================


def napari_get_reader(path: Union[str, List[str]]) -> Optional[Callable]:
    """
    Returns a reader function if `path` is a valid HDF5 or AVI file,
    or a directory containing such files. Otherwise returns None.
    """
    # Check for AVI files
    if (isinstance(path, str) and path.lower().endswith(".avi")) or (
        isinstance(path, list)
        and all(isinstance(p, str) and p.lower().endswith(".avi") for p in path)
    ):
        return reader_function_avi

    # Check for HDF5 files
    if not H5PY_AVAILABLE:
        logger.error("h5py not available. Cannot read HDF5 files.")
        return None

    if os.path.isdir(path):
        return reader_directory_function
    elif (isinstance(path, str) and path.lower().endswith((".h5", ".hdf5"))) or (
        isinstance(path, list)
        and all(
            isinstance(p, str) and p.lower().endswith((".h5", ".hdf5")) for p in path
        )
    ):
        return reader_function_dual_structure

    return None


def _read_avi_batch(paths: List[str], batch_loader_func) -> List[Tuple]:
    """
    Helper function to read multiple AVI files as a continuous timeseries.

    Args:
        paths: List of AVI file paths in temporal order
        batch_loader_func: The batch loading function from _avi_reader

    Returns:
        List of layer data tuples for napari
    """
    try:
        # Get target frame interval from first file's metadata or use default
        target_interval = 5.0  # Default: 5 seconds (like HDF5)

        # Check if metadata specifies frame interval
        try:
            from ._avi_reader import AVIVideoReader

            with AVIVideoReader(paths[0]) as reader:
                target_interval = reader.metadata.get("frame_interval", 5.0)
        except:
            pass

        logger.info(f"Target frame interval: {target_interval}s")

        # Load all videos with time concatenation
        frames, metadata = batch_loader_func(paths, target_interval)

        if frames is None or len(frames) == 0:
            logger.error("Could not load frames from batch AVI files")
            return []

        logger.info(f"Batch loaded {len(frames)} frames from {len(paths)} videos")
        logger.info(f"Total duration: {metadata.get('total_duration', 0):.1f}s")

        # Prepare layer data
        layers: List[Tuple] = []

        # Get first frame for display
        display_frame = frames[0, :, :, 0] if len(frames) > 0 else None

        if display_frame is None:
            logger.error("Could not extract first frame for display")
            return []

        # Create layer name
        layer_name = f"batch_{len(paths)}_videos"

        # Add image layer
        layers.append(
            (
                display_frame,
                {
                    "name": layer_name,
                    "metadata": {
                        "path": paths[0],  # First video path
                        "source_type": "avi_batch",
                        "is_hdf5": False,
                        "avi_file_paths": paths,
                        "frame_count": metadata.get("total_frames", len(frames)),
                        "fps": metadata.get("effective_fps", 5),
                        "frame_interval": target_interval,
                        "resolution": {},
                        "duration": metadata.get("total_duration", 0),
                        "full_metadata": metadata,
                        "processing_style": "avi_batch",
                        "video_count": len(paths),
                        "timestamps": metadata.get("timestamps", []),
                    },
                },
                "image",
            )
        )

        return layers

    except Exception as e:
        logger.error(f"Error reading batch AVI files: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return []


def reader_function_avi(path: Union[str, List[str]]) -> List[Tuple]:
    """
    Reader function for AVI video files.

    Compatible with existing MATLAB workflow (ActivityExtractorPolyp_v20230105.m)

    Args:
        path: Path to AVI file or list of paths

    Returns:
        List of layer data tuples for napari
    """
    try:
        from ._avi_reader import load_avi_with_metadata, load_avi_batch_timeseries
    except ImportError as e:
        logger.error(f"Could not import AVI reader: {e}")
        return []

    # Handle list input - batch processing for multiple AVIs
    if isinstance(path, list):
        if len(path) == 1:
            path = path[0]
        else:
            # Multiple AVI files - batch process with time concatenation
            logger.info(f"Batch processing {len(path)} AVI files")
            return _read_avi_batch(path, load_avi_batch_timeseries)

    try:
        # Check file exists
        if not os.path.exists(path):
            logger.error(f"AVI file does not exist: {path}")
            return []

        # Load AVI with metadata
        frames, metadata = load_avi_with_metadata(path)

        if frames is None or len(frames) == 0:
            logger.error(f"Could not load frames from AVI file: {path}")
            return []

        logger.info(f"Loaded AVI file: {path}")
        logger.info(f"Frames shape: {frames.shape}")
        logger.info(f"FPS: {metadata.get('fps', 'unknown')}")
        logger.info(f"Sampling rate: {metadata.get('sampling_rate', 1)}")

        # Prepare layer data
        layers: List[Tuple] = []

        # Get first frame for display
        display_frame = frames[0, :, :, 0] if len(frames) > 0 else None

        if display_frame is None:
            logger.error("Could not extract first frame for display")
            return []

        # Add image layer
        layers.append(
            (
                display_frame,
                {
                    "name": os.path.basename(path),
                    "metadata": {
                        "path": path,
                        "source_type": "avi",
                        "is_hdf5": False,
                        "avi_file_path": path,
                        "frame_count": metadata.get("frame_count", len(frames)),
                        "fps": metadata.get("fps", 5),
                        "frame_interval": metadata.get("frame_interval", 0.2),
                        "sampling_rate": metadata.get("sampling_rate", 1),
                        "resolution": metadata.get("resolution", {}),
                        "duration": metadata.get("duration", 0),
                        "full_metadata": metadata,
                        "processing_style": "avi",
                    },
                },
                "image",
            )
        )

        return layers

    except Exception as e:
        logger.error(f"Error reading AVI file {path}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return []


def reader_function_dual_structure(
    path: Union[str, List[str]], driver: Optional[str] = None
) -> List[Tuple]:
    """
    Enhanced reader function with automatic HDF5 structure detection and format handling.
    Handles both stacked frames and individual frames in images/ group.
    """
    if not H5PY_AVAILABLE:
        logger.error("h5py not available")
        return []

    # Handle list input
    if isinstance(path, list) and len(path) == 1:
        path = path[0]
    if isinstance(path, list):
        return reader_directory_function(
            os.path.dirname(path[0]), filenames=path, driver=driver
        )

    try:
        # Check file exists and is readable
        if not os.path.exists(path):
            logger.error(f"File does not exist: {path}")
            return []

        if not os.access(path, os.R_OK):
            logger.error(f"File is not readable: {path}")
            return []

        # Get first frame and detect structure with format conversion
        display_frame, processing_frame, structure_info = get_first_frame_enhanced(
            path, driver=driver
        )
        if display_frame is None or processing_frame is None:
            logger.error(f"Could not read first frame from {path}")
            if structure_info.get("error"):
                logger.error(f"Structure detection error: {structure_info['error']}")
            return []

        logger.info(
            f"Successfully loaded {structure_info['type']} structure from {path}"
        )
        logger.info(
            f"Image format: display={display_frame.shape}, processing={processing_frame.shape}"
        )

    except Exception as e:
        logger.error(f"Error reading HDF5 file: {e}")
        return []

    layers: List[Tuple] = []

    # Add the display frame (RGB) for visualization
    layers.append(
        (
            display_frame,
            {
                "name": os.path.basename(path),
                "metadata": {
                    "path": path,
                    "structure_type": structure_info["type"],
                    "frame_count": structure_info["frame_count"],
                    "frame_shape": structure_info["frame_shape"],
                    "dtype_size": structure_info["dtype_size"],
                    "data_location": structure_info["data_location"],
                    "structure_info": structure_info,  # Pass complete info for processing
                    "is_hdf5": True,
                    "hdf5_file_path": path,
                    "optimal_chunk_size": calculate_optimal_chunk_size(
                        structure_info["frame_shape"], structure_info["dtype_size"]
                    ),
                    "processing_style": "python_native",
                    "display_format": "rgb",
                    "processing_format": "grayscale",
                },
            },
            "image",
        )
    )

    return layers


# LEGACY FUNCTIONS FOR BACKWARD COMPATIBILITY


def get_first_frame(path: str, driver: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Legacy function - now returns display frame only.
    """
    display_frame, processing_frame, structure_info = get_first_frame_enhanced(
        path, driver
    )
    return display_frame


def reader_directory_function(
    path: str, filenames: Optional[List[str]] = None, driver: Optional[str] = None
) -> List[Tuple]:
    """
    Enhanced directory reader with dual structure support and format handling.
    """
    if not H5PY_AVAILABLE:
        logger.error("h5py not available")
        return []

    if filenames is None:
        try:
            if not os.path.exists(path):
                logger.error(f"Directory does not exist: {path}")
                return []

            filenames = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.lower().endswith((".h5", ".hdf5"))
                and os.path.isfile(os.path.join(path, f))
            ]
        except OSError as e:
            logger.error(f"Error reading directory {path}: {e}")
            return []

    if not filenames:
        logger.warning(f"No HDF5 files found in {path}")
        return []

    layers: List[Tuple] = []
    first_file_path = filenames[0]

    # Use enhanced structure detection with format conversion
    try:
        display_frame, processing_frame, structure_info = get_first_frame_enhanced(
            first_file_path, driver
        )

        if display_frame is None or processing_frame is None:
            logger.error(f"Could not read first frame from {first_file_path}")
            return []

        logger.info(
            f"Directory processing: detected {structure_info['type']} structure"
        )
        logger.info(
            f"Frame format: display={display_frame.shape}, processing={processing_frame.shape}"
        )

        # Get metadata for memory management
        frame_shape = structure_info["frame_shape"]
        dtype_size = structure_info["dtype_size"]

    except Exception as e:
        logger.error(f"Error reading first HDF5 file: {e}")
        return []

    # Detect ROIs only if OpenCV is available - use processing frame (grayscale)
    if CV2_AVAILABLE:
        masks, labeled_frame = detect_circles_and_create_masks(
            processing_frame,  # Use grayscale for detection
            min_radius=80,
            max_radius=150,
            dp=0.5,
            min_dist=150,
            param1=40,
            param2=40,
        )
        # Convert labeled frame to RGB for display
        if len(labeled_frame.shape) == 2:
            labeled_frame = convert_to_rgb_for_display(labeled_frame)
    else:
        logger.warning("OpenCV not available. Skipping ROI detection.")
        masks = []
        labeled_frame = display_frame  # Use RGB display frame

    # Add overview image
    layers.append(
        (
            labeled_frame,
            {
                "name": "Detected ROIs" if CV2_AVAILABLE else "First Frame",
                "metadata": {
                    "path": first_file_path,
                    "detected_rois": len(masks),
                    "total_files": len(filenames),
                    "roi_detection_available": CV2_AVAILABLE,
                    "frame_shape": frame_shape,
                    "dtype_size": dtype_size,
                    "structure_type": structure_info["type"],
                    "optimal_chunk_size": calculate_optimal_chunk_size(
                        frame_shape, dtype_size
                    ),
                    "processing_style": "python_native",
                    "display_format": "rgb",
                    "processing_format": "grayscale",
                },
            },
            "image",
        )
    )

    # Add preview of each file with memory-aware processing
    for filename in filenames[:10]:  # Limit previews to prevent memory issues
        try:
            display_preview, processing_preview, preview_structure = (
                get_first_frame_enhanced(filename, driver)
            )

            if display_preview is not None:
                layers.append(
                    (
                        display_preview,  # Use RGB for display
                        {
                            "name": os.path.basename(filename),
                            "metadata": {
                                "path": filename,
                                "frame_count": preview_structure["frame_count"],
                                "structure_type": preview_structure["type"],
                                "is_hdf5": True,
                                "processing_style": "python_native",
                                "display_format": "rgb",
                                "processing_format": "grayscale",
                            },
                        },
                        "image",
                    )
                )
            else:
                logger.warning(f"No frames found in {filename}")
        except Exception as e:
            logger.error(f"Error reading HDF5 file {filename}: {e}")

    # Add ROI mask layers if detection was successful
    if CV2_AVAILABLE and masks:
        for i, mask in enumerate(masks):
            layers.append(
                (mask, {"name": f"ROI {i+1} Mask", "visible": False}, "labels")
            )

    return layers


def sort_circles_left_to_right(circles: np.ndarray) -> np.ndarray:
    """
    Sort detected circles from left to right based on x-coordinate only.
    """
    if circles is None or len(circles) == 0:
        return np.array([])

    # Extract circle data
    circles_list = [(circle[0], circle[1], circle[2]) for circle in circles[0]]

    # Sort based on x-coordinate only (left to right)
    sorted_circles = sorted(circles_list, key=lambda circle: circle[0])

    # Convert back to numpy array format
    return np.array(sorted_circles)


def sort_circles_meandering_auto(circles: np.ndarray) -> np.ndarray:
    """
    Automatically sort circles in meandering pattern based on detected count.
    Supports 6-well (2x3), 12-well (3x4), and 24-well (4x6) plates.

    Pattern:
    Row 1: 1→2→3→4
    Row 2: 8←7←6←5
    Row 3: 9→10→11→12
    """
    if circles is None or len(circles) == 0:
        return circles

    # Remove extra dimension from HoughCircles output
    if len(circles.shape) == 3:
        circles = circles[0]

    num_circles = len(circles)

    # Auto-detect plate layout based on circle count
    if num_circles == 6:
        rows, cols = 2, 3  # 6-well plate
    elif num_circles == 12:
        rows, cols = 3, 4  # 12-well plate
    elif num_circles == 24:
        rows, cols = 4, 6  # 24-well plate
    elif num_circles == 4:
        rows, cols = 2, 2  # 4-well
    elif num_circles == 8:
        rows, cols = 2, 4  # 8-well
    elif num_circles == 16:
        rows, cols = 4, 4  # 16-well
    else:
        # For other counts, fall back to simple left-to-right sorting
        return sort_circles_left_to_right_simple(circles)

    # Group circles into rows based on Y coordinates
    circle_rows = _group_into_rows(circles, rows)

    # Apply meandering pattern
    sorted_circles = []
    for row_idx, row_circles in enumerate(circle_rows):
        if len(row_circles) == 0:
            continue

        # Sort current row by X coordinate (left to right)
        row_sorted = sorted(row_circles, key=lambda c: c[0])

        # Reverse every odd row (0-indexed) for meandering pattern
        if row_idx % 2 == 1:
            row_sorted = row_sorted[::-1]

        sorted_circles.extend(row_sorted)

    return np.array(sorted_circles, dtype=np.uint16)


def _group_into_rows(circles: np.ndarray, expected_rows: int) -> list:
    """Group circles into rows based on Y coordinates."""
    if len(circles) == 0:
        return []

    # Sort all circles by Y coordinate
    y_sorted_indices = np.argsort(circles[:, 1])
    y_sorted_circles = circles[y_sorted_indices]

    if expected_rows == 1:
        return [y_sorted_circles.tolist()]

    # Divide circles into expected number of rows
    circles_per_row = len(circles) // expected_rows
    rows = []

    for i in range(expected_rows):
        start_idx = i * circles_per_row
        end_idx = start_idx + circles_per_row

        # Handle last row (include remaining circles)
        if i == expected_rows - 1:
            end_idx = len(y_sorted_circles)

        row_circles = y_sorted_circles[start_idx:end_idx].tolist()
        rows.append(row_circles)

    return rows


def sort_circles_left_to_right_simple(circles: np.ndarray) -> np.ndarray:
    """Simple left-to-right sorting fallback."""
    if circles is None or len(circles) == 0:
        return circles

    if len(circles.shape) == 3:
        circles = circles[0]

    # Sort by X coordinate only
    sorted_indices = np.argsort(circles[:, 0])
    return circles[sorted_indices]


def detect_circles_and_create_masks(
    frame: np.ndarray,
    min_radius: int = 80,
    max_radius: int = 150,
    dp: float = 0.5,
    min_dist: int = 150,
    param1: int = 40,
    param2: int = 40,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Enhanced circle detection with automatic meandering sorting for multi-well plates.
    Now expects grayscale input and returns RGB labeled frame.
    """
    if not CV2_AVAILABLE:
        logger.error("OpenCV not available for circle detection")
        return [], (
            frame if frame is not None else np.zeros((100, 100, 3), dtype=np.uint8)
        )

    if frame is None:
        logger.error("Input frame is None")
        return [], np.zeros((100, 100, 3), dtype=np.uint8)

    try:
        # Validate frame
        if frame.size == 0:
            logger.error("Input frame is empty")
            return [], frame

        # Ensure we have grayscale input for circle detection
        if len(frame.shape) == 3:
            logger.warning(
                "Circle detection received RGB frame - converting to grayscale"
            )
            gray_frame = convert_to_grayscale(frame)
        else:
            gray_frame = frame.copy()

        # Validate grayscale frame
        if gray_frame.size == 0:
            logger.error("Grayscale conversion resulted in empty frame")
            return [], frame

        # Apply CLAHE enhancement with error handling
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_frame = clahe.apply(gray_frame)
        except Exception as e:
            logger.warning(f"CLAHE enhancement failed: {e}. Using original frame.")
            enhanced_frame = gray_frame

        # Detect circles with validation
        circles = cv2.HoughCircles(
            enhanced_frame,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        masks = []

        # Create labeled frame for visualization (always RGB)
        labeled_frame = convert_to_rgb_for_display(gray_frame)

        if circles is not None and len(circles[0]) > 0:
            circles = np.uint16(np.around(circles))

            # NEW: Apply automatic meandering sorting
            sorted_circles = sort_circles_meandering_auto(circles)

            # Log the sorting pattern used
            num_circles = len(sorted_circles)
            if num_circles in [6, 12, 24, 4, 8, 16]:
                if num_circles == 6:
                    pattern_info = "6-well plate (2x3) meandering"
                elif num_circles == 12:
                    pattern_info = "12-well plate (3x4) meandering"
                elif num_circles == 24:
                    pattern_info = "24-well plate (4x6) meandering"
                elif num_circles == 4:
                    pattern_info = "4-well (2x2) meandering"
                elif num_circles == 8:
                    pattern_info = "8-well (2x4) meandering"
                elif num_circles == 16:
                    pattern_info = "16-well (4x4) meandering"
                logger.info(f"Applied {pattern_info} sorting to {num_circles} ROIs")
            else:
                logger.info(
                    f"Applied simple left-to-right sorting to {num_circles} ROIs"
                )

            for idx, circle in enumerate(sorted_circles):
                try:
                    # Validate circle parameters
                    x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
                    if x < 0 or y < 0 or r <= 0:
                        logger.warning(f"Invalid circle parameters: ({x}, {y}, {r})")
                        continue

                    # Check if circle is within frame bounds
                    h, w = gray_frame.shape
                    if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
                        logger.warning(f"Circle {idx} extends beyond frame bounds")
                        # Adjust circle to fit within bounds
                        x = max(r, min(w - r - 1, x))
                        y = max(r, min(h - r - 1, y))

                    # Create mask for this ROI (grayscale)
                    mask = np.zeros(gray_frame.shape, dtype=np.uint8)
                    cv2.circle(mask, (x, y), r, 255, thickness=-1)
                    masks.append(mask)

                    # Draw circle on the labeled frame with enhanced visibility
                    cv2.circle(labeled_frame, (x, y), r, (0, 255, 0), 2)

                    # Add ROI number with better visibility
                    text = str(idx + 1)
                    font_scale = 1.5
                    thickness = 3

                    # Get text size for centering
                    (text_width, text_height), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )

                    # Add white background for better readability
                    cv2.rectangle(
                        labeled_frame,
                        (x - text_width // 2 - 5, y - text_height // 2 - 5),
                        (x + text_width // 2 + 5, y + text_height // 2 + 5),
                        (255, 255, 255),
                        -1,
                    )

                    # Add black text
                    cv2.putText(
                        labeled_frame,
                        text,
                        (x - text_width // 2, y + text_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),
                        thickness,
                    )

                except Exception as e:
                    logger.error(f"Error processing circle {idx}: {e}")
                    continue
        else:
            logger.warning("No circles detected with current parameters")

    except Exception as e:
        logger.error(f"Error in circle detection: {e}")
        labeled_frame = (
            convert_to_rgb_for_display(frame)
            if frame is not None
            else np.zeros((100, 100, 3), dtype=np.uint8)
        )

    return masks, labeled_frame


# =============================================================================
# CHUNK PROCESSING FUNCTIONS
# =============================================================================


def process_chunk(
    chunk_data: np.ndarray,
    masks: List[np.ndarray],
    start_time: float,
    frame_interval: float = 5,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Enhanced chunk processing - now expects grayscale input data.
    """
    roi_changes = {roi_idx + 1: [] for roi_idx in range(len(masks))}

    if len(chunk_data) < 2:
        return roi_changes

    try:
        logger.debug(
            f"Processing chunk shape: {chunk_data.shape}, dtype: {chunk_data.dtype}"
        )
        validate_data_ranges(chunk_data[0], "Raw_Input")

        # Data should already be grayscale from read_chunk_data_dual_structure
        # No need for RGB conversion here
        gray_frames = chunk_data.copy()

        logger.debug(
            f"Pre-normalization: dtype={gray_frames.dtype}, range={gray_frames.min()}-{gray_frames.max()}"
        )

        normalized_frames = normalize_image_to_float32(
            gray_frames, target_range=(0.0, 1.0)
        )

        logger.debug(
            f"Post-normalization: dtype={normalized_frames.dtype}, range={normalized_frames.min():.6f}-{normalized_frames.max():.6f}"
        )
        validate_data_ranges(normalized_frames[0], "Normalized_Output")

        # === ROI Processing ===
        for roi_idx, mask in enumerate(masks, start=1):
            try:
                if mask.size == 0:
                    logger.warning(f"Empty mask for ROI {roi_idx}")
                    continue

                mask_bool = mask > 0

                if mask_bool.shape != normalized_frames.shape[1:]:
                    logger.error(
                        f"Mask shape {mask_bool.shape} != frame shape {normalized_frames.shape[1:]}"
                    )
                    continue

                roi_intensities = []

                # Memory-efficient processing
                if normalized_frames.nbytes > 500 * 1024 * 1024:  # 500MB threshold
                    batch_size = max(1, min(50, len(normalized_frames) - 1))

                    for batch_start in range(0, len(normalized_frames) - 1, batch_size):
                        batch_end = min(
                            batch_start + batch_size + 1, len(normalized_frames)
                        )
                        batch_frames = normalized_frames[batch_start:batch_end]

                        for i in range(len(batch_frames) - 1):
                            frame_curr = batch_frames[i]
                            frame_next = batch_frames[i + 1]

                            diff_masked = np.abs(
                                frame_next[mask_bool] - frame_curr[mask_bool]
                            )
                            total_intensity = np.sum(diff_masked)
                            roi_intensities.append(total_intensity)
                else:
                    for i in range(len(normalized_frames) - 1):
                        frame_curr = normalized_frames[i]
                        frame_next = normalized_frames[i + 1]

                        diff_masked = np.abs(
                            frame_next[mask_bool] - frame_curr[mask_bool]
                        )
                        total_intensity = np.sum(diff_masked)
                        roi_intensities.append(total_intensity)

                time_array = start_time + frame_interval * np.arange(
                    len(roi_intensities)
                )
                roi_changes[roi_idx] = list(zip(time_array, roi_intensities))

                if roi_idx == 1 and roi_intensities:
                    roi_area = np.sum(mask_bool)
                    logger.info(
                        f"Enhanced ROI {roi_idx}: area={roi_area:,} pixels, "
                        f"intensity_range={np.min(roi_intensities):.6f}-{np.max(roi_intensities):.6f}"
                    )

            except Exception as e:
                logger.error(f"Error processing ROI {roi_idx}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error processing chunk: {e}")

    return roi_changes


# =============================================================================
# PARALLEL PROCESSING FUNCTIONS WITH DUAL STRUCTURE SUPPORT
# =============================================================================


def _process_single_chunk_dual_structure(
    args: Tuple,
) -> Tuple[int, Dict[int, List[Tuple[float, float]]]]:
    """
    Enhanced chunk processing that handles both HDF5 structures.
    """
    try:
        file_path, masks, start_idx, end_idx, frame_interval, structure_info = args

        # Read chunk using dual structure reader - request grayscale for processing
        chunk = read_chunk_data_dual_structure(
            file_path, start_idx, end_idx, structure_info, for_processing=True
        )

        if chunk is None:
            logger.error(f"Failed to read chunk {start_idx}:{end_idx}")
            return start_idx, {roi_idx + 1: [] for roi_idx in range(len(masks))}

        # Process chunk with existing logic
        chunk_results = process_chunk(
            chunk, masks, start_idx * frame_interval, frame_interval
        )
        return start_idx, chunk_results

    except Exception as e:
        logger.error(
            f"Error processing dual-structure chunk {start_idx}-{end_idx}: {e}"
        )
        return start_idx, {roi_idx + 1: [] for roi_idx in range(len(masks))}


def process_single_file_in_parallel_dual_structure(
    file_path: str,
    masks: List[np.ndarray],
    chunk_size: int = 50,
    progress_callback: Optional[Callable] = None,
    frame_interval: float = 5,
    num_processes: Optional[int] = None,
) -> Tuple[str, Dict, float]:
    """
    Process a single large HDF5 file using dual-structure support.
    """
    start_all = time.time()

    # Detect structure
    structure_info = detect_hdf5_structure_type(file_path)
    if structure_info["type"] == "error":
        logger.error(f"Cannot process file: {structure_info['error']}")
        return file_path, {}, 0.0

    logger.info(f"Processing {structure_info['type']} structure: {file_path}")
    logger.info(
        f"Frame count: {structure_info['frame_count']}, Shape: {structure_info['frame_shape']}"
    )

    num_frames = structure_info["frame_count"]
    if num_frames == 0:
        logger.error(f"No frames found in {file_path}")
        return file_path, {}, 0.0

    # Calculate optimal chunk size
    frame_shape = structure_info["frame_shape"]
    dtype_size = structure_info["dtype_size"]
    optimal_chunk_size = calculate_optimal_chunk_size(frame_shape, dtype_size)
    actual_chunk_size = max(MIN_CHUNK_SIZE, min(chunk_size, optimal_chunk_size))

    logger.info(
        f"Using chunk size: {actual_chunk_size} frames (structure: {structure_info['type']})"
    )

    total_chunks = (num_frames + actual_chunk_size - 1) // actual_chunk_size
    roi_changes = {roi_idx + 1: [] for roi_idx in range(len(masks))}

    if num_processes is None:
        if os.name == "nt":  # Windows
            num_processes = max(1, min(4, int(os.cpu_count() * 0.75)))
        else:
            num_processes = max(1, min(6, int(os.cpu_count() * 0.85)))

    # Prepare tasks with structure information
    tasks = []
    for start_idx in range(0, num_frames, actual_chunk_size):
        end_idx = min(start_idx + actual_chunk_size, num_frames)
        tasks.append(
            (file_path, masks, start_idx, end_idx, frame_interval, structure_info)
        )

    completed = 0

    try:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(_process_single_chunk_dual_structure, task)
                for task in tasks
            ]

            for future in as_completed(futures):
                try:
                    start_idx, chunk_res = future.result(timeout=600)

                    # Merge results
                    for roi_idx in chunk_res:
                        roi_changes[roi_idx].extend(chunk_res[roi_idx])

                    completed += 1

                    if progress_callback:
                        percent = (completed / total_chunks) * 100
                        msg = f"Dual-structure chunk {completed}/{total_chunks} for {os.path.basename(file_path)}"
                        progress_callback(percent, msg)

                except Exception as e:
                    logger.error(f"Error processing dual-structure chunk: {e}")
                    completed += 1

    except Exception as e:
        logger.error(f"ProcessPoolExecutor error (dual-structure): {e}")
        # Fallback to single-threaded processing
        logger.info("Falling back to single-threaded dual-structure processing")
        return process_hdf5_file_dual_structure(
            file_path, masks, actual_chunk_size, progress_callback, frame_interval
        )

    # Sort results by time
    for roi_idx in roi_changes:
        roi_changes[roi_idx].sort(key=lambda x: x[0])

    total_duration = (num_frames - 1) * frame_interval
    proc_time = time.time() - start_all
    logger.info(
        f"Dual-structure parallel processing complete: {file_path} processed in {proc_time:.2f}s"
    )

    if progress_callback:
        progress_callback(
            100, f"Completed dual-structure processing {os.path.basename(file_path)}"
        )

    return file_path, roi_changes, total_duration


def process_hdf5_file_dual_structure(
    file_path: str,
    masks: List[np.ndarray],
    chunk_size: int = 50,
    progress_callback: Optional[Callable] = None,
    frame_interval: float = 5,
) -> Tuple[str, Dict, float]:
    """
    Process a single HDF5 file with dual structure support (single-threaded).
    """
    start_all = time.time()

    # Detect structure
    structure_info = detect_hdf5_structure_type(file_path)
    if structure_info["type"] == "error":
        logger.error(f"Cannot process file: {structure_info['error']}")
        return file_path, {}, 0.0

    logger.info(f"Processing {structure_info['type']} structure: {file_path}")

    num_frames = structure_info["frame_count"]
    if num_frames == 0:
        logger.error(f"No frames found in {file_path}")
        return file_path, {}, 0.0

    # Calculate optimal chunk size
    frame_shape = structure_info["frame_shape"]
    dtype_size = structure_info["dtype_size"]
    optimal_chunk_size = calculate_optimal_chunk_size(frame_shape, dtype_size)
    actual_chunk_size = max(MIN_CHUNK_SIZE, min(chunk_size, optimal_chunk_size))

    total_chunks = (num_frames + actual_chunk_size - 1) // actual_chunk_size
    roi_changes = {roi_idx + 1: [] for roi_idx in range(len(masks))}

    last_frame = None

    for chunk_idx, start_idx in enumerate(range(0, num_frames, actual_chunk_size)):
        chunk_start = time.time()
        end_idx = min(start_idx + actual_chunk_size, num_frames)

        # Define msg outside the if block so it's always available
        msg = f"Dual-structure processing frames {start_idx} to {end_idx} ({chunk_idx+1}/{total_chunks} chunks)"

        if progress_callback:
            percent = (chunk_idx / total_chunks) * 100
            progress_callback(percent, msg)

        logger.info(msg)  # Now msg is always defined

        # Read chunk using dual structure method - request grayscale for processing
        try:
            this_chunk = read_chunk_data_dual_structure(
                file_path, start_idx, end_idx, structure_info, for_processing=True
            )

            if this_chunk is None:
                logger.error(f"Failed to read chunk {chunk_idx}")
                continue

        except Exception as e:
            logger.error(f"Error reading dual-structure chunk {chunk_idx}: {e}")
            continue

        # Prepend last frame for boundary diff if available
        if last_frame is not None:
            try:
                chunk = np.concatenate([last_frame[np.newaxis], this_chunk], axis=0)
                chunk_start_time = (start_idx - 1) * frame_interval
            except Exception as e:
                logger.warning(f"Error combining chunks: {e}")
                chunk = this_chunk
                chunk_start_time = start_idx * frame_interval
        else:
            chunk = this_chunk
            chunk_start_time = start_idx * frame_interval

        # Process chunk
        try:
            chunk_res = process_chunk(chunk, masks, chunk_start_time, frame_interval)
            for roi_idx in chunk_res:
                roi_changes[roi_idx].extend(chunk_res[roi_idx])
        except Exception as e:
            logger.error(f"Error processing dual-structure chunk {chunk_idx}: {e}")
            continue

        # Update last_frame for next iteration
        try:
            last_frame = this_chunk[-1].copy()
        except Exception as e:
            logger.warning(f"Error saving last frame: {e}")
            last_frame = None

        # Log chunk performance
        chunk_time = time.time() - chunk_start
        fps = (end_idx - start_idx) / chunk_time if chunk_time > 0 else 0
        logger.info(f"Dual-structure chunk {chunk_idx} processed at {fps:.2f} fps")

    # Sort results by timestamp
    for roi_idx in roi_changes:
        roi_changes[roi_idx].sort(key=lambda x: x[0])

    total_duration = (num_frames - 1) * frame_interval
    total_proc = time.time() - start_all
    logger.info(f"Dual-structure file processed in {total_proc:.2f} seconds")

    if progress_callback:
        progress_callback(
            100, f"Completed dual-structure processing {os.path.basename(file_path)}"
        )

    return file_path, roi_changes, total_duration


# =============================================================================
# ENHANCED PROCESSING FUNCTIONS
# =============================================================================


def process_hdf5_files(
    directory: str,
    masks: Optional[List[np.ndarray]] = None,
    num_processes: Optional[int] = None,
    chunk_size: int = 50,
    min_radius: int = 80,
    max_radius: int = 150,
    progress_callback: Optional[Callable] = None,
    frame_interval: float = 5,
    dp: float = 0.5,
    min_dist: int = 150,
    param1: int = 40,
    param2: int = 40,
) -> Tuple[Dict, Dict, List[np.ndarray], Optional[np.ndarray]]:
    """
    Enhanced HDF5 processing with automatic structure detection and format handling.
    """
    start_all = time.time()

    try:
        h5_files = sorted(
            [
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.lower().endswith((".h5", ".hdf5"))
                and os.path.isfile(os.path.join(directory, f))
            ]
        )
    except OSError as e:
        logger.error(f"Error reading directory {directory}: {e}")
        if progress_callback:
            progress_callback(0, f"Error reading directory: {e}")
        return {}, {}, [], None

    if not h5_files:
        logger.warning("No HDF5 files found in the directory.")
        if progress_callback:
            progress_callback(0, "No HDF5 files found.")
        return {}, {}, [], None

    # Enhanced: Detect structure of first file with format conversion
    if masks is None:
        first_file = h5_files[0]
        display_frame, processing_frame, structure_info = get_first_frame_enhanced(
            first_file
        )

        if display_frame is None or processing_frame is None:
            if progress_callback:
                progress_callback(
                    0, f"Error: Could not read the first frame of {first_file}"
                )
            logger.error(
                f"Structure detection failed: {structure_info.get('error', 'Unknown error')}"
            )
            return {}, {}, [], None

        logger.info(f"Detected HDF5 structure: {structure_info['type']}")
        logger.info(
            f"Frame count: {structure_info['frame_count']}, Data location: {structure_info['data_location']}"
        )
        logger.info(
            f"Image format: display={display_frame.shape}, processing={processing_frame.shape}"
        )

        # Use enhanced detection with widget-compatible parameters - use processing frame (grayscale)
        if CV2_AVAILABLE:
            masks, labeled_frame = detect_circles_and_create_masks(
                processing_frame, min_radius, max_radius, dp, min_dist, param1, param2
            )
        else:
            logger.warning("OpenCV not available. Cannot detect ROIs.")
            masks = []
            labeled_frame = display_frame  # Use RGB display frame

        if not masks:
            if progress_callback:
                progress_callback(
                    0, f"No ROIs detected in the first frame of {first_file}"
                )
            logger.warning(f"No ROIs detected in the first frame of {first_file}")
            return {}, {}, [], labeled_frame
    else:
        # Create labeled frame from existing masks
        first_file = h5_files[0]
        display_frame, processing_frame, structure_info = get_first_frame_enhanced(
            first_file
        )
        if display_frame is None:
            labeled_frame = None
        else:
            labeled_frame = display_frame.copy()  # Already RGB from enhanced reader

            # Draw existing masks on labeled frame
            for idx, mask in enumerate(masks):
                try:
                    if CV2_AVAILABLE:
                        contours, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if contours:
                            (x, y), radius = cv2.minEnclosingCircle(contours[0])
                            center = (int(x), int(y))
                            radius = int(radius)
                            cv2.circle(labeled_frame, center, radius, (0, 255, 0), 2)
                            cv2.putText(
                                labeled_frame,
                                f"{idx+1}",
                                (center[0] - 20, center[1] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (0, 0, 255),
                                3,
                            )
                except Exception as e:
                    logger.warning(f"Error drawing mask {idx}: {e}")

    # Memory and performance optimization
    if num_processes is None:
        available_memory_gb = get_available_memory() / (1024**3)
        if available_memory_gb < 4:  # Less than 4GB RAM
            num_processes = 1
        elif available_memory_gb < 8:  # Less than 8GB RAM
            num_processes = max(1, min(3, int(os.cpu_count() * 0.6)))
        else:
            num_processes = max(1, min(6, int(os.cpu_count() * 0.8)))

        logger.info(
            f"Auto-selected {num_processes} processes based on {available_memory_gb:.1f}GB available memory and {len(masks)} ROIs"
        )

    # Get optimal chunk size from first file
    try:
        frame_shape = structure_info["frame_shape"]
        dtype_size = structure_info["dtype_size"]
        optimal_chunk_size = calculate_optimal_chunk_size(frame_shape, dtype_size)
        actual_chunk_size = max(MIN_CHUNK_SIZE, min(chunk_size, optimal_chunk_size))
        logger.info(
            f"Adjusted chunk size to {actual_chunk_size} based on memory constraints and {len(masks)} ROIs"
        )
    except Exception as e:
        logger.warning(f"Could not determine optimal chunk size: {e}")
        actual_chunk_size = max(MIN_CHUNK_SIZE, chunk_size)

    # Processing strategy based on number of files and memory
    if len(h5_files) == 1:
        if progress_callback:
            progress_callback(
                0,
                f"Single file found; using dual-structure chunk-level parallelism with {num_processes} processes.",
            )
        file_path = h5_files[0]
        file_path, roi_changes, total_duration = (
            process_single_file_in_parallel_dual_structure(
                file_path,
                masks,
                chunk_size=actual_chunk_size,
                progress_callback=progress_callback,
                frame_interval=frame_interval,
                num_processes=num_processes,
            )
        )
        results = {file_path: roi_changes}
        durations = {file_path: total_duration}
        processed_files = 1
    else:
        if progress_callback:
            progress_callback(
                0,
                f"Multiple files found; processing with dual-structure method using {num_processes} processes.",
            )
        results = {}
        durations = {}
        processed_files = 0

        def file_progress_callback(file_idx: int, total_files: int, filename: str):
            def callback(percent: float, message: str):
                overall = ((file_idx - 1) + (percent / 100)) / total_files * 100
                if progress_callback:
                    progress_callback(
                        overall, f"File {file_idx}/{total_files}: {message}"
                    )

            return callback

        if num_processes == 1:
            # Single-threaded processing
            for file_idx, file_path in enumerate(h5_files, 1):
                fp_callback = file_progress_callback(file_idx, len(h5_files), file_path)
                file_path, roi_changes, tot_dur = process_hdf5_file_dual_structure(
                    file_path, masks, actual_chunk_size, fp_callback, frame_interval
                )
                if roi_changes:
                    processed_files += 1
                    results[file_path] = roi_changes
                    durations[file_path] = tot_dur
        else:
            # Multi-threaded processing with memory management
            try:
                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    futures = []
                    for idx, file_path in enumerate(h5_files, 1):
                        future = executor.submit(
                            process_hdf5_file_dual_structure,
                            file_path,
                            masks,
                            actual_chunk_size,
                            None,
                            frame_interval,
                        )
                        futures.append((future, file_path, idx))

                    for future, file_path, file_idx in futures:
                        try:
                            file_path, roi_changes, tot_dur = future.result(
                                timeout=3600
                            )  # 1 hour timeout
                            if roi_changes:
                                processed_files += 1
                                results[file_path] = roi_changes
                                durations[file_path] = tot_dur
                            if progress_callback:
                                fraction = (file_idx / len(h5_files)) * 100
                                progress_callback(
                                    fraction,
                                    f"Completed {file_idx}/{len(h5_files)}: {os.path.basename(file_path)}",
                                )
                        except Exception as e:
                            logger.error(f"Error processing {file_path}: {e}")
                            if progress_callback:
                                fraction = (file_idx / len(h5_files)) * 100
                                progress_callback(
                                    fraction,
                                    f"Error processing {os.path.basename(file_path)}: {e}",
                                )
            except Exception as e:
                logger.error(
                    f"Multi-processing failed: {e}. Falling back to single-threaded processing."
                )
                # Fallback to single-threaded
                for file_idx, file_path in enumerate(h5_files, 1):
                    fp_callback = file_progress_callback(
                        file_idx, len(h5_files), file_path
                    )
                    try:
                        file_path, roi_changes, tot_dur = (
                            process_hdf5_file_dual_structure(
                                file_path,
                                masks,
                                actual_chunk_size,
                                fp_callback,
                                frame_interval,
                            )
                        )
                        if roi_changes:
                            processed_files += 1
                            results[file_path] = roi_changes
                            durations[file_path] = tot_dur
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")

    elapsed_all = time.time() - start_all
    logger.info(
        f"Finished processing in {elapsed_all:.2f}s, processed {processed_files}/{len(h5_files)} files total."
    )
    if progress_callback:
        progress_callback(
            100,
            f"Analysis complete. Processed {processed_files}/{len(h5_files)} files.",
        )

    return results, durations, masks, labeled_frame


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def merge_results(
    results: Dict[str, Dict], durations: Dict[str, float]
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Merge results from multiple files into a continuous timeline with memory optimization.
    """
    merged_results = {}
    cumulative_time = 0.0
    sorted_paths = sorted(results.keys())

    for path in sorted_paths:
        roi_changes = results[path]
        for roi, pairs in roi_changes.items():
            if roi not in merged_results:
                merged_results[roi] = []

            # Memory-efficient merging for large datasets
            if len(pairs) > 10000:  # Large dataset threshold
                # Process in batches to avoid memory spikes
                batch_size = 5000
                for i in range(0, len(pairs), batch_size):
                    batch = pairs[i : i + batch_size]
                    adjusted_batch = [
                        (t_sec + cumulative_time, val) for (t_sec, val) in batch
                    ]
                    merged_results[roi].extend(adjusted_batch)
            else:
                # Standard processing for smaller datasets
                for t_sec, val in pairs:
                    merged_results[roi].append((t_sec + cumulative_time, val))

        cumulative_time += durations[path]

    return merged_results


def get_roi_colors(rois: List[int]) -> Dict[int, str]:
    """
    Generate consistent colors for ROIs with fallback when matplotlib is not available.
    """
    roi_colors = {}

    if MATPLOTLIB_AVAILABLE:
        try:
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        except Exception:
            # Fallback color cycle
            color_cycle = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]
    else:
        # Default color cycle when matplotlib is not available
        color_cycle = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

    for i, roi_id in enumerate(rois):
        roi_colors[roi_id] = color_cycle[i % len(color_cycle)]

    return roi_colors


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    if PSUTIL_AVAILABLE:
        try:
            memory = psutil.virtual_memory()
            return {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent_used": memory.percent,
            }
        except Exception:
            pass

    return {
        "total_gb": "unknown",
        "available_gb": "unknown",
        "used_gb": "unknown",
        "percent_used": "unknown",
    }


def log_memory_usage(context: str = ""):
    """Log current memory usage for debugging."""
    memory_info = get_memory_usage()
    if memory_info["percent_used"] != "unknown":
        logger.info(
            f"Memory usage {context}: {memory_info['used_gb']:.1f}GB/{memory_info['total_gb']:.1f}GB "
            f"({memory_info['percent_used']:.1f}% used, {memory_info['available_gb']:.1f}GB available)"
        )
    else:
        logger.info(f"Memory usage {context}: unable to determine")


def cleanup_large_arrays(*arrays):
    """Explicitly cleanup large numpy arrays to help with memory management."""
    for arr in arrays:
        if hasattr(arr, "nbytes") and arr.nbytes > 100 * 1024 * 1024:  # 100MB threshold
            logger.debug(f"Cleaning up large array: {arr.nbytes / (1024*1024):.1f}MB")
        del arr


# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================


def diagnose_preprocessing_impact(chunk_data: np.ndarray, masks: List[np.ndarray]):
    """
    Diagnostic function for preprocessing impact analysis.
    """
    if len(chunk_data) < 2 or len(masks) == 0:
        return

    print("=== PREPROCESSING DIAGNOSIS ===")

    try:
        # Data should already be grayscale from read_chunk_data_dual_structure
        gray_frames = chunk_data[:2].copy()

        print(f"Original frame dtype: {gray_frames.dtype}")
        print(f"Original frame range: {np.min(gray_frames)} to {np.max(gray_frames)}")

        # Normalization
        normalized_frames = normalize_image_to_float32(gray_frames)

        print(
            f"After normalization range: {np.min(normalized_frames):.6f} to {np.max(normalized_frames):.6f}"
        )

        # Test processing on first ROI
        mask = masks[0]
        mask_bool = mask > 0

        # Calculate difference
        frame_curr = normalized_frames[0]
        frame_next = normalized_frames[1]
        diff_masked = np.abs(frame_next[mask_bool] - frame_curr[mask_bool])
        total_intensity = np.sum(diff_masked)

        roi_area = np.sum(mask_bool)

        print(f"\nROI Analysis (first ROI):")
        print(f"ROI area: {roi_area:,} pixels")
        print(f"Total intensity change: {total_intensity:.6f}")
        print(f"Per-pixel change: {total_intensity/roi_area:.8f}")
        print(f"Expected value range: 0.001 - 10 (compatible with analysis pipeline)")

    except Exception as e:
        print(f"Error in preprocessing diagnosis: {e}")


# =============================================================================
# LEGACY FUNCTIONS FOR BACKWARD COMPATIBILITY
# =============================================================================


def _process_single_chunk(
    args: Tuple,
) -> Tuple[int, Dict[int, List[Tuple[float, float]]]]:
    """
    Legacy chunk processing function - redirects to dual structure version.
    """
    try:
        if len(args) == 6:
            # New format with structure_info
            return _process_single_chunk_dual_structure(args)
        else:
            # Old format - need to detect structure
            file_path, masks, start_idx, end_idx, frame_interval = args
            structure_info = detect_hdf5_structure_type(file_path)
            new_args = (
                file_path,
                masks,
                start_idx,
                end_idx,
                frame_interval,
                structure_info,
            )
            return _process_single_chunk_dual_structure(new_args)

    except Exception as e:
        logger.error(f"Error in legacy chunk processing: {e}")
        return 0, {}


def process_single_file_in_parallel(
    file_path: str,
    masks: List[np.ndarray],
    chunk_size: int = 50,
    progress_callback: Optional[Callable] = None,
    frame_interval: float = 5,
    num_processes: Optional[int] = None,
) -> Tuple[str, Dict, float]:
    """
    Legacy function - redirects to dual structure version.
    """
    return process_single_file_in_parallel_dual_structure(
        file_path, masks, chunk_size, progress_callback, frame_interval, num_processes
    )


def process_hdf5_file(
    file_path: str,
    masks: List[np.ndarray],
    chunk_size: int = 50,
    progress_callback: Optional[Callable] = None,
    frame_interval: float = 5,
) -> Tuple[str, Dict, float]:
    """
    Legacy function - redirects to dual structure version.
    """
    return process_hdf5_file_dual_structure(
        file_path, masks, chunk_size, progress_callback, frame_interval
    )
