# """
# _metadata.py - Enhanced HDF5 Metadata extraction module with Nematostella Timeseries Analysis

# This module handles extraction and formatting of HDF5 metadata for the analysis plugin.
# Enhanced with specialized Nematostella timeseries analysis capabilities and comprehensive Excel export.
# """

# import h5py
# import os
# import json
# import numpy as np
# from typing import Dict, List, Tuple, Optional, Any
# from datetime import datetime
# import csv
# try:
#     import pandas as pd
# except ImportError:
#     pd = None

# # ===================================================================
# # EXISTING METADATA FUNCTIONS (PRESERVED)
# # ===================================================================

# def extract_hdf5_metadata(file_path: str) -> Dict[str, Any]:
#     """
#     Extract comprehensive metadata from HDF5 file.

#     Args:
#         file_path: Path to HDF5 file

#     Returns:
#         Dictionary containing all metadata
#     """
#     metadata = {
#         'file_info': {},
#         'datasets': {},
#         'groups': {},
#         'attributes': {},
#         'experimental_info': {},
#         'technical_info': {},
#         'extraction_info': {
#             'extracted_at': datetime.now().isoformat(),
#             'file_path': file_path,
#             'file_name': os.path.basename(file_path)
#         }
#     }

#     if not os.path.exists(file_path):
#         metadata['extraction_info']['error'] = f"File not found: {file_path}"
#         return metadata

#     try:
#         with h5py.File(file_path, 'r') as f:
#             # File-level information
#             metadata['file_info'] = {
#                 'filename': os.path.basename(file_path),
#                 'file_size_bytes': os.path.getsize(file_path),
#                 'file_size_mb': round(os.path.getsize(file_path) / (1024*1024), 2),
#                 'hdf5_version': f.libver,
#                 'driver': f.driver if hasattr(f, 'driver') else 'default'
#             }

#             # Extract file-level attributes
#             metadata['attributes']['file_level'] = dict(f.attrs)

#             # Recursively extract all groups and datasets
#             def extract_item_metadata(name, obj):
#                 item_info = {
#                     'name': name,
#                     'type': type(obj).__name__,
#                     'attributes': dict(obj.attrs)
#                 }

#                 if isinstance(obj, h5py.Dataset):
#                     item_info.update({
#                         'shape': obj.shape,
#                         'dtype': str(obj.dtype),
#                         'size': obj.size,
#                         'ndim': obj.ndim,
#                         'chunks': obj.chunks,
#                         'compression': obj.compression,
#                         'compression_opts': obj.compression_opts,
#                         'shuffle': obj.shuffle,
#                         'fletcher32': obj.fletcher32,
#                         'fillvalue': obj.fillvalue
#                     })
#                     metadata['datasets'][name] = item_info

#                 elif isinstance(obj, h5py.Group):
#                     item_info.update({
#                         'keys': list(obj.keys()),
#                         'length': len(obj)
#                     })
#                     metadata['groups'][name] = item_info

#             # Visit all items in the file
#             f.visititems(extract_item_metadata)

#             # Extract experimental metadata
#             experimental_attrs = {}
#             for key, value in f.attrs.items():
#                 key_lower = key.lower()
#                 if any(exp_key in key_lower for exp_key in [
#                     'experiment', 'date', 'time', 'duration', 'interval',
#                     'frame', 'fps', 'resolution', 'microscope', 'objective',
#                     'magnification', 'exposure', 'gain', 'temperature',
#                     'humidity', 'light', 'treatment', 'condition', 'protocol',
#                     'researcher', 'lab', 'species', 'strain', 'age', 'sex'
#                 ]):
#                     experimental_attrs[key] = value

#             metadata['experimental_info'] = experimental_attrs

#             # Technical information
#             if 'frames' in f:
#                 frames_dataset = f['frames']
#                 metadata['technical_info'] = {
#                     'total_frames': len(frames_dataset),
#                     'frame_shape': frames_dataset.shape[1:] if len(frames_dataset.shape) > 1 else (),
#                     'frame_dtype': str(frames_dataset.dtype),
#                 }

#                 # Try to extract timing information
#                 if 'timestamps' in f:
#                     timestamps = f['timestamps'][:]
#                     if len(timestamps) > 1:
#                         intervals = np.diff(timestamps[:10])
#                         metadata['technical_info']['detected_frame_intervals'] = {
#                             'mean_interval': float(np.mean(intervals)),
#                             'std_interval': float(np.std(intervals)),
#                         }

#     except Exception as e:
#         metadata['extraction_info']['error'] = f"Error extracting metadata: {str(e)}"

#     return metadata


# def extract_hdf5_metadata_timeseries(file_path: str) -> Dict[str, Any]:
#     """Extract time-series data from JSON metadata attributes and dataset groups."""
#     metadata = extract_hdf5_metadata(file_path)

#     if not os.path.exists(file_path):
#         metadata['timeseries_data'] = {}
#         return metadata

#     try:
#         with h5py.File(file_path, 'r') as f:
#             timeseries_metadata = {}

#             # Method 1: Extract from JSON frame metadata (primary method)
#             if 'frames' in f:
#                 frames_dataset = f['frames']
#                 json_data = _extract_from_json_metadata(frames_dataset)
#                 timeseries_metadata.update(json_data)

#             # Method 2: Extract from separate timeseries datasets (fallback)
#             if 'timeseries' in f:
#                 timeseries_group = f['timeseries']
#                 for dataset_name in timeseries_group.keys():
#                     try:
#                         dataset = timeseries_group[dataset_name]

#                         # Handle scalar datasets
#                         if dataset.shape == ():
#                             data = dataset[()]
#                             timeseries_metadata[dataset_name] = [data]
#                         else:
#                             data = dataset[:]
#                             if len(data) > 0:
#                                 # Only add if not already found in JSON metadata
#                                 if dataset_name not in timeseries_metadata:
#                                     timeseries_metadata[dataset_name] = data.tolist()
#                     except Exception as e:
#                         metadata['extraction_info'][f'timeseries_error_{dataset_name}'] = str(e)

#             # Method 3: Check root level datasets
#             for dataset_name in f.keys():
#                 if dataset_name not in ['frames', 'images', 'metadata', 'timeseries']:
#                     try:
#                         dataset = f[dataset_name]
#                         if dataset.shape == ():
#                             data = dataset[()]
#                             if dataset_name not in timeseries_metadata:
#                                 timeseries_metadata[dataset_name] = [data]
#                         else:
#                             data = dataset[:]
#                             if len(data) > 0 and dataset_name not in timeseries_metadata:
#                                 timeseries_metadata[dataset_name] = data.tolist()
#                     except Exception as e:
#                         metadata['extraction_info'][f'root_error_{dataset_name}'] = str(e)

#             # Summary
#             metadata['timeseries_data'] = timeseries_metadata
#             metadata['timeseries_summary'] = {
#                 'total_parameters': len(timeseries_metadata),
#                 'parameters_found': list(timeseries_metadata.keys()),
#                 'max_length': max(len(data) if hasattr(data, '__len__') else 1
#                                 for data in timeseries_metadata.values()) if timeseries_metadata else 0,
#                 'data_types': {param: str(type(data[0]).__name__) if data and hasattr(data, '__len__') and len(data) > 0 else 'empty'
#                              for param, data in timeseries_metadata.items()}
#             }

#     except Exception as e:
#         metadata['extraction_info']['timeseries_extraction_error'] = str(e)
#         metadata['timeseries_data'] = {}

#     return metadata

# # ===================================================================
# # ENHANCED NEMATOSTELLA TIMESERIES ANALYSIS (NEW)
# # ===================================================================

# class NematostellaTimeseriesAnalyzer:
#     """
#     Enhanced timeseries analyzer specifically for Nematostella experiments.
#     Integrates with existing napari plugin architecture.
#     """

#     def __init__(self, file_path: str):
#         """Initialize analyzer with HDF5 file path."""
#         self.file_path = file_path
#         self.timeseries_data = {}
#         self.metadata = {}
#         self.analysis_results = {}

#     def extract_and_analyze_timeseries(self) -> pd.DataFrame:
#         """
#         Extract and analyze all timeseries data from HDF5 file.
#         Returns comprehensive DataFrame with all timeseries.
#         """
#         if pd is None:
#             raise ImportError("pandas is required for timeseries analysis")

#         timeseries_dict = {}

#         with h5py.File(self.file_path, 'r') as h5_file:
#             # Extract root metadata
#             self.metadata = dict(h5_file.attrs)

#             # Extract timeseries group metadata if exists
#             if 'timeseries' in h5_file:
#                 ts_group = h5_file['timeseries']
#                 self.timeseries_metadata = dict(ts_group.attrs)

#                 # Extract all timeseries datasets
#                 for key in ts_group.keys():
#                     dataset = ts_group[key]

#                     try:
#                         if dataset.shape == ():
#                             # Scalar dataset
#                             data = dataset[()]
#                             timeseries_dict[key] = [data]  # Make it a list for DataFrame
#                         else:
#                             # Array dataset
#                             data = dataset[:]
#                             timeseries_dict[key] = data

#                         # Store dataset attributes
#                         if dataset.attrs:
#                             attrs = dict(dataset.attrs)
#                             print(f"Nematostella analyzer: {key} attributes: {attrs}")

#                     except Exception as e:
#                         print(f"Warning: Could not read timeseries dataset '{key}': {e}")

#         # Create DataFrame
#         df = pd.DataFrame(timeseries_dict)

#         # Convert timestamps to datetime if they exist
#         if 'capture_timestamps' in df.columns:
#             df['capture_datetime'] = pd.to_datetime(df['capture_timestamps'], unit='s')
#         if 'expected_timestamps' in df.columns:
#             df['expected_datetime'] = pd.to_datetime(df['expected_timestamps'], unit='s')

#         return df

#     def analyze_timing_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
#         """Analyze timing-related data from the timeseries."""
#         analysis = {}

#         if 'actual_intervals' in df.columns and 'expected_intervals' in df.columns:
#             analysis['timing'] = {
#                 'mean_actual_interval': df['actual_intervals'].mean(),
#                 'mean_expected_interval': df['expected_intervals'].mean(),
#                 'interval_std': df['actual_intervals'].std(),
#                 'timing_accuracy': 1 - (df['actual_intervals'] - df['expected_intervals']).abs().mean() / df['expected_intervals'].mean()
#             }

#         if 'frame_drift' in df.columns:
#             analysis['drift'] = {
#                 'max_drift': df['frame_drift'].max(),
#                 'min_drift': df['frame_drift'].min(),
#                 'mean_drift': df['frame_drift'].mean(),
#                 'drift_trend': np.polyfit(range(len(df)), df['frame_drift'], 1)[0]  # Linear trend
#             }

#         if 'cumulative_drift' in df.columns:
#             analysis['cumulative_drift'] = {
#                 'final_drift': df['cumulative_drift'].iloc[-1],
#                 'max_cumulative_drift': df['cumulative_drift'].max(),
#                 'min_cumulative_drift': df['cumulative_drift'].min()
#             }

#         return analysis

#     def analyze_image_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
#         """Analyze image statistics from the timeseries."""
#         analysis = {}

#         image_stats = ['frame_mean', 'frame_std', 'frame_min', 'frame_max']
#         available_stats = [stat for stat in image_stats if stat in df.columns]

#         if available_stats:
#             analysis['image_statistics'] = {}
#             for stat in available_stats:
#                 analysis['image_statistics'][stat] = {
#                     'mean': df[stat].mean(),
#                     'std': df[stat].std(),
#                     'min': df[stat].min(),
#                     'max': df[stat].max(),
#                     'trend': np.polyfit(range(len(df)), df[stat], 1)[0]
#                 }

#         return analysis

#     def analyze_environmental_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
#         """Analyze environmental data from the timeseries."""
#         analysis = {}

#         env_vars = ['temperature', 'humidity']
#         available_vars = [var for var in env_vars if var in df.columns]

#         if available_vars:
#             analysis['environment'] = {}
#             for var in available_vars:
#                 analysis['environment'][var] = {
#                     'mean': df[var].mean(),
#                     'std': df[var].std(),
#                     'min': df[var].min(),
#                     'max': df[var].max(),
#                     'range': df[var].max() - df[var].min()
#                 }

#         return analysis

#     def analyze_led_system(self, df: pd.DataFrame) -> Dict[str, Any]:
#         """Analyze LED-related data from the timeseries."""
#         analysis = {}

#         if 'led_power_percent' in df.columns:
#             analysis['led_power'] = {
#                 'mean': df['led_power_percent'].mean(),
#                 'std': df['led_power_percent'].std(),
#                 'min': df['led_power_percent'].min(),
#                 'max': df['led_power_percent'].max()
#             }

#         if 'led_duration_ms' in df.columns:
#             analysis['led_duration'] = {
#                 'mean': df['led_duration_ms'].mean(),
#                 'std': df['led_duration_ms'].std(),
#                 'min': df['led_duration_ms'].min(),
#                 'max': df['led_duration_ms'].max()
#             }

#         if 'led_sync_success' in df.columns:
#             analysis['led_sync'] = {
#                 'success_rate': df['led_sync_success'].mean(),
#                 'total_frames': len(df),
#                 'successful_syncs': df['led_sync_success'].sum(),
#                 'failed_syncs': (~df['led_sync_success']).sum()
#             }

#         return analysis

#     def generate_comprehensive_report(self) -> str:
#         """Generate comprehensive analysis report for Nematostella experiment."""
#         print("Nematostella Analyzer: Extracting timeseries data...")
#         df = self.extract_and_analyze_timeseries()

#         print("Nematostella Analyzer: Running specialized analysis...")
#         timing_analysis = self.analyze_timing_performance(df)
#         image_analysis = self.analyze_image_statistics(df)
#         env_analysis = self.analyze_environmental_conditions(df)
#         led_analysis = self.analyze_led_system(df)

#         # Store analysis results for export
#         self.analysis_results = {
#             'dataframe': df,
#             'timing_analysis': timing_analysis,
#             'image_analysis': image_analysis,
#             'env_analysis': env_analysis,
#             'led_analysis': led_analysis
#         }

#         # Generate report text
#         report = f"""
# # NEMATOSTELLA TIMELAPSE ANALYSIS REPORT
# {'='*50}

# ## File Information
# - File: {self.file_path}
# - Total Frames: {len(df)}
# - Data Columns: {len(df.columns)}

# ## Root Metadata
# """

#         for key, value in self.metadata.items():
#             report += f"- {key}: {value}\n"

#         if hasattr(self, 'timeseries_metadata'):
#             report += "\n## Timeseries Metadata\n"
#             for key, value in self.timeseries_metadata.items():
#                 report += f"- {key}: {value}\n"

#         # Timing Analysis
#         if timing_analysis:
#             report += "\n## Timing Analysis\n"
#             if 'timing' in timing_analysis:
#                 timing = timing_analysis['timing']
#                 report += f"- Mean Actual Interval: {timing['mean_actual_interval']:.4f} seconds\n"
#                 report += f"- Mean Expected Interval: {timing['mean_expected_interval']:.4f} seconds\n"
#                 report += f"- Interval Standard Deviation: {timing['interval_std']:.4f} seconds\n"
#                 report += f"- Timing Accuracy: {timing['timing_accuracy']:.2%}\n"

#             if 'drift' in timing_analysis:
#                 drift = timing_analysis['drift']
#                 report += f"- Mean Drift: {drift['mean_drift']:.4f} seconds\n"
#                 report += f"- Max Drift: {drift['max_drift']:.4f} seconds\n"
#                 report += f"- Drift Trend: {drift['drift_trend']:.6f} seconds/frame\n"

#         # Image Analysis
#         if image_analysis and 'image_statistics' in image_analysis:
#             report += "\n## Image Statistics Analysis\n"
#             for stat, values in image_analysis['image_statistics'].items():
#                 report += f"### {stat.replace('_', ' ').title()}\n"
#                 report += f"- Mean: {values['mean']:.2f}\n"
#                 report += f"- Std: {values['std']:.2f}\n"
#                 report += f"- Range: {values['min']:.2f} - {values['max']:.2f}\n"
#                 report += f"- Trend: {values['trend']:.6f} units/frame\n\n"

#         # Environmental Analysis
#         if env_analysis and 'environment' in env_analysis:
#             report += "## Environmental Conditions\n"
#             for var, values in env_analysis['environment'].items():
#                 report += f"### {var.title()}\n"
#                 report += f"- Mean: {values['mean']:.2f}\n"
#                 report += f"- Range: {values['min']:.2f} - {values['max']:.2f}\n"
#                 report += f"- Variability: {values['std']:.2f}\n\n"

#         # LED Analysis
#         if led_analysis:
#             report += "## LED System Analysis\n"
#             if 'led_sync' in led_analysis:
#                 sync = led_analysis['led_sync']
#                 report += f"- Sync Success Rate: {sync['success_rate']:.2%}\n"
#                 report += f"- Successful Syncs: {sync['successful_syncs']}/{sync['total_frames']}\n"

#             if 'led_power' in led_analysis:
#                 power = led_analysis['led_power']
#                 report += f"- Mean LED Power: {power['mean']:.1f}%\n"
#                 report += f"- LED Power Range: {power['min']:.1f}% - {power['max']:.1f}%\n"

#         return report

#     def export_to_excel_enhanced(self, filename: str = "nematostella_analysis.xlsx") -> List[str]:
#         """
#         Export comprehensive analysis to Excel with multiple sheets.
#         Returns list of created sheet names.
#         """
#         if pd is None:
#             raise ImportError("pandas and openpyxl are required for Excel export")

#         if not hasattr(self, 'analysis_results') or not self.analysis_results:
#             # Run analysis first
#             self.generate_comprehensive_report()

#         df = self.analysis_results['dataframe']
#         created_sheets = []

#         with pd.ExcelWriter(filename, engine='openpyxl') as writer:
#             # Sheet 1: Complete timeseries data
#             df.to_excel(writer, sheet_name='All_Timeseries_Data', index=False)
#             created_sheets.append('All_Timeseries_Data')

#             # Sheet 2-N: Individual timeseries parameters
#             for column in df.columns:
#                 if 'datetime' in column.lower():
#                     continue  # Skip datetime derivatives

#                 # Create DataFrame for this parameter
#                 param_df = pd.DataFrame({
#                     'Index': range(len(df)),
#                     column: df[column]
#                 })

#                 # Add timestamp columns if they exist
#                 if 'capture_timestamps' in df.columns:
#                     param_df['capture_timestamps'] = df['capture_timestamps']
#                 if 'capture_datetime' in df.columns:
#                     param_df['capture_datetime'] = df['capture_datetime']

#                 # Clean sheet name for Excel
#                 sheet_name = clean_sheet_name(column)

#                 try:
#                     param_df.to_excel(writer, sheet_name=sheet_name, index=False)
#                     created_sheets.append(sheet_name)
#                 except Exception as e:
#                     print(f"Warning: Could not create sheet for {column}: {e}")

#             # Summary sheet with analysis results
#             if self.analysis_results:
#                 summary_data = []

#                 # Add timing metrics
#                 if 'timing_analysis' in self.analysis_results and self.analysis_results['timing_analysis']:
#                     timing = self.analysis_results['timing_analysis']
#                     for category, metrics in timing.items():
#                         if isinstance(metrics, dict):
#                             for metric, value in metrics.items():
#                                 summary_data.append({
#                                     'Category': f'Timing - {category}',
#                                     'Metric': metric,
#                                     'Value': value
#                                 })

#                 # Add environmental metrics
#                 if 'env_analysis' in self.analysis_results and self.analysis_results['env_analysis']:
#                     env = self.analysis_results['env_analysis']
#                     for category, metrics in env.items():
#                         if isinstance(metrics, dict):
#                             for var, values in metrics.items():
#                                 for metric, value in values.items():
#                                     summary_data.append({
#                                         'Category': f'Environment - {var}',
#                                         'Metric': metric,
#                                         'Value': value
#                                     })

#                 # Add LED metrics
#                 if 'led_analysis' in self.analysis_results and self.analysis_results['led_analysis']:
#                     led = self.analysis_results['led_analysis']
#                     for category, metrics in led.items():
#                         if isinstance(metrics, dict):
#                             for metric, value in metrics.items():
#                                 summary_data.append({
#                                     'Category': f'LED - {category}',
#                                     'Metric': metric,
#                                     'Value': value
#                                 })

#                 if summary_data:
#                     summary_df = pd.DataFrame(summary_data)
#                     summary_df.to_excel(writer, sheet_name='Analysis_Summary', index=False)
#                     created_sheets.append('Analysis_Summary')

#         print(f"Nematostella Analyzer: Created Excel file with {len(created_sheets)} sheets")
#         return created_sheets

# # ===================================================================
# # INTEGRATION FUNCTIONS FOR NAPARI WIDGET
# # ===================================================================

# def analyze_nematostella_hdf5_file(file_path: str) -> Dict[str, Any]:
#     """
#     Integration function for napari widget to analyze Nematostella HDF5 files.

#     Args:
#         file_path: Path to HDF5 file

#     Returns:
#         Dictionary containing analysis results and export paths
#     """
#     try:
#         # Initialize analyzer
#         analyzer = NematostellaTimeseriesAnalyzer(file_path)

#         # Generate comprehensive report
#         report = analyzer.generate_comprehensive_report()

#         # Export to Excel
#         base_filename = os.path.splitext(os.path.basename(file_path))[0]
#         excel_filename = f"nematostella_analysis_{base_filename}.xlsx"
#         created_sheets = analyzer.export_to_excel_enhanced(excel_filename)

#         # Save text report
#         report_filename = f"nematostella_analysis_{base_filename}.txt"
#         with open(report_filename, 'w') as f:
#             f.write(report)

#         return {
#             'success': True,
#             'report': report,
#             'excel_file': excel_filename,
#             'report_file': report_filename,
#             'sheets_created': created_sheets,
#             'timeseries_data': analyzer.analysis_results.get('dataframe'),
#             'analysis_results': analyzer.analysis_results
#         }

#     except Exception as e:
#         return {
#             'success': False,
#             'error': str(e),
#             'report': f"Analysis failed: {e}"
#         }

# def get_nematostella_timeseries_summary(file_path: str) -> str:
#     """
#     Quick summary function for napari widget to show what timeseries data is available.

#     Args:
#         file_path: Path to HDF5 file

#     Returns:
#         String summary of available timeseries data
#     """
#     try:
#         with h5py.File(file_path, 'r') as h5_file:
#             if 'timeseries' not in h5_file:
#                 return "No 'timeseries' group found in HDF5 file."

#             ts_group = h5_file['timeseries']
#             summary = f"Found {len(ts_group.keys())} timeseries datasets:\n"

#             for key in ts_group.keys():
#                 dataset = ts_group[key]
#                 summary += f"- {key}: shape {dataset.shape}, dtype {dataset.dtype}\n"

#                 if dataset.attrs:
#                     attrs = dict(dataset.attrs)
#                     if 'description' in attrs:
#                         summary += f"  Description: {attrs['description']}\n"
#                     if 'units' in attrs:
#                         summary += f"  Units: {attrs['units']}\n"

#             return summary

#     except Exception as e:
#         return f"Error reading timeseries summary: {e}"

# # ===================================================================
# # EXISTING HELPER FUNCTIONS (PRESERVED AND EXTENDED)
# # ===================================================================

# def _extract_from_json_metadata(frames_dataset) -> Dict[str, List]:
#     """Extract time-series data from JSON metadata stored in frame attributes."""
#     import json
#     timeseries_data = {}

#     # Look for metadata attributes
#     metadata_attrs = []
#     for attr_name in frames_dataset.attrs.keys():
#         if 'metadata' in attr_name.lower():
#             metadata_attrs.append(attr_name)

#     if not metadata_attrs:
#         return {}

#     # Try different attribute naming patterns
#     for i in range(min(1000, len(frames_dataset))):  # Limit for performance
#         found_metadata = False

#         # Try various naming patterns for frame metadata
#         attr_patterns = [
#             f'frame_{i}_metadata',
#             f'metadata_{i}',
#             f'frame_metadata_{i}',
#             'metadata' if i == 0 else None  # Single metadata for first frame
#         ]

#         for pattern in attr_patterns:
#             if pattern and pattern in frames_dataset.attrs:
#                 try:
#                     metadata_str = frames_dataset.attrs[pattern]
#                     if isinstance(metadata_str, bytes):
#                         metadata_str = metadata_str.decode('utf-8')

#                     frame_data = json.loads(metadata_str)

#                     # Extract time-series parameters from nested structure
#                     extracted_values = _extract_numeric_values_from_dict(frame_data)

#                     # Add to time-series arrays
#                     for param_name, value in extracted_values.items():
#                         if param_name not in timeseries_data:
#                             timeseries_data[param_name] = []
#                         timeseries_data[param_name].append(value)

#                     found_metadata = True
#                     break

#                 except (json.JSONDecodeError, ValueError, KeyError):
#                     continue

#         if not found_metadata and i > 10:
#             # If we haven't found metadata for 10+ frames, probably wrong naming pattern
#             break

#     return timeseries_data

# def _extract_numeric_values_from_dict(data: dict, prefix: str = "") -> Dict[str, float]:
#     """Recursively extract numeric values from nested dictionary."""
#     extracted = {}

#     for key, value in data.items():
#         # Create clean parameter name
#         if prefix:
#             if prefix in ['esp32_timing', 'python_timing', 'frame_metadata']:
#                 param_name = key  # Skip common prefixes for cleaner names
#             else:
#                 param_name = f"{prefix}_{key}"
#         else:
#             param_name = key

#         if isinstance(value, dict):
#             # Recursively extract from nested dictionaries
#             nested_values = _extract_numeric_values_from_dict(value, key)
#             extracted.update(nested_values)
#         elif isinstance(value, (int, float)) and not isinstance(value, bool):
#             # This is a numeric time-series value
#             extracted[param_name] = value

#     return extracted

# def create_hdf5_metadata_timeseries_dataframe(hdf5_metadata: dict, frame_interval: float):
#     """
#     Create DataFrame specifically for HDF5 metadata time-series.
#     This keeps HDF5 metadata separate from analysis results.

#     Args:
#         hdf5_metadata: Dictionary of HDF5 time-series metadata
#         frame_interval: Time interval between frames in seconds

#     Returns:
#         pandas.DataFrame with time-series metadata
#     """
#     if pd is None:
#         raise ImportError("pandas is required for DataFrame creation")

#     if not hdf5_metadata:
#         return pd.DataFrame({'Time (min)': [], 'Message': ['No HDF5 time-series metadata found']})

#     # Find the maximum length
#     max_length = max(len(data) if hasattr(data, '__len__') else 1 for data in hdf5_metadata.values())

#     # Create time column aligned with analysis frame interval
#     time_minutes = [(i * frame_interval) / 60.0 for i in range(max_length)]

#     # Build DataFrame
#     df_data = {'Time (min)': time_minutes}

#     for param_name, param_data in hdf5_metadata.items():
#         if hasattr(param_data, '__len__') and len(param_data) > 0:
#             # Pad shorter series with NaN
#             # Pad shorter series with NaN
#             padded_data = list(param_data) + [np.nan] * (max_length - len(param_data))
#             df_data[param_name] = padded_data
#         else:
#             # Single value or empty data
#             df_data[param_name] = [param_data] * max_length if max_length > 0 else []

#     return pd.DataFrame(df_data)

# def create_metadata_dataframe(metadata: dict, source_name: str):
#     """
#     Helper to create DataFrame from static metadata.

#     Args:
#         metadata: Dictionary of metadata
#         source_name: Name of the metadata source

#     Returns:
#         pandas.DataFrame with flattened metadata
#     """
#     if pd is None:
#         raise ImportError("pandas is required for DataFrame creation")

#     rows = []

#     def flatten_metadata(d, prefix=""):
#         for key, value in d.items():
#             full_key = f"{prefix}.{key}" if prefix else key

#             if isinstance(value, dict):
#                 flatten_metadata(value, full_key)
#             else:
#                 try:
#                     str_value = str(value)
#                     data_type = type(value).__name__
#                 except:
#                     str_value = f"<{type(value).__name__}>"
#                     data_type = 'Complex'

#                 rows.append({
#                     'Category': prefix if prefix else 'Root',
#                     'Parameter': key,
#                     'Value': str_value,
#                     'Data_Type': data_type,
#                     'Source': source_name
#                 })

#     flatten_metadata(metadata)
#     return pd.DataFrame(rows)


# def write_metadata_to_csv(writer, metadata: dict, prefix: str):
#     """
#     Helper to write nested metadata to CSV writer.

#     Args:
#         writer: CSV writer object
#         metadata: Dictionary of metadata to write
#         prefix: Prefix for parameter names
#     """
#     def write_nested_dict(d, current_prefix):
#         for key, value in d.items():
#             full_key = f"{current_prefix}.{key}"

#             if isinstance(value, dict):
#                 write_nested_dict(value, full_key)
#             elif isinstance(value, (list, tuple)):
#                 if len(value) <= 5:
#                     writer.writerow([full_key, str(value)])
#                 else:
#                     writer.writerow([full_key, f"[List with {len(value)} items]"])
#             else:
#                 try:
#                     str_value = str(value)
#                     writer.writerow([full_key, str_value])
#                 except:
#                     writer.writerow([full_key, f"<{type(value).__name__}>"])

#     write_nested_dict(metadata, prefix)


# def filter_hdf5_metadata_only(ts_data: dict) -> dict:
#     """
#     Filter to keep only actual HDF5 metadata, excluding analysis results.
#     Be specific about what to exclude to preserve legitimate HDF5 time-series.

#     Args:
#         ts_data: Dictionary of time-series data

#     Returns:
#         Dictionary with only HDF5 metadata (excludes analysis results)
#     """
#     hdf5_metadata_only = {}

#     # Only exclude specific analysis result patterns, not general HDF5 metadata
#     analysis_result_patterns = [
#         'roi_', 'baseline_', 'threshold_', 'upper_threshold', 'lower_threshold',
#         'movement_data', 'fraction_data', 'sleep_data', 'quiescence_data',
#         'intensity_roi_', 'analysis_', 'processed_', 'calculated_'
#     ]

#     for param_name, param_data in ts_data.items():
#         param_lower = param_name.lower()

#         # Keep the parameter unless it matches specific analysis result patterns
#         is_analysis_result = any(pattern in param_lower for pattern in analysis_result_patterns)

#         if not is_analysis_result:
#             hdf5_metadata_only[param_name] = param_data

#     return hdf5_metadata_only


# def clean_sheet_name(param_name: str) -> str:
#     """
#     Clean parameter name to be valid Excel sheet name.
#     Excel sheet names: max 31 chars, no special characters.

#     Args:
#         param_name: Original parameter name

#     Returns:
#         Cleaned sheet name valid for Excel
#     """
#     # Remove or replace invalid characters
#     clean = param_name.replace('/', '_').replace('\\', '_').replace(':', '_')
#     clean = clean.replace('*', 'star').replace('?', 'q').replace('[', '').replace(']', '')
#     clean = clean.replace('<', 'lt').replace('>', 'gt').replace('|', '_')

#     # Truncate to 31 characters max
#     if len(clean) > 31:
#         # Try to keep meaningful parts
#         if '_' in clean:
#             parts = clean.split('_')
#             if len(parts[0]) <= 25:  # Keep first part if reasonable length
#                 clean = parts[0] + '_' + ''.join(p[0] for p in parts[1:] if p)[:5]
#             else:
#                 clean = clean[:31]
#         else:
#             clean = clean[:31]

#     # Remove trailing underscores
#     clean = clean.rstrip('_')

#     # Ensure it's not empty
#     if not clean:
#         clean = 'unnamed_param'

#     return clean


# def create_individual_timeseries_sheets(writer, hdf5_metadata: dict, frame_interval: float) -> list:
#     """
#     Create individual Excel sheets for each time-series parameter.

#     Args:
#         writer: Excel writer object
#         hdf5_metadata: Dictionary of HDF5 time-series metadata
#         frame_interval: Time interval between frames in seconds

#     Returns:
#         List of created sheet names
#     """
#     if pd is None:
#         raise ImportError("pandas is required for Excel sheet creation")

#     sheets_created = []

#     for param_name, param_data in hdf5_metadata.items():
#         try:
#             if not hasattr(param_data, '__len__') or len(param_data) == 0:
#                 continue

#             # Create DataFrame with just this one parameter
#             single_param_data = {param_name: param_data}
#             param_df = create_hdf5_metadata_timeseries_dataframe(
#                 single_param_data, frame_interval
#             )

#             # Clean parameter name for Excel sheet name
#             clean_name = clean_sheet_name(param_name)

#             # Ensure unique sheet name
#             original_clean_name = clean_name
#             counter = 1
#             while clean_name in sheets_created:
#                 clean_name = f"{original_clean_name[:28]}_{counter}"
#                 counter += 1

#             # Create the sheet
#             param_df.to_excel(writer, sheet_name=clean_name, index=False)
#             sheets_created.append(clean_name)

#         except Exception as e:
#             print(f"Failed to create sheet for {param_name}: {e}")
#             continue

#     return sheets_created


# def create_combined_timeseries_sheet(writer, hdf5_metadata: dict, frame_interval: float, sheet_name: str = 'All_HDF5_Data'):
#     """
#     Create combined Excel sheet with all time-series parameters.

#     Args:
#         writer: Excel writer object
#         hdf5_metadata: Dictionary of HDF5 time-series metadata
#         frame_interval: Time interval between frames in seconds
#         sheet_name: Name for the combined sheet
#     """
#     if pd is None:
#         raise ImportError("pandas is required for Excel sheet creation")

#     if not hdf5_metadata:
#         return

#     try:
#         combined_df = create_hdf5_metadata_timeseries_dataframe(
#             hdf5_metadata, frame_interval
#         )
#         combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
#     except Exception as e:
#         print(f"Failed to create combined sheet: {e}")


# def create_metadata_summary(metadata: Dict[str, Any]) -> str:
#     """
#     Create a human-readable summary of HDF5 metadata.

#     Args:
#         metadata: Metadata dictionary

#     Returns:
#         Formatted summary string
#     """
#     summary = []
#     summary.append("=== HDF5 FILE METADATA SUMMARY ===")

#     # File information
#     if 'file_info' in metadata:
#         file_info = metadata['file_info']
#         summary.append(f"\nFile Information:")
#         summary.append(f"  Name: {file_info.get('filename', 'Unknown')}")
#         summary.append(f"  Size: {file_info.get('file_size_mb', 0)} MB")

#     # Technical information
#     if 'technical_info' in metadata:
#         tech_info = metadata['technical_info']
#         summary.append(f"\nTechnical Information:")
#         summary.append(f"  Total Frames: {tech_info.get('total_frames', 'Unknown')}")
#         summary.append(f"  Frame Shape: {tech_info.get('frame_shape', 'Unknown')}")
#         summary.append(f"  Frame Data Type: {tech_info.get('frame_dtype', 'Unknown')}")

#     # Experimental information
#     if 'experimental_info' in metadata and metadata['experimental_info']:
#         exp_info = metadata['experimental_info']
#         summary.append(f"\nExperimental Information:")
#         for key, value in list(exp_info.items())[:5]:
#             summary.append(f"  {key}: {value}")
#         if len(exp_info) > 5:
#             summary.append(f"  ... and {len(exp_info) - 5} more parameters")

#     # Time-series metadata summary
#     if 'timeseries_summary' in metadata:
#         ts_summary = metadata['timeseries_summary']
#         summary.append(f"\nTime-Series Metadata:")
#         summary.append(f"  Parameters Found: {ts_summary.get('total_parameters', 0)}")
#         summary.append(f"  Max Length: {ts_summary.get('max_length', 0)} time points")
#         if ts_summary.get('parameters_found'):
#             param_names = ts_summary['parameters_found'][:5]  # Show first 5
#             summary.append(f"  Examples: {', '.join(param_names)}")
#             if len(ts_summary['parameters_found']) > 5:
#                 summary.append(f"    ... and {len(ts_summary['parameters_found']) - 5} more")

#     summary.append("\n" + "=" * 40)
#     return "\n".join(summary)


# def create_timeseries_metadata_summary(metadata: Dict[str, Any]) -> str:
#     """
#     Create a detailed summary of time-series metadata.

#     Args:
#         metadata: Metadata dictionary with timeseries_data

#     Returns:
#         Formatted time-series summary string
#     """
#     if 'timeseries_data' not in metadata or not metadata['timeseries_data']:
#         return "No time-series metadata found in HDF5 file."

#     summary = []
#     summary.append("=== TIME-SERIES METADATA DETAILED SUMMARY ===")

#     timeseries_data = metadata['timeseries_data']

#     for param_name, param_data in timeseries_data.items():
#         summary.append(f"\n{param_name}:")

#         if isinstance(param_data, (list, np.ndarray)):
#             summary.append(f"  Length: {len(param_data)} time points")

#             # Basic statistics for numeric data
#             try:
#                 data_array = np.array(param_data)
#                 if np.issubdtype(data_array.dtype, np.number):
#                     finite_data = data_array[np.isfinite(data_array)]
#                     if len(finite_data) > 0:
#                         summary.append(f"  Range: {np.min(finite_data):.3f} to {np.max(finite_data):.3f}")
#                         summary.append(f"  Mean: {np.mean(finite_data):.3f}")
#                         if len(finite_data) > 1:
#                             summary.append(f"  Std: {np.std(finite_data):.3f}")
#                 else:
#                     summary.append(f"  Data Type: {data_array.dtype}")
#                     unique_values = len(np.unique(data_array))
#                     summary.append(f"  Unique Values: {unique_values}")

#             except Exception:
#                 summary.append(f"  Data Type: {type(param_data[0]).__name__}")
#         else:
#             summary.append(f"  Single Value: {param_data}")

#     summary.append("\n" + "=" * 50)
#     return "\n".join(summary)


# # ===================================================================
# # WIDGET INTEGRATION HELPERS
# # ===================================================================

# def add_nematostella_analysis_to_widget(widget_instance):
#     """
#     Add Nematostella-specific analysis capabilities to the existing widget.
#     This function can be called from the widget to enable enhanced analysis.

#     Args:
#         widget_instance: Instance of HDF5AnalysisWidget
#     """
#     # Add new button to widget if it doesn't exist
#     if not hasattr(widget_instance, 'btn_nematostella_analysis'):
#         from qtpy.QtWidgets import QPushButton

#         widget_instance.btn_nematostella_analysis = QPushButton("Nematostella Timeseries Analysis")
#         widget_instance.btn_nematostella_analysis.setToolTip("Run specialized Nematostella timeseries analysis")
#         widget_instance.btn_nematostella_analysis.setStyleSheet(
#             "QPushButton { background-color: #9C27B0; color: white; font-weight: bold; }"
#         )

#         # Add to the existing results tab layout
#         if hasattr(widget_instance, 'tab_results'):
#             layout = widget_instance.tab_results.layout()
#             if layout:
#                 # Find the plot buttons group and add after it
#                 for i in range(layout.count()):
#                     item = layout.itemAt(i)
#                     if item and hasattr(item.widget(), 'title') and 'Controls' in item.widget().title():
#                         # Insert after plot controls group
#                         layout.insertWidget(i + 1, widget_instance.btn_nematostella_analysis)
#                         break
#                 else:
#                     # Fallback: add at the end
#                     layout.addWidget(widget_instance.btn_nematostella_analysis)

#         # Connect the button
#         widget_instance.btn_nematostella_analysis.clicked.connect(
#             lambda: run_nematostella_analysis_from_widget(widget_instance)
#         )


# def run_nematostella_analysis_from_widget(widget_instance):
#     """
#     Run Nematostella analysis from within the napari widget.

#     Args:
#         widget_instance: Instance of HDF5AnalysisWidget
#     """
#     if not hasattr(widget_instance, 'file_path') or not widget_instance.file_path:
#         widget_instance._log_message("No HDF5 file loaded for Nematostella analysis")
#         widget_instance.results_label.setText("Error: No HDF5 file loaded")
#         return

#     try:
#         widget_instance._log_message("Starting Nematostella timeseries analysis...")
#         widget_instance.results_label.setText("Running Nematostella analysis...")

#         # Get quick summary first
#         summary = get_nematostella_timeseries_summary(widget_instance.file_path)
#         widget_instance._log_message("Timeseries summary:")
#         for line in summary.split('\n'):
#             if line.strip():
#                 widget_instance._log_message(f"  {line}")

#         # Run full analysis
#         results = analyze_nematostella_hdf5_file(widget_instance.file_path)

#         if results['success']:
#             widget_instance._log_message("Nematostella analysis completed successfully!")
#             widget_instance._log_message(f"Excel file created: {results['excel_file']}")
#             widget_instance._log_message(f"Report file created: {results['report_file']}")
#             widget_instance._log_message(f"Sheets created: {', '.join(results['sheets_created'])}")

#             # Update results display
#             widget_instance.results_label.setText(
#                 f"Nematostella analysis complete: {len(results['sheets_created'])} Excel sheets created"
#             )

#             # Log key findings from report
#             widget_instance._log_message("Key Analysis Results:")
#             report_lines = results['report'].split('\n')
#             in_important_section = False
#             for line in report_lines:
#                 if any(section in line for section in ['## Timing Analysis', '## LED System Analysis', '## Environmental Conditions']):
#                     in_important_section = True
#                     widget_instance._log_message(line)
#                 elif line.startswith('##') and in_important_section:
#                     in_important_section = False
#                 elif in_important_section and line.strip().startswith('-'):
#                     widget_instance._log_message(f"  {line.strip()}")

#         else:
#             widget_instance._log_message(f"Nematostella analysis failed: {results['error']}")
#             widget_instance.results_label.setText(f"Analysis failed: {results['error']}")

#     except Exception as e:
#         error_msg = f"Nematostella analysis error: {e}"
#         widget_instance._log_message(error_msg)
#         widget_instance.results_label.setText(error_msg)


# # ===================================================================
# # LEGACY COMPATIBILITY FUNCTIONS
# # ===================================================================

# def _safe_filter_hdf5_metadata(ts_data: dict) -> dict:
#     """Safe version with fallback when module function not available."""
#     # Use the main function
#     return filter_hdf5_metadata_only(ts_data)


# def debug_metadata_quick(file_path: str):
#     """Quick debug to see what's in the HDF5 file"""
#     if not file_path or not os.path.exists(file_path):
#         print("No valid file path provided")
#         return

#     try:
#         with h5py.File(file_path, 'r') as f:
#             print(f"HDF5 groups: {list(f.keys())}")

#             if 'frames' in f:
#                 frames = f['frames']
#                 print(f"Frame attributes: {list(frames.attrs.keys())}")

#                 # Look for metadata in first few attributes
#                 for attr_name in list(frames.attrs.keys())[:5]:
#                     if 'metadata' in attr_name.lower():
#                         print(f"Found metadata attribute: {attr_name}")
#                         attr_value = frames.attrs[attr_name]
#                         print(f"Type: {type(attr_value)}, Preview: {str(attr_value)[:200]}")

#             if 'timeseries' in f:
#                 ts_group = f['timeseries']
#                 print(f"Timeseries datasets: {list(ts_group.keys())}")
#                 for key in list(ts_group.keys())[:5]:
#                     dataset = ts_group[key]
#                     print(f"  {key}: shape {dataset.shape}, dtype {dataset.dtype}")

#     except Exception as e:
#         print(f"Debug failed: {e}")
"""
_metadata.py - Enhanced HDF5 Metadata extraction module with Nematostella Timeseries Analysis

This module handles extraction and formatting of HDF5 metadata for the analysis plugin.
Enhanced with specialized Nematostella timeseries analysis capabilities and comprehensive Excel export.
"""

import h5py
import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import csv

try:
    import pandas as pd
except ImportError:
    pd = None

# ===================================================================
# EXISTING METADATA FUNCTIONS (PRESERVED)
# ===================================================================


def extract_hdf5_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from HDF5 file.

    Args:
        file_path: Path to HDF5 file

    Returns:
        Dictionary containing all metadata
    """
    metadata = {
        "file_info": {},
        "datasets": {},
        "groups": {},
        "attributes": {},
        "experimental_info": {},
        "technical_info": {},
        "extraction_info": {
            "extracted_at": datetime.now().isoformat(),
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
        },
    }

    if not os.path.exists(file_path):
        metadata["extraction_info"]["error"] = f"File not found: {file_path}"
        return metadata

    try:
        with h5py.File(file_path, "r") as f:
            # File-level information
            metadata["file_info"] = {
                "filename": os.path.basename(file_path),
                "file_size_bytes": os.path.getsize(file_path),
                "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
                "hdf5_version": f.libver,
                "driver": f.driver if hasattr(f, "driver") else "default",
            }

            # Extract file-level attributes
            metadata["attributes"]["file_level"] = dict(f.attrs)

            # Recursively extract all groups and datasets
            def extract_item_metadata(name, obj):
                item_info = {
                    "name": name,
                    "type": type(obj).__name__,
                    "attributes": dict(obj.attrs),
                }

                if isinstance(obj, h5py.Dataset):
                    item_info.update(
                        {
                            "shape": obj.shape,
                            "dtype": str(obj.dtype),
                            "size": obj.size,
                            "ndim": obj.ndim,
                            "chunks": obj.chunks,
                            "compression": obj.compression,
                            "compression_opts": obj.compression_opts,
                            "shuffle": obj.shuffle,
                            "fletcher32": obj.fletcher32,
                            "fillvalue": obj.fillvalue,
                        }
                    )
                    metadata["datasets"][name] = item_info

                elif isinstance(obj, h5py.Group):
                    item_info.update({"keys": list(obj.keys()), "length": len(obj)})
                    metadata["groups"][name] = item_info

            # Visit all items in the file
            f.visititems(extract_item_metadata)

            # Extract experimental metadata
            experimental_attrs = {}
            for key, value in f.attrs.items():
                key_lower = key.lower()
                if any(
                    exp_key in key_lower
                    for exp_key in [
                        "experiment",
                        "date",
                        "time",
                        "duration",
                        "interval",
                        "frame",
                        "fps",
                        "resolution",
                        "microscope",
                        "objective",
                        "magnification",
                        "exposure",
                        "gain",
                        "temperature",
                        "humidity",
                        "light",
                        "treatment",
                        "condition",
                        "protocol",
                        "researcher",
                        "lab",
                        "species",
                        "strain",
                        "age",
                        "sex",
                    ]
                ):
                    experimental_attrs[key] = value

            metadata["experimental_info"] = experimental_attrs

            # Technical information
            if "frames" in f:
                frames_dataset = f["frames"]
                metadata["technical_info"] = {
                    "total_frames": len(frames_dataset),
                    "frame_shape": (
                        frames_dataset.shape[1:]
                        if len(frames_dataset.shape) > 1
                        else ()
                    ),
                    "frame_dtype": str(frames_dataset.dtype),
                }

                # Try to extract timing information
                if "timestamps" in f:
                    timestamps = f["timestamps"][:]
                    if len(timestamps) > 1:
                        intervals = np.diff(timestamps[:10])
                        metadata["technical_info"]["detected_frame_intervals"] = {
                            "mean_interval": float(np.mean(intervals)),
                            "std_interval": float(np.std(intervals)),
                        }

    except Exception as e:
        metadata["extraction_info"]["error"] = f"Error extracting metadata: {str(e)}"

    return metadata


def detect_hdf5_structure_type(file_path: str) -> str:
    """
    Detect the type of HDF5 structure.

    Returns:
        'individual_frames': Individual frames in images/ group
        'stacked_frames': All frames in single 'frames' dataset
        'unknown': Cannot determine structure
    """
    try:
        with h5py.File(file_path, "r") as h5_file:
            if "images" in h5_file and "timeseries" in h5_file:
                images_group = h5_file["images"]
                if len(images_group.keys()) > 100:  # Many individual frames
                    return "individual_frames"
            elif "frames" in h5_file:
                return "stacked_frames"
            else:
                return "unknown"
    except:
        return "unknown"


def extract_hdf5_metadata_timeseries(file_path: str) -> Dict[str, Any]:
    """Extract time-series data from JSON metadata attributes and dataset groups."""
    metadata = extract_hdf5_metadata(file_path)

    if not os.path.exists(file_path):
        metadata["timeseries_data"] = {}
        return metadata

    try:
        with h5py.File(file_path, "r") as f:
            timeseries_metadata = {}

            # Method 1: Extract from JSON frame metadata (primary method)
            if "frames" in f:
                frames_dataset = f["frames"]
                json_data = _extract_from_json_metadata(frames_dataset)
                timeseries_metadata.update(json_data)

            # Method 2: Extract from separate timeseries datasets (fallback)
            if "timeseries" in f:
                timeseries_group = f["timeseries"]
                for dataset_name in timeseries_group.keys():
                    try:
                        dataset = timeseries_group[dataset_name]

                        # Handle scalar datasets
                        if dataset.shape == ():
                            data = dataset[()]
                            timeseries_metadata[dataset_name] = [data]
                        else:
                            data = dataset[:]
                            if len(data) > 0:
                                # Only add if not already found in JSON metadata
                                if dataset_name not in timeseries_metadata:
                                    timeseries_metadata[dataset_name] = data.tolist()
                    except Exception as e:
                        metadata["extraction_info"][
                            f"timeseries_error_{dataset_name}"
                        ] = str(e)

            # Method 3: Check root level datasets
            for dataset_name in f.keys():
                if dataset_name not in ["frames", "images", "metadata", "timeseries"]:
                    try:
                        dataset = f[dataset_name]
                        if dataset.shape == ():
                            data = dataset[()]
                            if dataset_name not in timeseries_metadata:
                                timeseries_metadata[dataset_name] = [data]
                        else:
                            data = dataset[:]
                            if (
                                len(data) > 0
                                and dataset_name not in timeseries_metadata
                            ):
                                timeseries_metadata[dataset_name] = data.tolist()
                    except Exception as e:
                        metadata["extraction_info"][f"root_error_{dataset_name}"] = str(
                            e
                        )

            # Summary
            metadata["timeseries_data"] = timeseries_metadata
            metadata["timeseries_summary"] = {
                "total_parameters": len(timeseries_metadata),
                "parameters_found": list(timeseries_metadata.keys()),
                "max_length": (
                    max(
                        len(data) if hasattr(data, "__len__") else 1
                        for data in timeseries_metadata.values()
                    )
                    if timeseries_metadata
                    else 0
                ),
                "data_types": {
                    param: (
                        str(type(data[0]).__name__)
                        if data and hasattr(data, "__len__") and len(data) > 0
                        else "empty"
                    )
                    for param, data in timeseries_metadata.items()
                },
            }

    except Exception as e:
        metadata["extraction_info"]["timeseries_extraction_error"] = str(e)
        metadata["timeseries_data"] = {}

    return metadata


# ===================================================================
# ENHANCED NEMATOSTELLA TIMESERIES ANALYSIS (NEW)
# ===================================================================


class NematostellaTimeseriesAnalyzer:
    """
    Enhanced timeseries analyzer specifically for Nematostella experiments.
    Integrates with existing napari plugin architecture.
    """

    def __init__(self, file_path: str):
        """Initialize analyzer with HDF5 file path."""
        self.file_path = file_path
        self.timeseries_data = {}
        self.metadata = {}
        self.analysis_results = {}

    def extract_and_analyze_timeseries(self) -> pd.DataFrame:
        """
        Extract and analyze all timeseries data from HDF5 file.
        Returns comprehensive DataFrame with all timeseries.
        """
        if pd is None:
            raise ImportError("pandas is required for timeseries analysis")

        timeseries_dict = {}

        with h5py.File(self.file_path, "r") as h5_file:
            # Extract root metadata
            self.metadata = dict(h5_file.attrs)

            # Extract timeseries group metadata if exists
            if "timeseries" in h5_file:
                ts_group = h5_file["timeseries"]
                self.timeseries_metadata = dict(ts_group.attrs)

                # Extract all timeseries datasets
                for key in ts_group.keys():
                    dataset = ts_group[key]

                    try:
                        if dataset.shape == ():
                            # Scalar dataset
                            data = dataset[()]
                            timeseries_dict[key] = [
                                data
                            ]  # Make it a list for DataFrame
                        else:
                            # Array dataset
                            data = dataset[:]
                            timeseries_dict[key] = data

                        # Store dataset attributes
                        if dataset.attrs:
                            attrs = dict(dataset.attrs)
                            print(f"Nematostella analyzer: {key} attributes: {attrs}")

                    except Exception as e:
                        print(
                            f"Warning: Could not read timeseries dataset '{key}': {e}"
                        )

        # Create DataFrame
        df = pd.DataFrame(timeseries_dict)

        # Convert timestamps to datetime if they exist
        if "capture_timestamps" in df.columns:
            df["capture_datetime"] = pd.to_datetime(df["capture_timestamps"], unit="s")
        if "expected_timestamps" in df.columns:
            df["expected_datetime"] = pd.to_datetime(
                df["expected_timestamps"], unit="s"
            )

        return df

    def analyze_timing_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze timing-related data from the timeseries."""
        analysis = {}

        if "actual_intervals" in df.columns and "expected_intervals" in df.columns:
            analysis["timing"] = {
                "mean_actual_interval": df["actual_intervals"].mean(),
                "mean_expected_interval": df["expected_intervals"].mean(),
                "interval_std": df["actual_intervals"].std(),
                "timing_accuracy": 1
                - (df["actual_intervals"] - df["expected_intervals"]).abs().mean()
                / df["expected_intervals"].mean(),
            }

        if "frame_drift" in df.columns:
            analysis["drift"] = {
                "max_drift": df["frame_drift"].max(),
                "min_drift": df["frame_drift"].min(),
                "mean_drift": df["frame_drift"].mean(),
                "drift_trend": np.polyfit(range(len(df)), df["frame_drift"], 1)[
                    0
                ],  # Linear trend
            }

        if "cumulative_drift" in df.columns:
            analysis["cumulative_drift"] = {
                "final_drift": df["cumulative_drift"].iloc[-1],
                "max_cumulative_drift": df["cumulative_drift"].max(),
                "min_cumulative_drift": df["cumulative_drift"].min(),
            }

        return analysis

    def analyze_image_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze image statistics from the timeseries."""
        analysis = {}

        image_stats = ["frame_mean", "frame_std", "frame_min", "frame_max"]
        available_stats = [stat for stat in image_stats if stat in df.columns]

        if available_stats:
            analysis["image_statistics"] = {}
            for stat in available_stats:
                analysis["image_statistics"][stat] = {
                    "mean": df[stat].mean(),
                    "std": df[stat].std(),
                    "min": df[stat].min(),
                    "max": df[stat].max(),
                    "trend": np.polyfit(range(len(df)), df[stat], 1)[0],
                }

        return analysis

    def analyze_environmental_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze environmental data from the timeseries."""
        analysis = {}

        env_vars = ["temperature", "humidity"]
        available_vars = [var for var in env_vars if var in df.columns]

        if available_vars:
            analysis["environment"] = {}
            for var in available_vars:
                analysis["environment"][var] = {
                    "mean": df[var].mean(),
                    "std": df[var].std(),
                    "min": df[var].min(),
                    "max": df[var].max(),
                    "range": df[var].max() - df[var].min(),
                }

        return analysis

    def analyze_led_system(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze LED-related data from the timeseries."""
        analysis = {}

        if "led_power_percent" in df.columns:
            analysis["led_power"] = {
                "mean": df["led_power_percent"].mean(),
                "std": df["led_power_percent"].std(),
                "min": df["led_power_percent"].min(),
                "max": df["led_power_percent"].max(),
            }

        if "led_duration_ms" in df.columns:
            analysis["led_duration"] = {
                "mean": df["led_duration_ms"].mean(),
                "std": df["led_duration_ms"].std(),
                "min": df["led_duration_ms"].min(),
                "max": df["led_duration_ms"].max(),
            }

        if "led_sync_success" in df.columns:
            analysis["led_sync"] = {
                "success_rate": df["led_sync_success"].mean(),
                "total_frames": len(df),
                "successful_syncs": df["led_sync_success"].sum(),
                "failed_syncs": (~df["led_sync_success"]).sum(),
            }

        return analysis

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive analysis report for Nematostella experiment."""
        print("Nematostella Analyzer: Extracting timeseries data...")
        df = self.extract_and_analyze_timeseries()

        print("Nematostella Analyzer: Running specialized analysis...")
        timing_analysis = self.analyze_timing_performance(df)
        image_analysis = self.analyze_image_statistics(df)
        env_analysis = self.analyze_environmental_conditions(df)
        led_analysis = self.analyze_led_system(df)

        # Store analysis results for export
        self.analysis_results = {
            "dataframe": df,
            "timing_analysis": timing_analysis,
            "image_analysis": image_analysis,
            "env_analysis": env_analysis,
            "led_analysis": led_analysis,
        }

        # Generate report text
        report = f"""
# NEMATOSTELLA TIMELAPSE ANALYSIS REPORT
{'='*50}

## File Information
- File: {self.file_path}
- Total Frames: {len(df)}
- Data Columns: {len(df.columns)}

## Root Metadata
"""

        for key, value in self.metadata.items():
            report += f"- {key}: {value}\n"

        if hasattr(self, "timeseries_metadata"):
            report += "\n## Timeseries Metadata\n"
            for key, value in self.timeseries_metadata.items():
                report += f"- {key}: {value}\n"

        # Timing Analysis
        if timing_analysis:
            report += "\n## Timing Analysis\n"
            if "timing" in timing_analysis:
                timing = timing_analysis["timing"]
                report += f"- Mean Actual Interval: {timing['mean_actual_interval']:.4f} seconds\n"
                report += f"- Mean Expected Interval: {timing['mean_expected_interval']:.4f} seconds\n"
                report += f"- Interval Standard Deviation: {timing['interval_std']:.4f} seconds\n"
                report += f"- Timing Accuracy: {timing['timing_accuracy']:.2%}\n"

            if "drift" in timing_analysis:
                drift = timing_analysis["drift"]
                report += f"- Mean Drift: {drift['mean_drift']:.4f} seconds\n"
                report += f"- Max Drift: {drift['max_drift']:.4f} seconds\n"
                report += f"- Drift Trend: {drift['drift_trend']:.6f} seconds/frame\n"

        # Image Analysis
        if image_analysis and "image_statistics" in image_analysis:
            report += "\n## Image Statistics Analysis\n"
            for stat, values in image_analysis["image_statistics"].items():
                report += f"### {stat.replace('_', ' ').title()}\n"
                report += f"- Mean: {values['mean']:.2f}\n"
                report += f"- Std: {values['std']:.2f}\n"
                report += f"- Range: {values['min']:.2f} - {values['max']:.2f}\n"
                report += f"- Trend: {values['trend']:.6f} units/frame\n\n"

        # Environmental Analysis
        if env_analysis and "environment" in env_analysis:
            report += "## Environmental Conditions\n"
            for var, values in env_analysis["environment"].items():
                report += f"### {var.title()}\n"
                report += f"- Mean: {values['mean']:.2f}\n"
                report += f"- Range: {values['min']:.2f} - {values['max']:.2f}\n"
                report += f"- Variability: {values['std']:.2f}\n\n"

        # LED Analysis
        if led_analysis:
            report += "## LED System Analysis\n"
            if "led_sync" in led_analysis:
                sync = led_analysis["led_sync"]
                report += f"- Sync Success Rate: {sync['success_rate']:.2%}\n"
                report += f"- Successful Syncs: {sync['successful_syncs']}/{sync['total_frames']}\n"

            if "led_power" in led_analysis:
                power = led_analysis["led_power"]
                report += f"- Mean LED Power: {power['mean']:.1f}%\n"
                report += (
                    f"- LED Power Range: {power['min']:.1f}% - {power['max']:.1f}%\n"
                )

        return report

    def export_to_excel_enhanced(
        self, filename: str = "nematostella_analysis.xlsx"
    ) -> List[str]:
        """
        Export comprehensive analysis to Excel with multiple sheets.
        Returns list of created sheet names.
        """
        if pd is None:
            raise ImportError("pandas and openpyxl are required for Excel export")

        if not hasattr(self, "analysis_results") or not self.analysis_results:
            # Run analysis first
            self.generate_comprehensive_report()

        df = self.analysis_results["dataframe"]
        created_sheets = []

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            # Sheet 1: Complete timeseries data
            df.to_excel(writer, sheet_name="All_Timeseries_Data", index=False)
            created_sheets.append("All_Timeseries_Data")

            # Sheet 2-N: Individual timeseries parameters
            for column in df.columns:
                if "datetime" in column.lower():
                    continue  # Skip datetime derivatives

                # Create DataFrame for this parameter
                param_df = pd.DataFrame({"Index": range(len(df)), column: df[column]})

                # Add timestamp columns if they exist
                if "capture_timestamps" in df.columns:
                    param_df["capture_timestamps"] = df["capture_timestamps"]
                if "capture_datetime" in df.columns:
                    param_df["capture_datetime"] = df["capture_datetime"]

                # Clean sheet name for Excel
                sheet_name = clean_sheet_name(column)

                try:
                    param_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    created_sheets.append(sheet_name)
                except Exception as e:
                    print(f"Warning: Could not create sheet for {column}: {e}")

            # Summary sheet with analysis results
            if self.analysis_results:
                summary_data = []

                # Add timing metrics
                if (
                    "timing_analysis" in self.analysis_results
                    and self.analysis_results["timing_analysis"]
                ):
                    timing = self.analysis_results["timing_analysis"]
                    for category, metrics in timing.items():
                        if isinstance(metrics, dict):
                            for metric, value in metrics.items():
                                summary_data.append(
                                    {
                                        "Category": f"Timing - {category}",
                                        "Metric": metric,
                                        "Value": value,
                                    }
                                )

                # Add environmental metrics
                if (
                    "env_analysis" in self.analysis_results
                    and self.analysis_results["env_analysis"]
                ):
                    env = self.analysis_results["env_analysis"]
                    for category, metrics in env.items():
                        if isinstance(metrics, dict):
                            for var, values in metrics.items():
                                for metric, value in values.items():
                                    summary_data.append(
                                        {
                                            "Category": f"Environment - {var}",
                                            "Metric": metric,
                                            "Value": value,
                                        }
                                    )

                # Add LED metrics
                if (
                    "led_analysis" in self.analysis_results
                    and self.analysis_results["led_analysis"]
                ):
                    led = self.analysis_results["led_analysis"]
                    for category, metrics in led.items():
                        if isinstance(metrics, dict):
                            for metric, value in metrics.items():
                                summary_data.append(
                                    {
                                        "Category": f"LED - {category}",
                                        "Metric": metric,
                                        "Value": value,
                                    }
                                )

                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(
                        writer, sheet_name="Analysis_Summary", index=False
                    )
                    created_sheets.append("Analysis_Summary")

        print(
            f"Nematostella Analyzer: Created Excel file with {len(created_sheets)} sheets"
        )
        return created_sheets


# ===================================================================
# INTEGRATION FUNCTIONS FOR NAPARI WIDGET
# ===================================================================


def analyze_nematostella_hdf5_file(file_path: str) -> Dict[str, Any]:
    """
    Integration function for napari widget to analyze Nematostella HDF5 files.

    Args:
        file_path: Path to HDF5 file

    Returns:
        Dictionary containing analysis results and export paths
    """
    try:
        # Initialize analyzer
        analyzer = NematostellaTimeseriesAnalyzer(file_path)

        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report()

        # Export to Excel
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        excel_filename = f"nematostella_analysis_{base_filename}.xlsx"
        created_sheets = analyzer.export_to_excel_enhanced(excel_filename)

        # Save text report
        report_filename = f"nematostella_analysis_{base_filename}.txt"
        with open(report_filename, "w") as f:
            f.write(report)

        return {
            "success": True,
            "report": report,
            "excel_file": excel_filename,
            "report_file": report_filename,
            "sheets_created": created_sheets,
            "timeseries_data": analyzer.analysis_results.get("dataframe"),
            "analysis_results": analyzer.analysis_results,
        }

    except Exception as e:
        return {"success": False, "error": str(e), "report": f"Analysis failed: {e}"}


def get_nematostella_timeseries_summary(file_path: str) -> str:
    """
    Quick summary function for napari widget to show what timeseries data is available.

    Args:
        file_path: Path to HDF5 file

    Returns:
        String summary of available timeseries data
    """
    try:
        with h5py.File(file_path, "r") as h5_file:
            if "timeseries" not in h5_file:
                return "No 'timeseries' group found in HDF5 file."

            ts_group = h5_file["timeseries"]
            summary = f"Found {len(ts_group.keys())} timeseries datasets:\n"

            for key in ts_group.keys():
                dataset = ts_group[key]
                summary += f"- {key}: shape {dataset.shape}, dtype {dataset.dtype}\n"

                if dataset.attrs:
                    attrs = dict(dataset.attrs)
                    if "description" in attrs:
                        summary += f"  Description: {attrs['description']}\n"
                    if "units" in attrs:
                        summary += f"  Units: {attrs['units']}\n"

            return summary

    except Exception as e:
        return f"Error reading timeseries summary: {e}"


# ===================================================================
# EXISTING HELPER FUNCTIONS (PRESERVED AND EXTENDED)
# ===================================================================


def _extract_from_json_metadata(frames_dataset) -> Dict[str, List]:
    """Extract time-series data from JSON metadata stored in frame attributes."""
    import json

    timeseries_data = {}

    # Look for metadata attributes
    metadata_attrs = []
    for attr_name in frames_dataset.attrs.keys():
        if "metadata" in attr_name.lower():
            metadata_attrs.append(attr_name)

    if not metadata_attrs:
        return {}

    # Try different attribute naming patterns
    for i in range(min(1000, len(frames_dataset))):  # Limit for performance
        found_metadata = False

        # Try various naming patterns for frame metadata
        attr_patterns = [
            f"frame_{i}_metadata",
            f"metadata_{i}",
            f"frame_metadata_{i}",
            "metadata" if i == 0 else None,  # Single metadata for first frame
        ]

        for pattern in attr_patterns:
            if pattern and pattern in frames_dataset.attrs:
                try:
                    metadata_str = frames_dataset.attrs[pattern]
                    if isinstance(metadata_str, bytes):
                        metadata_str = metadata_str.decode("utf-8")

                    frame_data = json.loads(metadata_str)

                    # Extract time-series parameters from nested structure
                    extracted_values = _extract_numeric_values_from_dict(frame_data)

                    # Add to time-series arrays
                    for param_name, value in extracted_values.items():
                        if param_name not in timeseries_data:
                            timeseries_data[param_name] = []
                        timeseries_data[param_name].append(value)

                    found_metadata = True
                    break

                except (json.JSONDecodeError, ValueError, KeyError):
                    continue

        if not found_metadata and i > 10:
            # If we haven't found metadata for 10+ frames, probably wrong naming pattern
            break

    return timeseries_data


def _extract_numeric_values_from_dict(data: dict, prefix: str = "") -> Dict[str, float]:
    """Recursively extract numeric values from nested dictionary."""
    extracted = {}

    for key, value in data.items():
        # Create clean parameter name
        if prefix:
            if prefix in ["esp32_timing", "python_timing", "frame_metadata"]:
                param_name = key  # Skip common prefixes for cleaner names
            else:
                param_name = f"{prefix}_{key}"
        else:
            param_name = key

        if isinstance(value, dict):
            # Recursively extract from nested dictionaries
            nested_values = _extract_numeric_values_from_dict(value, key)
            extracted.update(nested_values)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            # This is a numeric time-series value
            extracted[param_name] = value

    return extracted


def create_hdf5_metadata_timeseries_dataframe(
    hdf5_metadata: dict, frame_interval: float
):
    """
    Create DataFrame specifically for HDF5 metadata time-series.
    This keeps HDF5 metadata separate from analysis results.

    Args:
        hdf5_metadata: Dictionary of HDF5 time-series metadata
        frame_interval: Time interval between frames in seconds

    Returns:
        pandas.DataFrame with time-series metadata
    """
    if pd is None:
        raise ImportError("pandas is required for DataFrame creation")

    if not hdf5_metadata:
        return pd.DataFrame(
            {"Time (min)": [], "Message": ["No HDF5 time-series metadata found"]}
        )

    # Find the maximum length
    max_length = max(
        len(data) if hasattr(data, "__len__") else 1 for data in hdf5_metadata.values()
    )

    # Create time column aligned with analysis frame interval
    time_minutes = [(i * frame_interval) / 60.0 for i in range(max_length)]

    # Build DataFrame
    df_data = {"Time (min)": time_minutes}

    for param_name, param_data in hdf5_metadata.items():
        if hasattr(param_data, "__len__") and len(param_data) > 0:
            # Pad shorter series with NaN
            # Pad shorter series with NaN
            padded_data = list(param_data) + [np.nan] * (max_length - len(param_data))
            df_data[param_name] = padded_data
        else:
            # Single value or empty data
            df_data[param_name] = [param_data] * max_length if max_length > 0 else []

    return pd.DataFrame(df_data)


def create_metadata_dataframe(metadata: dict, source_name: str):
    """
    Helper to create DataFrame from static metadata.

    Args:
        metadata: Dictionary of metadata
        source_name: Name of the metadata source

    Returns:
        pandas.DataFrame with flattened metadata
    """
    if pd is None:
        raise ImportError("pandas is required for DataFrame creation")

    rows = []

    def flatten_metadata(d, prefix=""):
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                flatten_metadata(value, full_key)
            else:
                try:
                    str_value = str(value)
                    data_type = type(value).__name__
                except:
                    str_value = f"<{type(value).__name__}>"
                    data_type = "Complex"

                rows.append(
                    {
                        "Category": prefix if prefix else "Root",
                        "Parameter": key,
                        "Value": str_value,
                        "Data_Type": data_type,
                        "Source": source_name,
                    }
                )

    flatten_metadata(metadata)
    return pd.DataFrame(rows)


def write_metadata_to_csv(writer, metadata: dict, prefix: str):
    """
    Helper to write nested metadata to CSV writer.

    Args:
        writer: CSV writer object
        metadata: Dictionary of metadata to write
        prefix: Prefix for parameter names
    """

    def write_nested_dict(d, current_prefix):
        for key, value in d.items():
            full_key = f"{current_prefix}.{key}"

            if isinstance(value, dict):
                write_nested_dict(value, full_key)
            elif isinstance(value, (list, tuple)):
                if len(value) <= 5:
                    writer.writerow([full_key, str(value)])
                else:
                    writer.writerow([full_key, f"[List with {len(value)} items]"])
            else:
                try:
                    str_value = str(value)
                    writer.writerow([full_key, str_value])
                except:
                    writer.writerow([full_key, f"<{type(value).__name__}>"])

    write_nested_dict(metadata, prefix)


def filter_hdf5_metadata_only(ts_data: dict) -> dict:
    """
    Filter to keep only actual HDF5 metadata, excluding analysis results.
    Be specific about what to exclude to preserve legitimate HDF5 time-series.

    Args:
        ts_data: Dictionary of time-series data

    Returns:
        Dictionary with only HDF5 metadata (excludes analysis results)
    """
    hdf5_metadata_only = {}

    # Only exclude specific analysis result patterns, not general HDF5 metadata
    analysis_result_patterns = [
        "roi_",
        "baseline_",
        "threshold_",
        "upper_threshold",
        "lower_threshold",
        "movement_data",
        "fraction_data",
        "sleep_data",
        "quiescence_data",
        "intensity_roi_",
        "analysis_",
        "processed_",
        "calculated_",
    ]

    for param_name, param_data in ts_data.items():
        param_lower = param_name.lower()

        # Keep the parameter unless it matches specific analysis result patterns
        is_analysis_result = any(
            pattern in param_lower for pattern in analysis_result_patterns
        )

        if not is_analysis_result:
            hdf5_metadata_only[param_name] = param_data

    return hdf5_metadata_only


def clean_sheet_name(param_name: str) -> str:
    """
    Clean parameter name to be valid Excel sheet name.
    Excel sheet names: max 31 chars, no special characters.

    Args:
        param_name: Original parameter name

    Returns:
        Cleaned sheet name valid for Excel
    """
    # Remove or replace invalid characters
    clean = param_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    clean = (
        clean.replace("*", "star").replace("?", "q").replace("[", "").replace("]", "")
    )
    clean = clean.replace("<", "lt").replace(">", "gt").replace("|", "_")

    # Truncate to 31 characters max
    if len(clean) > 31:
        # Try to keep meaningful parts
        if "_" in clean:
            parts = clean.split("_")
            if len(parts[0]) <= 25:  # Keep first part if reasonable length
                clean = parts[0] + "_" + "".join(p[0] for p in parts[1:] if p)[:5]
            else:
                clean = clean[:31]
        else:
            clean = clean[:31]

    # Remove trailing underscores
    clean = clean.rstrip("_")

    # Ensure it's not empty
    if not clean:
        clean = "unnamed_param"

    return clean


def create_individual_timeseries_sheets(
    writer, hdf5_metadata: dict, frame_interval: float
) -> list:
    """
    Create individual Excel sheets for each time-series parameter.

    Args:
        writer: Excel writer object
        hdf5_metadata: Dictionary of HDF5 time-series metadata
        frame_interval: Time interval between frames in seconds

    Returns:
        List of created sheet names
    """
    if pd is None:
        raise ImportError("pandas is required for Excel sheet creation")

    sheets_created = []

    for param_name, param_data in hdf5_metadata.items():
        try:
            if not hasattr(param_data, "__len__") or len(param_data) == 0:
                continue

            # Create DataFrame with just this one parameter
            single_param_data = {param_name: param_data}
            param_df = create_hdf5_metadata_timeseries_dataframe(
                single_param_data, frame_interval
            )

            # Clean parameter name for Excel sheet name
            clean_name = clean_sheet_name(param_name)

            # Ensure unique sheet name
            original_clean_name = clean_name
            counter = 1
            while clean_name in sheets_created:
                clean_name = f"{original_clean_name[:28]}_{counter}"
                counter += 1

            # Create the sheet
            param_df.to_excel(writer, sheet_name=clean_name, index=False)
            sheets_created.append(clean_name)

        except Exception as e:
            print(f"Failed to create sheet for {param_name}: {e}")
            continue

    return sheets_created


def create_combined_timeseries_sheet(
    writer,
    hdf5_metadata: dict,
    frame_interval: float,
    sheet_name: str = "All_HDF5_Data",
):
    """
    Create combined Excel sheet with all time-series parameters.

    Args:
        writer: Excel writer object
        hdf5_metadata: Dictionary of HDF5 time-series metadata
        frame_interval: Time interval between frames in seconds
        sheet_name: Name for the combined sheet
    """
    if pd is None:
        raise ImportError("pandas is required for Excel sheet creation")

    if not hdf5_metadata:
        return

    try:
        combined_df = create_hdf5_metadata_timeseries_dataframe(
            hdf5_metadata, frame_interval
        )
        combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        print(f"Failed to create combined sheet: {e}")


def create_metadata_summary(metadata: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of HDF5 metadata.

    Args:
        metadata: Metadata dictionary

    Returns:
        Formatted summary string
    """
    summary = []
    summary.append("=== HDF5 FILE METADATA SUMMARY ===")

    # File information
    if "file_info" in metadata:
        file_info = metadata["file_info"]
        summary.append(f"\nFile Information:")
        summary.append(f"  Name: {file_info.get('filename', 'Unknown')}")
        summary.append(f"  Size: {file_info.get('file_size_mb', 0)} MB")

    # Technical information
    if "technical_info" in metadata:
        tech_info = metadata["technical_info"]
        summary.append(f"\nTechnical Information:")
        summary.append(f"  Total Frames: {tech_info.get('total_frames', 'Unknown')}")
        summary.append(f"  Frame Shape: {tech_info.get('frame_shape', 'Unknown')}")
        summary.append(f"  Frame Data Type: {tech_info.get('frame_dtype', 'Unknown')}")

    # Experimental information
    if "experimental_info" in metadata and metadata["experimental_info"]:
        exp_info = metadata["experimental_info"]
        summary.append(f"\nExperimental Information:")
        for key, value in list(exp_info.items())[:5]:
            summary.append(f"  {key}: {value}")
        if len(exp_info) > 5:
            summary.append(f"  ... and {len(exp_info) - 5} more parameters")

    # Time-series metadata summary
    if "timeseries_summary" in metadata:
        ts_summary = metadata["timeseries_summary"]
        summary.append(f"\nTime-Series Metadata:")
        summary.append(f"  Parameters Found: {ts_summary.get('total_parameters', 0)}")
        summary.append(f"  Max Length: {ts_summary.get('max_length', 0)} time points")
        if ts_summary.get("parameters_found"):
            param_names = ts_summary["parameters_found"][:5]  # Show first 5
            summary.append(f"  Examples: {', '.join(param_names)}")
            if len(ts_summary["parameters_found"]) > 5:
                summary.append(
                    f"    ... and {len(ts_summary['parameters_found']) - 5} more"
                )

    summary.append("\n" + "=" * 40)
    return "\n".join(summary)


def create_timeseries_metadata_summary(metadata: Dict[str, Any]) -> str:
    """
    Create a detailed summary of time-series metadata.

    Args:
        metadata: Metadata dictionary with timeseries_data

    Returns:
        Formatted time-series summary string
    """
    if "timeseries_data" not in metadata or not metadata["timeseries_data"]:
        return "No time-series metadata found in HDF5 file."

    summary = []
    summary.append("=== TIME-SERIES METADATA DETAILED SUMMARY ===")

    timeseries_data = metadata["timeseries_data"]

    for param_name, param_data in timeseries_data.items():
        summary.append(f"\n{param_name}:")

        if isinstance(param_data, (list, np.ndarray)):
            summary.append(f"  Length: {len(param_data)} time points")

            # Basic statistics for numeric data
            try:
                data_array = np.array(param_data)
                if np.issubdtype(data_array.dtype, np.number):
                    finite_data = data_array[np.isfinite(data_array)]
                    if len(finite_data) > 0:
                        summary.append(
                            f"  Range: {np.min(finite_data):.3f} to {np.max(finite_data):.3f}"
                        )
                        summary.append(f"  Mean: {np.mean(finite_data):.3f}")
                        if len(finite_data) > 1:
                            summary.append(f"  Std: {np.std(finite_data):.3f}")
                else:
                    summary.append(f"  Data Type: {data_array.dtype}")
                    unique_values = len(np.unique(data_array))
                    summary.append(f"  Unique Values: {unique_values}")

            except Exception:
                summary.append(f"  Data Type: {type(param_data[0]).__name__}")
        else:
            summary.append(f"  Single Value: {param_data}")

    summary.append("\n" + "=" * 50)
    return "\n".join(summary)


# ===================================================================
# WIDGET INTEGRATION HELPERS
# ===================================================================


def add_nematostella_analysis_to_widget(widget_instance):
    """
    Add Nematostella-specific analysis capabilities to the existing widget.
    This function can be called from the widget to enable enhanced analysis.

    Args:
        widget_instance: Instance of HDF5AnalysisWidget
    """
    # Add new button to widget if it doesn't exist
    if not hasattr(widget_instance, "btn_nematostella_analysis"):
        from qtpy.QtWidgets import QPushButton

        widget_instance.btn_nematostella_analysis = QPushButton(
            "Nematostella Timeseries Analysis"
        )
        widget_instance.btn_nematostella_analysis.setToolTip(
            "Run specialized Nematostella timeseries analysis"
        )
        widget_instance.btn_nematostella_analysis.setStyleSheet(
            "QPushButton { background-color: #9C27B0; color: white; font-weight: bold; }"
        )

        # Add to the existing results tab layout
        if hasattr(widget_instance, "tab_results"):
            layout = widget_instance.tab_results.layout()
            if layout:
                # Find the plot buttons group and add after it
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if (
                        item
                        and hasattr(item.widget(), "title")
                        and "Controls" in item.widget().title()
                    ):
                        # Insert after plot controls group
                        layout.insertWidget(
                            i + 1, widget_instance.btn_nematostella_analysis
                        )
                        break
                else:
                    # Fallback: add at the end
                    layout.addWidget(widget_instance.btn_nematostella_analysis)

        # Connect the button
        widget_instance.btn_nematostella_analysis.clicked.connect(
            lambda: run_nematostella_analysis_from_widget(widget_instance)
        )


def run_nematostella_analysis_from_widget(widget_instance):
    """
    Run Nematostella analysis from within the napari widget.

    Args:
        widget_instance: Instance of HDF5AnalysisWidget
    """
    if not hasattr(widget_instance, "file_path") or not widget_instance.file_path:
        widget_instance._log_message("No HDF5 file loaded for Nematostella analysis")
        widget_instance.results_label.setText("Error: No HDF5 file loaded")
        return

    try:
        widget_instance._log_message("Starting Nematostella timeseries analysis...")
        widget_instance.results_label.setText("Running Nematostella analysis...")

        # Get quick summary first
        summary = get_nematostella_timeseries_summary(widget_instance.file_path)
        widget_instance._log_message("Timeseries summary:")
        for line in summary.split("\n"):
            if line.strip():
                widget_instance._log_message(f"  {line}")

        # Run full analysis
        results = analyze_nematostella_hdf5_file(widget_instance.file_path)

        if results["success"]:
            widget_instance._log_message(
                "Nematostella analysis completed successfully!"
            )
            widget_instance._log_message(f"Excel file created: {results['excel_file']}")
            widget_instance._log_message(
                f"Report file created: {results['report_file']}"
            )
            widget_instance._log_message(
                f"Sheets created: {', '.join(results['sheets_created'])}"
            )

            # Update results display
            widget_instance.results_label.setText(
                f"Nematostella analysis complete: {len(results['sheets_created'])} Excel sheets created"
            )

            # Log key findings from report
            widget_instance._log_message("Key Analysis Results:")
            report_lines = results["report"].split("\n")
            in_important_section = False
            for line in report_lines:
                if any(
                    section in line
                    for section in [
                        "## Timing Analysis",
                        "## LED System Analysis",
                        "## Environmental Conditions",
                    ]
                ):
                    in_important_section = True
                    widget_instance._log_message(line)
                elif line.startswith("##") and in_important_section:
                    in_important_section = False
                elif in_important_section and line.strip().startswith("-"):
                    widget_instance._log_message(f"  {line.strip()}")

        else:
            widget_instance._log_message(
                f"Nematostella analysis failed: {results['error']}"
            )
            widget_instance.results_label.setText(
                f"Analysis failed: {results['error']}"
            )

    except Exception as e:
        error_msg = f"Nematostella analysis error: {e}"
        widget_instance._log_message(error_msg)
        widget_instance.results_label.setText(error_msg)


# ===================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# ===================================================================


def _safe_filter_hdf5_metadata(ts_data: dict) -> dict:
    """Safe version with fallback when module function not available."""
    # Use the main function
    return filter_hdf5_metadata_only(ts_data)


def debug_metadata_quick(file_path: str):
    """Quick debug to see what's in the HDF5 file"""
    if not file_path or not os.path.exists(file_path):
        print("No valid file path provided")
        return

    try:
        with h5py.File(file_path, "r") as f:
            print(f"HDF5 groups: {list(f.keys())}")

            if "frames" in f:
                frames = f["frames"]
                print(f"Frame attributes: {list(frames.attrs.keys())}")

                # Look for metadata in first few attributes
                for attr_name in list(frames.attrs.keys())[:5]:
                    if "metadata" in attr_name.lower():
                        print(f"Found metadata attribute: {attr_name}")
                        attr_value = frames.attrs[attr_name]
                        print(
                            f"Type: {type(attr_value)}, Preview: {str(attr_value)[:200]}"
                        )

            if "timeseries" in f:
                ts_group = f["timeseries"]
                print(f"Timeseries datasets: {list(ts_group.keys())}")
                for key in list(ts_group.keys())[:5]:
                    dataset = ts_group[key]
                    print(f"  {key}: shape {dataset.shape}, dtype {dataset.dtype}")

    except Exception as e:
        print(f"Debug failed: {e}")
