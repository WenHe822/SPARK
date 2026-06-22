'''
--- Global Projection Statistics ---
Number of files processed: 120000
Total data points:       31457280000
Minimum value:           -0.000006
Maximum value:           154.662323
Mean value:              16.994633
Standard deviation:      21.609632
Variance:                466.976194
'''



import os
import numpy as np
import glob
from tqdm import tqdm

def calculate_projection_stats(data_dir):
    """
    Calculates statistics (min, max, mean, std) for all .npy projection files
    within the specified data directory (including train and test subfolders).

    Args:
        data_dir (str): The root directory containing 'train' and 'test' folders.

    Returns:
        dict: A dictionary containing the calculated statistics:
              'min', 'max', 'mean', 'std', 'count', 'num_files'.
              Returns None if no .npy files are found.
    """
    projection_files = []
    for subset in ['train', 'test']:
        subset_dir = os.path.join(data_dir, subset)
        if not os.path.isdir(subset_dir):
            print(f"Warning: Directory not found: {subset_dir}")
            continue
        # Find all patient directories within the subset
        patient_dirs = glob.glob(os.path.join(subset_dir, 'patient_*_cone'))
        for patient_dir in patient_dirs:
            # Find all projection .npy files within each patient directory
            files_in_patient_dir = glob.glob(os.path.join(patient_dir, 'projection*.npy'))
            projection_files.extend(files_in_patient_dir)

    if not projection_files:
        print("Error: No projection .npy files found in the specified directory structure.")
        return None

    print(f"Found {len(projection_files)} projection files.")

    global_min = np.inf
    global_max = -np.inf
    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0

    # Use float64 for accumulators to maintain precision
    dtype_accum = np.float64

    for file_path in tqdm(projection_files, desc="Processing projections"):
        try:
            # Load projection data
            proj_data = np.load(file_path)

            # Ensure data is float for calculations
            proj_data = proj_data.astype(dtype_accum)

            # Update global min and max
            current_min = np.min(proj_data)
            current_max = np.max(proj_data)
            if current_min < global_min:
                global_min = current_min
            if current_max > global_max:
                global_max = current_max

            # Update sums for mean and variance calculation
            total_sum += np.sum(proj_data)
            total_sum_sq += np.sum(proj_data**2)
            total_count += proj_data.size

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue # Skip to the next file if loading fails

    if total_count == 0:
        print("Error: No valid data processed.")
        return None

    # Calculate final mean and standard deviation
    mean = total_sum / total_count
    # Variance = E[X^2] - (E[X])^2
    variance = (total_sum_sq / total_count) - (mean**2)
    # Handle potential floating point inaccuracies causing negative variance
    if variance < 0 and variance > -1e-9: # Allow small negative values close to zero
         variance = 0.0
    elif variance < 0:
        print(f"Warning: Calculated negative variance ({variance}). This might indicate numerical instability.")
        std_dev = np.nan # Cannot compute std dev from negative variance
    else:
        std_dev = np.sqrt(variance)


    stats = {
        'min': global_min,
        'max': global_max,
        'mean': mean,
        'std': std_dev,
        'variance': variance,
        'count': total_count,
        'num_files': len(projection_files)
    }

    return stats

if __name__ == "__main__":
    # Set the path to your dataset directory
    dataset_root = '/Disk_16TB/zhouhaowei/code/network_GAS/TMI/data/TCIA_projections_512res_600numproj'

    print(f"Starting statistics calculation for dataset: {dataset_root}")
    statistics = calculate_projection_stats(dataset_root)

    if statistics:
        print("\n--- Global Projection Statistics ---")
        print(f"Number of files processed: {statistics['num_files']}")
        print(f"Total data points:       {statistics['count']}")
        print(f"Minimum value:           {statistics['min']:.6f}")
        print(f"Maximum value:           {statistics['max']:.6f}")
        print(f"Mean value:              {statistics['mean']:.6f}")
        print(f"Standard deviation:      {statistics['std']:.6f}")
        print(f"Variance:                {statistics['variance']:.6f}")
        print("------------------------------------\n")
    else:
        print("Statistics calculation failed.")
