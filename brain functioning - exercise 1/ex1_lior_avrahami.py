"""
=============================================================================
Brain Functioning Exercise 1 - EEG and ERP Analysis
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')

# Set plotting style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# Task 1: Data Exploration
# =============================================================================

print("\n" + "=" * 80)
print("Task 1: Data Exploration")
print("=" * 80)

# Load the MATLAB data file
data = loadmat('brain functioning - exercise 1/ex1data.mat')

# Extract variables from the loaded data
data_raw = data['data']  # Raw EEG data (shape: channels x timepoints x trials)
time_vec = data['time_vec'][0]  # Time vector in milliseconds
condition = data['all_conds'][0]  # Condition labels for each trial

n_faces = np.sum(condition == 1)
n_faces_texture = np.sum(condition == 3)

# Sampling frequency can be calculated from time vector (assuming equally spaced samples)
time_diff = time_vec[1] - time_vec[0]  
sample_freq = 1000.0 / time_diff  

# Nyquist frequency for this sampling rate
nyquist_freq = sample_freq / 2.0

print("Data loaded successfully")
print(f"Data shape: {data_raw.shape}")
print(f"  - Dimension 1 (Channels): {data_raw.shape[0]}")
print(f"  - Dimension 2 (Time points): {data_raw.shape[1]}")
print(f"  - Dimension 3 (Trials): {data_raw.shape[2]}")
print(f"Number of trials with Faces (condition 1): {n_faces}")
print(f"Number of trials with Faces as Texture (condition 3): {n_faces_texture}")
print(f"Sampling rate: {sample_freq} Hz")
print(f"Nyquist frequency: {nyquist_freq:.1f} Hz")

 # =============================================================================
# Task 2: Data Visualization
# =============================================================================

print("\n" + "=" * 80)
print("Task 2: Data Visualization")
print("=" * 80)

# Plot signals from first 10 trials of channel 7, each in different color
channel_viz = 7  
n_trials_viz = 10

# Create color map for different trials
colors = plt.cm.tab10(np.linspace(0, 1, n_trials_viz))

fig, ax = plt.subplots(figsize=(14, 8))

for trial_idx in range(n_trials_viz):
    signal = data_raw[channel_viz, :, trial_idx]
    ax.plot(time_vec, signal, color=colors[trial_idx], linewidth=1.5, 
            label=f'Trial {trial_idx + 1}', alpha=0.8)

ax.axvline(x=0, color='k', linestyle='--', linewidth=2, alpha=0.7, label='Stimulus onset')
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('Amplitude (μV)', fontsize=12)
ax.set_title(f'First 10 Trials - Channel {channel_viz}', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Task2a_first_10_trials.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'Taks2a_first_10_trials.png'")
plt.close()

# Extract data from face trials (condition 1)
face_trial_indices = np.where(condition == 1)[0]
data_face = data_raw[:, :, face_trial_indices]

print(f"Dimensions of data_face matrix: {data_face.shape}")
print(f"   - Channels: {data_face.shape[0]}")
print(f"   - Time points: {data_face.shape[1]}")
print(f"   - Face trials: {data_face.shape[2]}")

# Plot trials 11-20 from data_face, channel 7
trial_start = 10 # index of trial 11
trial_end = 20 
n_trials_plot = trial_end - trial_start

colors2 = plt.cm.tab10(np.linspace(0, 1, n_trials_plot))

fig, ax = plt.subplots(figsize=(14, 8))

for i, trial_idx in enumerate(range(trial_start, trial_end)):
    signal = data_face[channel_viz, :, trial_idx]
    ax.plot(time_vec, signal, color=colors2[i], linewidth=1.5, 
            label=f'Face Trial {trial_idx + 1}', alpha=0.8)

ax.axvline(x=0, color='k', linestyle='--', linewidth=2, alpha=0.7, label='Stimulus onset')
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_ylabel('Amplitude (μV)', fontsize=12)
ax.set_title(f'Trials 11-20 from Face Data - Channel {channel_viz}', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Task2c_trials_11_20_faces.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'Task2c_trials_11_20_faces.png'")
plt.close()

# =============================================================================
# Task 3: Baseline Normalization 
# =============================================================================

print("\n" + "=" * 80)
print("Task 3: Baseline Normalization")
print("=" * 80)

# Define baseline period: -200 to 0 milliseconds (200ms before stimulus)
baseline_start = -200
baseline_end = 0

# Find indices corresponding to baseline period
baseline_indices = np.where((time_vec >= baseline_start) & (time_vec <= baseline_end))[0]

# Calculate baseline mean for each channel and each trial (average across the time dimension)
baseline_mean = np.mean(data_raw[:, baseline_indices, :], axis=1, keepdims=True)

# Perform baseline normalization by subtracting baseline mean from each trial
data_norm_baseline = data_raw - baseline_mean

# Verify normalization: check that baseline period has mean ≈ 0
baseline_before_norm = np.mean(data_raw[:, baseline_indices, :], axis=1)
baseline_after_norm = np.mean(data_norm_baseline[:, baseline_indices, :], axis=1)
print(f"Mean baseline value before normalization: {np.mean(np.abs(baseline_before_norm)):.2e}")
print(f"Mean baseline value after normalization (should be ≈0): {np.mean(np.abs(baseline_after_norm)):.2e}")

# =============================================================================
# Task 4: Compute ERPs for All Conditions 
# =============================================================================

print("\n" + "=" * 80)
print("Task 4: Compute ERPs for All Conditions")
print("=" * 80)

# Identify unique conditions
conditions_unique = np.unique(condition)
n_conditions = len(conditions_unique)

print(f"\nNumber of conditions: {n_conditions}")
print(f"Condition values: {conditions_unique}")

# Count trials per condition
for cond in conditions_unique:
    n_trials = np.sum(condition == cond)
    print(f"  Condition {int(cond)}: {n_trials} trials")

# Initialize ERPs array: (channels, time_points, conditions)
n_channels = data_norm_baseline.shape[0]
n_timepoints = data_norm_baseline.shape[1]
all_erps = np.zeros((n_channels, n_timepoints, n_conditions))

# Compute ERP for each condition (average across trials)
for idx, cond in enumerate(conditions_unique):
    # Find trials belonging to this condition
    trial_indices = np.where(condition == cond)[0]
    
    # Average across trials for this condition
    all_erps[:, :, idx] = np.mean(data_norm_baseline[:, :, trial_indices], axis=2)
    
print(f"\nFinal all_erps shape: {all_erps.shape}")
print(f"  Dimension 1 (Channels): {all_erps.shape[0]}")
print(f"  Dimension 2 (Time points): {all_erps.shape[1]}")
print(f"  Dimension 3 (Conditions): {all_erps.shape[2]}")

# =============================================================================
# Task 5: Analyze Channel 7 ERPs
# =============================================================================

print("\n" + "=" * 80)
print("Task 5: Analyze Channel 7 ERPs")
print("=" * 80)

channel_of_interest = 7
faces_idx = 0  
faces_texture_idx = 2  

# Plot ERP for Faces and Faces-as-Texture conditions
erp_faces = all_erps[channel_of_interest, :, faces_idx]
erp_faces_texture = all_erps[channel_of_interest, :, faces_texture_idx]

# Calculate y-axis limits from raw ERPs (for consistency across Figures 1 and 2)
y_min = min(np.min(erp_faces), np.min(erp_faces_texture))
y_max = max(np.max(erp_faces), np.max(erp_faces_texture))
y_margin = (y_max - y_min) * 0.1
y_limits = [y_min - y_margin, y_max + y_margin]

# Create Figure 1
plt.figure(figsize=(12, 6))
plt.plot(time_vec, erp_faces, 'b-', linewidth=2, label='Faces')
plt.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.7, label='Stimulus onset')
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Amplitude (μV)', fontsize=12)
plt.title(f'ERP for Faces Condition - Channel {channel_of_interest}', 
          fontsize=14, fontweight='bold')
plt.ylim(y_limits)  # Apply consistent y-axis limits
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Task5a_Figure1_faces_ERP.png', dpi=300, bbox_inches='tight')
print("Figure 1 saved as 'Task5a_Figure1_faces_ERP.png'")
plt.close()

# Create Figure 1 with both conditions
plt.figure(figsize=(12, 6))
plt.plot(time_vec, erp_faces, 'b-', linewidth=2, label='Faces')
plt.plot(time_vec, erp_faces_texture, 'r-', linewidth=2, label='Faces as Texture')
plt.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.7, label='Stimulus onset')
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Amplitude (μV)', fontsize=12)
plt.title(f'ERPs for Faces and Faces-as-Texture - Channel {channel_of_interest}', 
          fontsize=14, fontweight='bold')
plt.ylim(y_limits)  # Apply consistent y-axis limits
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Task5b_Figure1_both_faces_conditions.png', dpi=300, bbox_inches='tight')
print("Figure 1 (updated) saved as 'Task5b_Figure1_both_faces_conditions.png'")
plt.close()

# Implement Smoothing Function

def my_smooth(signal):
    """
    Smooth a signal using a simple moving average over 11 sample points.
    
    Parameters:
    -----------
    signal : numpy array
        Input signal to be smoothed
        
    Returns:
    --------
    smoothed_signal : numpy array
        Smoothed signal with same length as input
        
    Implementation:
    ---------------
    - Uses an 11-point moving average window
    - Pads edges with first/last values to maintain signal length
    """

    window_size = 11
    half_window = window_size // 2  # 5
    
    signal_length = len(signal)
    smoothed_signal = np.zeros(signal_length)
    
    # Pad the signal at edges
    padded_signal = np.concatenate([
        np.repeat(signal[0], half_window),  # Pad beginning
        signal,
        np.repeat(signal[-1], half_window)  # Pad end
    ])
    
    # Apply moving average
    for i in range(signal_length):
        window_start = i
        window_end = i + window_size
        smoothed_signal[i] = np.mean(padded_signal[window_start:window_end])
    
    return smoothed_signal

# Smooth both ERPs
erp_faces_smooth = my_smooth(erp_faces)
erp_faces_texture_smooth = my_smooth(erp_faces_texture)

# Create Figure 2 with smoothed ERPs
plt.figure(figsize=(12, 6))
plt.plot(time_vec, erp_faces_smooth, 'b-', linewidth=2, label='Faces (smoothed)')
plt.plot(time_vec, erp_faces_texture_smooth, 'r-', linewidth=2, label='Faces as Texture (smoothed)')
plt.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.7, label='Stimulus onset')
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Amplitude (μV)', fontsize=12)
plt.title(f'Smoothed ERPs for Faces and Faces-as-Texture - Channel {channel_of_interest}', 
          fontsize=14, fontweight='bold')
plt.ylim(y_limits)  # Use same y-scale as Figure 1
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Task5d_Figure2_smoothed_ERPs.png', dpi=300, bbox_inches='tight')
print("Figure 2 saved as 'Task5d_Figure2_smoothed_ERPs.png'")
print(f"Y-axis limits: {y_limits}")
plt.close()

# Compute difference: Faces - Faces as Texture
erp_difference = erp_faces_smooth - erp_faces_texture_smooth

# Create Figure 3 with difference wave
plt.figure(figsize=(12, 6))
plt.plot(time_vec, erp_difference, 'purple', linewidth=2.5, label='Difference (Faces - Faces as Texture)')
plt.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.7, label='Stimulus onset')
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Amplitude Difference (μV)', fontsize=12)
plt.title(f'Difference Wave: Faces - Faces as Texture - Channel {channel_of_interest}', 
          fontsize=14, fontweight='bold')
plt.ylim(y_limits)  # Use same y-scale for comparison
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Task5e_Figure3_difference_wave.png', dpi=300, bbox_inches='tight')
print("Figure 3 saved as 'Task5e_Figure3_difference_wave.png'")
plt.close()
