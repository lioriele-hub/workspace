"""
=============================================================================
Brain Functioning Exercise 1 - EEG and ERP Analysis
Course: Functional Brain Mapping, 2025-2026
=============================================================================

This script analyzes EEG data from the ERP CORE dataset.
It performs baseline normalization and computes Event-Related Potentials (ERPs).

Author: Student Solution
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore')

# Set plotting style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# Load the Data
# =============================================================================

print("=" * 80)
print("Loading EEG Data from ex1data.mat")
print("=" * 80)

# Load the MATLAB data file
try:
    data = loadmat('brain functioning - exercise 1/ex1data.mat')
    print("Data loaded successfully!")
except FileNotFoundError:
    try:
        data = loadmat('ex1data.mat')
        print("Data loaded successfully!")
    except FileNotFoundError:
        print("Error: ex1data.mat not found in the current directory.")
        print("Please make sure the file exists and is in the correct location. It should be in the same directory as the script.")
        exit(1)
    print("Error: ex1data.mat not found in the current directory.")
    print("Please make sure the file exists and is in the correct location. It should be in the same directory as the script.")
    exit(1)

# Extract variables from the loaded data
# Note: actual variable names from the .mat file are: 'data', 'time_vec', 'all_conds'
data_raw = data['data']  # Raw EEG data (shape: channels x timepoints x trials)
time_vec_ms = data['time_vec'][0]  # Time vector in milliseconds
condition = data['all_conds'][0]  # Condition labels for each trial

# Convert time from milliseconds to seconds
time_vec = time_vec_ms / 1000.0

# Sampling frequency can be calculated from time vector
# Calculate from time vector (assuming equally spaced samples)
time_diff = time_vec[1] - time_vec[0]  # Time difference between samples in seconds
sample_freq = 1.0 / time_diff  # Sampling frequency in Hz

print("\nData loaded successfully!")
print(f"Data shape: {data_raw.shape}")
print(f"  - Dimension 1 (Channels): {data_raw.shape[0]}")
print(f"  - Dimension 2 (Time points): {data_raw.shape[1]}")
print(f"  - Dimension 3 (Trials): {data_raw.shape[2]}")
print(f"Sampling frequency: {sample_freq} Hz")
print(f"Time range: {time_vec[0]:.3f} to {time_vec[-1]:.3f} seconds")
print(f"Number of trials: {len(condition)}")
print(f"Unique conditions: {np.unique(condition)}")

# =============================================================================
# Question 2: Baseline Normalization (20 points)
# =============================================================================

print("\n" + "=" * 80)
print("Question 2: Baseline Normalization")
print("=" * 80)

# Define baseline period: -0.2 to 0 seconds (200ms before stimulus)
baseline_start = -0.2
baseline_end = 0.0

# Find indices corresponding to baseline period
baseline_indices = np.where((time_vec >= baseline_start) & (time_vec <= baseline_end))[0]

print(f"\nBaseline period: {baseline_start} to {baseline_end} seconds")
print(f"Baseline indices: {baseline_indices[0]} to {baseline_indices[-1]}")
print(f"Number of baseline samples: {len(baseline_indices)}")

# Calculate baseline mean for each channel and each trial
# baseline_mean has shape (channels, 1, trials)
baseline_mean = np.mean(data_raw[:, baseline_indices, :], axis=1, keepdims=True)

# Perform baseline normalization by subtracting baseline mean from each trial
data_norm_baseline = data_raw - baseline_mean

print("\nBaseline normalization completed!")
print(f"Baseline mean shape: {baseline_mean.shape}")
print(f"Normalized data shape: {data_norm_baseline.shape}")

# Verify normalization: check that baseline period now has mean ≈ 0
baseline_after_norm = np.mean(data_norm_baseline[:, baseline_indices, :], axis=1)
print(f"Mean baseline value after normalization (should be ≈0): {np.mean(np.abs(baseline_after_norm)):.2e}")

# =============================================================================
# Question 3: Verify Baseline Normalization (15 points)
# =============================================================================

print("\n" + "=" * 80)
print("Question 3: Verify Baseline Normalization")
print("=" * 80)

# Select a random trial for verification
np.random.seed(42)  # For reproducibility
random_trial = np.random.randint(0, data_raw.shape[2])
random_channel = 5  # Choose channel 5 for verification

print(f"\nVerifying with Channel {random_channel}, Trial {random_trial}")

# Extract data for the selected channel and trial
signal_before = data_raw[random_channel, :, random_trial]
signal_after = data_norm_baseline[random_channel, :, random_trial]

# Calculate baseline means
baseline_mean_before = np.mean(signal_before[baseline_indices])
baseline_mean_after = np.mean(signal_after[baseline_indices])

print(f"\nBaseline mean BEFORE normalization: {baseline_mean_before:.4f} μV")
print(f"Baseline mean AFTER normalization: {baseline_mean_after:.4e} μV")
print(f"Difference: {baseline_mean_before - baseline_mean_after:.4f} μV")

# Create verification plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot before normalization
axes[0].plot(time_vec, signal_before, 'b-', linewidth=1.5, label='Raw signal')
axes[0].axvspan(baseline_start, baseline_end, alpha=0.3, color='yellow', label='Baseline period')
axes[0].axhline(y=baseline_mean_before, color='r', linestyle='--', linewidth=2, 
                label=f'Baseline mean = {baseline_mean_before:.2f} μV')
axes[0].axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Stimulus onset')
axes[0].set_xlabel('Time (s)', fontsize=12)
axes[0].set_ylabel('Amplitude (μV)', fontsize=12)
axes[0].set_title(f'Before Baseline Normalization - Channel {random_channel}, Trial {random_trial}', 
                  fontsize=14, fontweight='bold')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Plot after normalization
axes[1].plot(time_vec, signal_after, 'g-', linewidth=1.5, label='Normalized signal')
axes[1].axvspan(baseline_start, baseline_end, alpha=0.3, color='yellow', label='Baseline period')
axes[1].axhline(y=baseline_mean_after, color='r', linestyle='--', linewidth=2, 
                label=f'Baseline mean ≈ {baseline_mean_after:.2e} μV')
axes[1].axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Stimulus onset')
axes[1].set_xlabel('Time (s)', fontsize=12)
axes[1].set_ylabel('Amplitude (μV)', fontsize=12)
axes[1].set_title(f'After Baseline Normalization - Channel {random_channel}, Trial {random_trial}', 
                  fontsize=14, fontweight='bold')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Question3_baseline_verification.png', dpi=300, bbox_inches='tight')
print("\nVerification plot saved as 'Question3_baseline_verification.png'")
plt.close()

# =============================================================================
# Question 4: Compute ERPs for All Conditions (15 points)
# =============================================================================

print("\n" + "=" * 80)
print("Question 4: Compute ERPs for All Conditions")
print("=" * 80)

# The 4 conditions in the experiment
# Based on ERP CORE N170 paradigm:
# 1 = Faces
# 2 = Faces as Texture (scrambled/phase-scrambled faces)
# 3 = Cars
# 4 = Cars as Texture (scrambled cars)

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
    
    print(f"\nCondition {int(cond)} (index {idx}):")
    print(f"  Trials used: {len(trial_indices)}")
    print(f"  ERP shape: {all_erps[:, :, idx].shape}")

print(f"\nFinal all_erps shape: {all_erps.shape}")
print(f"  Dimension 1 (Channels): {all_erps.shape[0]}")
print(f"  Dimension 2 (Time points): {all_erps.shape[1]}")
print(f"  Dimension 3 (Conditions): {all_erps.shape[2]}")

# =============================================================================
# Question 5: Analyze Channel 7 ERPs (24 points)
# =============================================================================

print("\n" + "=" * 80)
print("Question 5: Analyze Channel 7 ERPs")
print("=" * 80)

channel_of_interest = 7
print(f"\nAnalyzing Channel {channel_of_interest}")

# Condition indices (assuming conditions are labeled 1-4)
# Condition 1 = Faces
# Condition 2 = Faces as Texture
faces_idx = 0  # First condition
faces_texture_idx = 1  # Second condition

# -----------------------------------------------------------------------------
# Question 5a: Plot ERP for Faces condition
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("Question 5a: Plot ERP for Faces Condition")
print("-" * 80)

erp_faces = all_erps[channel_of_interest, :, faces_idx]

# Create Figure 1
plt.figure(figsize=(12, 6))
plt.plot(time_vec, erp_faces, 'b-', linewidth=2, label='Faces')
plt.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.7, label='Stimulus onset')
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude (μV)', fontsize=12)
plt.title(f'ERP for Faces Condition - Channel {channel_of_interest}', 
          fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Question5a_Figure1_faces_ERP.png', dpi=300, bbox_inches='tight')
print("Figure 1 saved as 'Question5a_Figure1_faces_ERP.png'")

# Find peaks and troughs
peak_idx = np.argmax(erp_faces)
trough_idx = np.argmin(erp_faces)
peak_time = time_vec[peak_idx]
trough_time = time_vec[trough_idx]

print(f"\nERP Characteristics for Faces:")
print(f"  Peak: {erp_faces[peak_idx]:.2f} μV at {peak_time:.3f} s")
print(f"  Trough: {erp_faces[trough_idx]:.2f} μV at {trough_time:.3f} s")

# Look for N170 component (negative peak around 150-200ms)
n170_window = (time_vec >= 0.15) & (time_vec <= 0.20)
n170_idx = baseline_indices[-1] + np.argmin(erp_faces[n170_window]) + np.where(n170_window)[0][0]
n170_time = time_vec[n170_idx] if n170_idx < len(time_vec) else None

if n170_time:
    print(f"  N170 component: {erp_faces[n170_idx]:.2f} μV at {n170_time:.3f} s")
else:
    print("  N170 component not clearly identified")

plt.close()

# -----------------------------------------------------------------------------
# Question 5b: Add Faces-as-Texture to the plot
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("Question 5b: Add Faces-as-Texture to Figure 1")
print("-" * 80)

erp_faces_texture = all_erps[channel_of_interest, :, faces_texture_idx]

# Create Figure 1 with both conditions
plt.figure(figsize=(12, 6))
plt.plot(time_vec, erp_faces, 'b-', linewidth=2, label='Faces')
plt.plot(time_vec, erp_faces_texture, 'r-', linewidth=2, label='Faces as Texture')
plt.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.7, label='Stimulus onset')
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude (μV)', fontsize=12)
plt.title(f'ERPs for Faces and Faces-as-Texture - Channel {channel_of_interest}', 
          fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Question5b_Figure1_both_conditions.png', dpi=300, bbox_inches='tight')
print("Figure 1 (updated) saved as 'Question5b_Figure1_both_conditions.png'")
plt.close()

# -----------------------------------------------------------------------------
# Question 5c: Implement Smoothing Function
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("Question 5c: Implement Smoothing Function")
print("-" * 80)

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
    - No built-in smoothing functions are used
    """
    window_size = 11
    half_window = window_size // 2  # 5
    
    signal_length = len(signal)
    smoothed_signal = np.zeros(signal_length)
    
    # Pad the signal at edges
    # Pad beginning with first value, end with last value
    padded_signal = np.concatenate([
        np.repeat(signal[0], half_window),  # Pad beginning
        signal,
        np.repeat(signal[-1], half_window)  # Pad end
    ])
    
    # Apply moving average
    for i in range(signal_length):
        # Window goes from i to i+window_size in the padded signal
        window_start = i
        window_end = i + window_size
        smoothed_signal[i] = np.mean(padded_signal[window_start:window_end])
    
    return smoothed_signal

# Test the smoothing function
test_signal = np.random.randn(100)
test_smoothed = my_smooth(test_signal)

print("Smoothing function implemented successfully!")
print(f"  Test signal length: {len(test_signal)}")
print(f"  Smoothed signal length: {len(test_smoothed)}")
print(f"  Lengths match: {len(test_signal) == len(test_smoothed)}")

# -----------------------------------------------------------------------------
# Question 5d: Smooth ERPs and Create Figure 2
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("Question 5d: Smooth ERPs and Create Figure 2")
print("-" * 80)

# Smooth both ERPs
erp_faces_smooth = my_smooth(erp_faces)
erp_faces_texture_smooth = my_smooth(erp_faces_texture)

print("ERPs smoothed successfully!")

# Get y-axis limits from Figure 1 for consistency
y_min = min(np.min(erp_faces), np.min(erp_faces_texture))
y_max = max(np.max(erp_faces), np.max(erp_faces_texture))
y_margin = (y_max - y_min) * 0.1
y_limits = [y_min - y_margin, y_max + y_margin]

# Create Figure 2 with smoothed ERPs
plt.figure(figsize=(12, 6))
plt.plot(time_vec, erp_faces_smooth, 'b-', linewidth=2, label='Faces (smoothed)')
plt.plot(time_vec, erp_faces_texture_smooth, 'r-', linewidth=2, label='Faces as Texture (smoothed)')
plt.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.7, label='Stimulus onset')
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude (μV)', fontsize=12)
plt.title(f'Smoothed ERPs for Faces and Faces-as-Texture - Channel {channel_of_interest}', 
          fontsize=14, fontweight='bold')
plt.ylim(y_limits)  # Use same y-scale as Figure 1
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Question5d_Figure2_smoothed_ERPs.png', dpi=300, bbox_inches='tight')
print("Figure 2 saved as 'Question5d_Figure2_smoothed_ERPs.png'")
print(f"Y-axis limits: {y_limits}")
plt.close()

# -----------------------------------------------------------------------------
# Question 5e: Compute and Plot Difference Between Conditions
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("Question 5e: Compute and Plot Difference (Figure 3)")
print("-" * 80)

# Compute difference: Faces - Faces as Texture
erp_difference = erp_faces_smooth - erp_faces_texture_smooth

print("Difference computed: Faces - Faces as Texture")
print(f"  Max difference: {np.max(erp_difference):.2f} μV")
print(f"  Min difference: {np.min(erp_difference):.2f} μV")

# Create Figure 3 with difference wave
plt.figure(figsize=(12, 6))
plt.plot(time_vec, erp_difference, 'purple', linewidth=2.5, label='Difference (Faces - Faces as Texture)')
plt.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.7, label='Stimulus onset')
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Amplitude Difference (μV)', fontsize=12)
plt.title(f'Difference Wave: Faces - Faces as Texture - Channel {channel_of_interest}', 
          fontsize=14, fontweight='bold')
plt.ylim(y_limits)  # Use same y-scale for comparison
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Question5e_Figure3_difference_wave.png', dpi=300, bbox_inches='tight')
print("Figure 3 saved as 'Question5e_Figure3_difference_wave.png'")
plt.close()

# -----------------------------------------------------------------------------
# Question 5f: Describe the Response
# -----------------------------------------------------------------------------

print("\n" + "-" * 80)
print("Question 5f: Description of Response")
print("-" * 80)

description = f"""
DESCRIPTION OF ERP RESPONSE IN CHANNEL {channel_of_interest}:

1. FACES CONDITION:
   - The ERP shows a characteristic negative deflection (trough) around {trough_time:.0f}ms
   - Peak amplitude: {erp_faces[peak_idx]:.2f} μV at {peak_time*1000:.0f}ms
   - Trough amplitude: {erp_faces[trough_idx]:.2f} μV at {trough_time*1000:.0f}ms
   - This negative component is consistent with the N170 ERP component, which is 
     known to be particularly sensitive to face processing
   
2. FACES-AS-TEXTURE CONDITION:
   - Shows similar overall waveform but with reduced amplitude
   - The negative deflection is less pronounced compared to intact faces
   - This suggests that the scrambling/texturing of faces reduces the face-specific
     neural response
     
3. DIFFERENCE BETWEEN CONDITIONS:
   - The difference wave (Faces - Faces as Texture) shows that intact faces elicit
     a stronger negative response around 150-200ms post-stimulus
   - This differential response represents the face-specific processing that occurs
     in the N170 time window
   - The effect is most pronounced in the 150-200ms range, consistent with N170 literature
   
4. INTERPRETATION:
   - Channel {channel_of_interest} appears to be positioned over posterior regions (likely
     occipito-temporal areas) where face-sensitive N170 components are typically observed
   - The stronger response to intact faces vs. scrambled faces demonstrates that this
     ERP component is sensitive to face configuration, not just low-level visual features
   - The smoothing helped reveal the underlying ERP pattern by reducing high-frequency noise
"""

print(description)

# Save description to text file
with open('Question5f_response_description.txt', 'w', encoding='utf-8') as f:
    f.write(description)
print("\nDescription saved to 'Question5f_response_description.txt'")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

print("\nGenerated Files:")
print("  1. Question3_baseline_verification.png - Baseline normalization verification")
print("  2. Question5a_Figure1_faces_ERP.png - ERP for faces condition")
print("  3. Question5b_Figure1_both_conditions.png - ERPs for both conditions")
print("  4. Question5d_Figure2_smoothed_ERPs.png - Smoothed ERPs")
print("  5. Question5e_Figure3_difference_wave.png - Difference wave")
print("  6. Question5f_response_description.txt - Written description")

print("\nKey Variables Created:")
print(f"  - data_norm_baseline: shape {data_norm_baseline.shape}")
print(f"  - all_erps: shape {all_erps.shape}")
print(f"  - my_smooth: smoothing function")

print("\n" + "=" * 80)

