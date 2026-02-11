"""
=============================================================================
Brain Functioning Exercise 3 - fMRI Signal Analysis 
=============================================================================
This solution analyzes fMRI data using the General Linear Model (GLM) approach,
including HRF convolution and signal prediction.

Topics covered:
- Part 1: Understanding Hemodynamic Response Function (HRF)
- Part 2: Convolution of stimulus with HRF for fMRI signal prediction
- Part 3: GLM analysis of fMRI data from FAO experiment
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
from scipy import io as sio
from scipy.signal import convolve
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')


# =============================================================================
# PART 1 - Understanding the Hemodynamic Response Function (HRF)
# =============================================================================

print("\n" + "=" * 80)
print("PART 1 - Understanding the Hemodynamic Response Function (HRF)")
print("=" * 80)

# Load the HRF data for Question 1
hrf_data = sio.loadmat('hrf_Q1.mat')
hrf = hrf_data['hrf'].flatten()

print(f"\nHRF loaded: {len(hrf)} time points")
print(f"HRF values:\n{hrf}")

# Plot the HRF
fig, ax = plt.subplots(figsize=(10, 6))
TR_hrf = 2  # TR = 2 seconds as specified in exercise instructions
time_hrf = np.arange(len(hrf)) * TR_hrf  # Time in seconds (0 to 32 seconds in 2-second steps)
ax.plot(time_hrf, hrf, 'b-o', linewidth=2, markersize=6)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('HRF Amplitude', fontsize=12)
ax.set_title('Hemodynamic Response Function (HRF)', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Part1_HRF.png', dpi=150)
plt.close()
print("HRF plot saved as 'Part1_HRF.png'")

# Describe HRF characteristics
print("\nHRF Characteristics:")
print(f"  Peak amplitude: {np.max(hrf):.4f}")
print(f"  Peak time: {np.argmax(hrf) * TR_hrf} seconds (at index {np.argmax(hrf)})")
print(f"  Minimum (undershoot): {np.min(hrf):.4f}")
print(f"  Undershoot time: {np.argmin(hrf) * TR_hrf} seconds (at index {np.argmin(hrf)})")


# =============================================================================
# PART 2 - Stimulus Timing Effects on Hemodynamic Response (Question 2)
# =============================================================================

print("\n" + "=" * 80)
print("PART 2 - Stimulus Timing Effects on Hemodynamic Response")
print("=" * 80)

# Define experimental parameters 
TR = 2  # Repetition time in seconds
total_time = 96  # Total experiment duration in seconds
n_timepoints = int(total_time / TR)  # 48 time points
time = np.arange(0, total_time, TR)

# =============================================================================
# PART 2a - Dense vs Sparse Stimuli
# =============================================================================

print("\n" + "-" * 80)
print("PART 2a - Dense vs Sparse Stimuli")
print("-" * 80)

# Create dense stimuli every 4 seconds (every 2 TRs since TR=2)
dense_interval = 4  # seconds
stimulus_dense = np.zeros(n_timepoints)
stimulus_dense[::int(dense_interval/TR)] = 1

# Create sparse stimuli every 16 seconds (every 8 TRs since TR=2)
sparse_interval = 16  # seconds
stimulus_sparse = np.zeros(n_timepoints)
stimulus_sparse[::int(sparse_interval/TR)] = 1

# Convolve both with HRF
response_dense = convolve(stimulus_dense, hrf, mode='full')[:n_timepoints]
response_sparse = convolve(stimulus_sparse, hrf, mode='full')[:n_timepoints]

# Plot Dense Stimuli
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1 = axes[0]
ax1.stem(time, stimulus_dense, linefmt='b-', markerfmt='bo', basefmt='k-', label='Stimulus events')
ax1.set_xlabel('Time (seconds)', fontsize=12)
ax1.set_ylabel('Stimulus (Delta Function)', fontsize=12)
ax1.set_title('Dense Stimuli - Timing (ISI = 4 seconds)', fontsize=13)
ax1.set_ylim(-0.1, 1.3)
ax1.set_xticks(np.arange(0, total_time + 1, 16))
ax1.tick_params(labelbottom=True)
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2 = axes[1]
ax2.plot(time, response_dense, 'r-', linewidth=2.5, label='BOLD Response')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time (seconds)', fontsize=12)
ax2.set_ylabel('BOLD Response', fontsize=12)
ax2.set_title('Hemodynamic Response to Dense Stimuli (after convolution with HRF)', fontsize=13)
ax2.set_xticks(np.arange(0, total_time + 1, 16))
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('Part2a_dense_stimuli.png', dpi=150)
plt.close()
print("\nDense stimuli plot saved as 'Part2a_dense_stimuli.png'")

# Plot Sparse Stimuli
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1 = axes[0]
ax1.stem(time, stimulus_sparse, linefmt='g-', markerfmt='go', basefmt='k-', label='Stimulus events')
ax1.set_xlabel('Time (seconds)', fontsize=12)
ax1.set_ylabel('Stimulus (Delta Function)', fontsize=12)
ax1.set_title('Sparse Stimuli - Timing (ISI = 16 seconds)', fontsize=13)
ax1.set_ylim(-0.1, 1.3)
ax1.set_xticks(np.arange(0, total_time + 1, 16))
ax1.tick_params(labelbottom=True)
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2 = axes[1]
ax2.plot(time, response_sparse, 'purple', linewidth=2.5, label='BOLD Response')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time (seconds)', fontsize=12)
ax2.set_ylabel('BOLD Response', fontsize=12)
ax2.set_title('Hemodynamic Response to Sparse Stimuli (after convolution with HRF)', fontsize=13)
ax2.set_xticks(np.arange(0, total_time + 1, 16))
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('Part2a_sparse_stimuli.png', dpi=150)
plt.close()
print("Sparse stimuli plot saved as 'Part2a_sparse_stimuli.png'")

print("\nAnalysis:")
print(f"  Dense max amplitude: {np.max(response_dense):.4f}")
print(f"  Sparse max amplitude: {np.max(response_sparse):.4f}")

# =============================================================================
# PART 2b - Expected PSC for Sparse Stimuli
# =============================================================================

print("\n" + "-" * 80)
print("PART 2b - Expected PSC (Percentage Signal Change) for Sparse Stimuli")
print("-" * 80)

# Scale sparse response to have max PSC of 1.8%
max_psc = 1.8  # percent
psc_sparse = (response_sparse / np.max(response_sparse)) * max_psc

print(f"\nExpected maximum PSC: {max_psc}%")
print(f"Actual maximum PSC: {np.max(psc_sparse):.4f}%")

# Plot expected PSC
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(time, psc_sparse, 'purple', linewidth=2.5, label=f'Expected PSC (max={max_psc}%)')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=max_psc, color='red', linestyle=':', alpha=0.5, label=f'Max PSC = {max_psc}%')
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Signal Change (%)', fontsize=12)
ax.set_title('Expected BOLD Response for Sparse Stimuli (without noise)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Part2b_expected_psc.png', dpi=150)
plt.close()
print("Expected PSC plot saved as 'Part2b_expected_psc.png'")

# =============================================================================
# PART 2c - Add Noise to PSC Signal
# =============================================================================

print("\n" + "-" * 80)
print("PART 2c - Effect of Noise on Signal")
print("-" * 80)

# Add low noise (SD = 0.3%)
np.random.seed(42)  # For reproducibility
noise_low = np.random.normal(0, 0.3, n_timepoints)
psc_low_noise = psc_sparse + noise_low

# Add high noise (SD = 3%)
noise_high = np.random.normal(0, 3.0, n_timepoints)
psc_high_noise = psc_sparse + noise_high

print(f"\nLow noise: SD = 0.3%")
print(f"  SNR (signal/noise): {max_psc / 0.3:.2f}")
print(f"\nHigh noise: SD = 3.0%")
print(f"  SNR (signal/noise): {max_psc / 3.0:.2f}")

# Plot signals with noise 
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(time, psc_sparse, 'purple', linewidth=2.5, label='Original (no noise)', zorder=3)
ax.plot(time, psc_low_noise, 'b-', linewidth=1.5, alpha=0.7, label='With low noise (SD=0.3%)', zorder=2)
ax.plot(time, psc_high_noise, 'r-', linewidth=1.5, alpha=0.7, label='With high noise (SD=3%)', zorder=1)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Signal Change (%)', fontsize=12)
ax.set_title('Effect of Noise on Signal: Original vs Low Noise (SD=0.3%) vs High Noise (SD=3%)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(-7, 7)
ax.set_xticks(np.arange(0, total_time + 1, 16))
plt.tight_layout()
plt.savefig('Part2c_noise_effects.png', dpi=150)
plt.close()
print("Noise effects plot saved as 'Part2c_noise_effects.png'")

# =============================================================================
# PART 2d - Block Design
# =============================================================================

print("\n" + "-" * 80)
print("PART 2d - Block Design vs Individual Stimuli")
print("-" * 80)

# Create block design: 16s ON, 16s OFF
stimulus_block = np.zeros(n_timepoints)
block_on = 16  # seconds
block_off = 16  # seconds
cycle_duration = block_on + block_off  # 32 seconds

# Set values to 1 during ON periods
stimulus_block[time % cycle_duration < block_on] = 1

# Convolve with HRF
response_block = convolve(stimulus_block, hrf, mode='full')[:n_timepoints]

print(f"\nMaximum amplitudes:")
print(f"  Block design: {np.max(response_block):.4f}")
print(f"  Dense stimuli (ISI=4s): {np.max(response_dense):.4f}")
print(f"  Ratio (Block/Dense): {np.max(response_block)/np.max(response_dense):.4f}")

# Plot block design
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

ax1 = axes[0]
ax1.fill_between(time, 0, stimulus_block, alpha=0.6, color='orange', step='pre')
ax1.plot(time, stimulus_block, 'orange', linewidth=1.5, drawstyle='steps-pre')
ax1.set_ylabel('Stimulus', fontsize=12)
ax1.set_title(f'Block Design - Stimulus (16s ON / 16s OFF)', fontsize=13)
ax1.set_ylim(-0.1, 1.3)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['OFF', 'ON'])
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(time, response_block, 'darkorange', linewidth=2.5, label='Block design response')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time (seconds)', fontsize=12)
ax2.set_ylabel('BOLD Response', fontsize=12)
ax2.set_title('Hemodynamic Response to Block Design', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Part2d_block_design.png', dpi=150)
plt.close()
print("Block design plot saved as 'Part2d_block_design.png'")

# =============================================================================
# PART 3 - FAO (Faces, Animals, Objects) fMRI Experiment Analysis
# =============================================================================

print("\n" + "=" * 80)
print("PART 3 - FAO fMRI Experiment: Design Matrix and GLM Analysis")
print("=" * 80)

# =============================================================================
# PART 3a - Calculate Experiment Parameters
# =============================================================================

print("\n" + "-" * 80)
print("PART 3a - Experiment Parameters")
print("-" * 80)

# Experiment parameters
TR_fao = 2.5  # Repetition time in seconds
image_duration = 1.0  # Image display duration in seconds
isi_duration = 0.25  # Inter-stimulus interval in seconds
images_per_block = 12  # Number of images per condition block
baseline_duration = 20  # Baseline block duration in seconds

# Calculate active block duration (for F, A, O conditions)
active_block_duration = images_per_block * (image_duration + isi_duration)

print(f"\nExperiment Timing:")
print(f"  Image duration: {image_duration} second")
print(f"  ISI (Inter-Stimulus Interval): {isi_duration} seconds")
print(f"  Images per block: {images_per_block}")
print(f"  Duration per image + ISI: {image_duration + isi_duration} seconds")
print(f"  Active block duration (F/A/O): {active_block_duration} seconds")
print(f"  Baseline block duration: {baseline_duration} seconds")
print(f"  TR: {TR_fao} seconds")

# Block sequence: B F A O B A O F B O F A B F O A B O A F B A F O B
block_sequence = ['B', 'F', 'A', 'O', 'B', 'A', 'O', 'F', 'B', 'O', 'F', 'A', 
                  'B', 'F', 'O', 'A', 'B', 'O', 'A', 'F', 'B', 'A', 'F', 'O', 'B']

# Calculate total experiment duration
total_blocks = len(block_sequence)
total_time_fao = sum([baseline_duration if block == 'B' else active_block_duration for block in block_sequence])
n_timepoints_fao = int(np.ceil(total_time_fao / TR_fao))

print(f"\nBlock sequence: {' -> '.join(block_sequence)}")
print(f"  Total blocks: {total_blocks}")
print(f"  Total experiment duration: {total_time_fao} seconds")
print(f"  Number of volumes (TR={TR_fao}s): {n_timepoints_fao}")

# =============================================================================
# PART 3b - Create Initial Design Matrix (before HRF convolution)
# =============================================================================

print("\n" + "-" * 80)
print("PART 3b - Initial Design Matrix (before HRF convolution)")
print("-" * 80)

# Create time array
time_fao = np.arange(n_timepoints_fao) * TR_fao

# Create initial design matrix for F, A, O conditions
design_matrix_initial = np.zeros((n_timepoints_fao, 3))  # F, A, O columns
current_time = 0

for block_type in block_sequence:
    block_dur = baseline_duration if block_type == 'B' else active_block_duration
    block_start_idx = int(np.round(current_time / TR_fao))
    block_end_idx = int(np.round((current_time + block_dur) / TR_fao))
    
    if block_type == 'F':
        design_matrix_initial[block_start_idx:block_end_idx, 0] = 1
    elif block_type == 'A':
        design_matrix_initial[block_start_idx:block_end_idx, 1] = 1
    elif block_type == 'O':
        design_matrix_initial[block_start_idx:block_end_idx, 2] = 1
    
    current_time += block_dur

print(f"Initial Design Matrix shape: {design_matrix_initial.shape}")
print(f"  Columns: 0=Faces (F), 1=Animals (A), 2=Objects (O)")
print(f"  Rows: {n_timepoints_fao} time points")

# Plot initial design matrix as heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(design_matrix_initial.T, aspect='auto', cmap='gray', interpolation='nearest')
ax.set_xlabel('Time points', fontsize=12)
ax.set_ylabel('Condition', fontsize=12)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['Faces (F)', 'Animals (A)', 'Objects (O)'])
ax.set_title('Initial Design Matrix (before HRF convolution)', fontsize=14)
plt.colorbar(im, ax=ax, label='Condition (0=baseline, 1=active)')
plt.tight_layout()
plt.savefig('Part3b_design_matrix_initial.png', dpi=150)
plt.close()
print("Initial design matrix heatmap saved as 'Part3b_design_matrix_initial.png'")

# =============================================================================
# PART 3c - Create Final Design Matrix (after HRF convolution)
# =============================================================================

print("\n" + "-" * 80)
print("PART 3c - Final Design Matrix (after HRF convolution)")
print("-" * 80)

# Load the HRF for FAO analysis
hrf_fao_data = sio.loadmat('hrf_FAO_Q3.mat')
hrf_fao = hrf_fao_data['hrf'].flatten()

print(f"\nHRF loaded: {len(hrf_fao)} time points")
print(f"HRF values:\n{hrf_fao}")

# Convolve each condition with HRF
design_matrix_convolved = np.zeros((n_timepoints_fao, 3))
for condition_idx in range(3):
    design_matrix_convolved[:, condition_idx] = convolve(design_matrix_initial[:, condition_idx], 
                                                          hrf_fao, mode='full')[:n_timepoints_fao]

# Create final design matrix with intercept
design_matrix_final = np.column_stack([design_matrix_convolved, np.ones(n_timepoints_fao)])

print(f"Final Design Matrix shape: {design_matrix_final.shape}")
print(f"  Columns: 0=Faces (convolved), 1=Animals (convolved), 2=Objects (convolved), 3=Intercept")
print(f"  Rows: {n_timepoints_fao} time points")

# Plot final design matrix as heatmap
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(design_matrix_final.T, aspect='auto', cmap='gray', interpolation='nearest')
ax.set_xlabel('Time points', fontsize=12)
ax.set_ylabel('Condition', fontsize=12)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['Faces (F)', 'Animals (A)', 'Objects (O)', 'Intercept'])
ax.set_title('Final Design Matrix (after HRF convolution)', fontsize=14)
plt.colorbar(im, ax=ax, label='Amplitude')
plt.tight_layout()
plt.savefig('Part3c_design_matrix_final.png', dpi=150)
plt.close()
print("\nFinal design matrix heatmap saved as 'Part3c_design_matrix_final.png'")

# =============================================================================
# Load fMRI Data
# =============================================================================

print("\n" + "-" * 80)
print("Loading fMRI Data")
print("-" * 80)

# Load the FAO data
fmri_data = sio.loadmat('fmri_data_FAO.mat')
data = fmri_data['data']  # Shape: (20 voxels, n_timepoints_fao)

n_voxels, n_timepoints_data = data.shape
print(f"\nfMRI Data loaded:")
print(f"  Number of voxels: {n_voxels}")
print(f"  Number of time points: {n_timepoints_data}")
print(f"  Expected time points from experiment: {n_timepoints_fao}")

# =============================================================================
# PART 3d - Voxel 1: Plot BOLD signal with Faces regressor
# =============================================================================

print("\n" + "-" * 80)
print("PART 3d - Voxel 1: BOLD Signal and Faces Regressor")
print("-" * 80)

voxel_1_signal = data[0, :]
faces_regressor = design_matrix_final[:, 0]  # Faces condition

# Normalize for visualization
voxel_1_normalized = (voxel_1_signal - np.mean(voxel_1_signal)) / np.std(voxel_1_signal)
faces_regressor_scaled = (faces_regressor - np.mean(faces_regressor)) / np.std(faces_regressor)

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(time_fao, voxel_1_normalized, 'b-', linewidth=2, label='Voxel 1 BOLD signal', alpha=0.8)
ax.plot(time_fao, faces_regressor_scaled, 'r-', linewidth=2, label='Faces (F) regressor', alpha=0.8)
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Z-scored Signal', fontsize=12)
ax.set_title('Voxel 1: BOLD Signal and Faces Regressor (z-scored for visualization)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Part3d_voxel1_signal_with_faces.png', dpi=150)
plt.close()
print("Voxel 1 signal with faces regressor saved as 'Part3d_voxel1_signal_with_faces.png'")

# Assess temporal correlation
correlation_voxel1_faces = np.corrcoef(voxel_1_signal, faces_regressor)[0, 1]
print(f"\nTemporal correlation between Voxel 1 and Faces regressor: {correlation_voxel1_faces:.4f}")
print(f"Interpretation based on timing alignment:")
if abs(correlation_voxel1_faces) > 0.3:
    print(f"  -> Strong timing correlation suggests Voxel 1 responds to face stimuli")
else:
    print(f"  -> Weak timing correlation suggests limited response to face stimuli")

# =============================================================================
# PART 3e - Voxel 1: Average time course per condition
# =============================================================================

print("\n" + "-" * 80)
print("PART 3e - Voxel 1: Average Time Course Aligned to Block Start")
print("-" * 80)

# Define time window: 0 to 25 seconds from block start
window_duration = 25  # seconds
window_samples = int(window_duration / TR_fao) + 1  # Number of samples

print(f"\nTime window for averaging:")
print(f"  Duration: {window_duration} seconds")
print(f"  TR: {TR_fao} seconds")
print(f"  Number of samples: {window_samples}")

# Extract block onsets for each condition
F_onsets = []  # Faces
A_onsets = []  # Animals
O_onsets = []  # Objects

current_time = 0
for block_type in block_sequence:
    block_dur = baseline_duration if block_type == 'B' else active_block_duration
    block_start_idx = int(np.round(current_time / TR_fao))
    
    if block_type == 'F':
        F_onsets.append(block_start_idx)
    elif block_type == 'A':
        A_onsets.append(block_start_idx)
    elif block_type == 'O':
        O_onsets.append(block_start_idx)
    
    current_time += block_dur

print(f"\nBlock onsets (in timepoint indices):")
print(f"  Faces (F): {F_onsets}")
print(f"  Animals (A): {A_onsets}")
print(f"  Objects (O): {O_onsets}")

# Extract and average time courses
def extract_average_timecourse(signal, onsets, window_samples, max_samples):
    timecourses = []
    for onset in onsets:
        if onset + window_samples <= max_samples:
            timecourse = signal[onset:onset + window_samples]
            timecourses.append(timecourse)
    if timecourses:
        return np.mean(timecourses, axis=0)
    else:
        return None

avg_timecourse_F = extract_average_timecourse(voxel_1_signal, F_onsets, window_samples, n_timepoints_fao)
avg_timecourse_A = extract_average_timecourse(voxel_1_signal, A_onsets, window_samples, n_timepoints_fao)
avg_timecourse_O = extract_average_timecourse(voxel_1_signal, O_onsets, window_samples, n_timepoints_fao)

time_window = np.arange(window_samples) * TR_fao 

# Plot average time courses
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(time_window, avg_timecourse_F, 'r-', linewidth=2.5, marker='o', label='Faces (F)')
ax.plot(time_window, avg_timecourse_A, 'g-', linewidth=2.5, marker='s', label='Animals (A)')
ax.plot(time_window, avg_timecourse_O, 'b-', linewidth=2.5, marker='^', label='Objects (O)')

ax.axhline(y=np.mean(voxel_1_signal), color='gray', linestyle='--', alpha=0.5, label='Baseline mean')
ax.set_xlabel('Time from block start (seconds)', fontsize=12)
ax.set_ylabel('BOLD Signal (a.u.)', fontsize=12)
ax.set_title('Voxel 1: Average Time Course for Each Condition (aligned to block start)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Part3e_voxel1_timecourse_per_condition.png', dpi=150)
plt.close()
print("\nAverage time courses saved as 'Part3e_voxel1_timecourse_per_condition.png'")

# =============================================================================
# PART 3f - GLM Analysis: Voxel 1
# =============================================================================

print("\n" + "-" * 80)
print("PART 3f - GLM Analysis for Voxel 1")
print("-" * 80)

# Perform GLM for voxel 1
y_voxel1 = voxel_1_signal
X = design_matrix_final

# Ordinary Least Squares: beta = (X'X)^(-1) X'y
XtX = X.T @ X
Xty = X.T @ y_voxel1
beta_voxel1 = np.linalg.solve(XtX, Xty)

print(f"\nGLM Results for Voxel 1:")
print(f"  Beta (Faces):     {beta_voxel1[0]:.4f}")
print(f"  Beta (Animals):   {beta_voxel1[1]:.4f}")
print(f"  Beta (Objects):   {beta_voxel1[2]:.4f}")
print(f"  Beta (Intercept): {beta_voxel1[3]:.4f}")

# =============================================================================
# PART 3g - Residuals for Voxel 1
# =============================================================================

print("\n" + "-" * 80)
print("PART 3g - Residuals for Voxel 1")
print("-" * 80)

# Calculate predicted values and residuals
y_pred_voxel1 = X @ beta_voxel1
residuals_voxel1 = y_voxel1 - y_pred_voxel1

# Plot residuals
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(time_fao, residuals_voxel1, 'b-', linewidth=1, alpha=0.8)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.7)
ax.fill_between(time_fao, residuals_voxel1, 0, alpha=0.3)
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Residuals (a.u.)', fontsize=12)
ax.set_title('Voxel 1: GLM Residuals Over Time', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Part3g_voxel1_residuals.png', dpi=150)
plt.close()
print("Residuals plot saved as 'Part3g_voxel1_residuals.png'")

# =============================================================================
# PART 3h - GLM for All Voxels
# =============================================================================

print("\n" + "-" * 80)
print("PART 3h - GLM Analysis for All Voxels")
print("-" * 80)

# Fit GLM for all voxels
beta_matrix = np.zeros((n_voxels, 4))  # 4 columns: F, A, O, Intercept

for voxel_idx in range(n_voxels):
    y = data[voxel_idx, :]
    XtX = X.T @ X
    Xty = X.T @ y
    beta_matrix[voxel_idx, :] = np.linalg.solve(XtX, Xty)

print(f"\nBeta matrix dimensions: {beta_matrix.shape}")
print(f"  Rows: {n_voxels} voxels")
print(f"  Columns: 4 regressors (Faces, Animals, Objects, Intercept)")

# =============================================================================
# PART 3i - ROI Analysis
# =============================================================================

print("\n" + "-" * 80)
print("PART 3i - ROI Analysis: Average Beta Values")
print("-" * 80)

# Define ROIs
roi_definitions = {
    'ROI 1': range(0, 5),    # Voxels 1-5 (0-indexed: 0-4)
    'ROI 2': range(5, 10),   # Voxels 6-10 (0-indexed: 5-9)
    'ROI 3': range(10, 15)   # Voxels 11-15 (0-indexed: 10-14)
}

# Calculate ROI-level statistics
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
roi_names = list(roi_definitions.keys())
conditions = ['Faces', 'Animals', 'Objects']

for roi_idx, (roi_name, voxel_indices) in enumerate(roi_definitions.items()):
    voxel_list = list(voxel_indices)
    roi_betas = beta_matrix[voxel_list, :3]  # Exclude intercept
    roi_mean_betas = np.mean(roi_betas, axis=0)
    roi_std_betas = np.std(roi_betas, axis=0)
    
    print(f"\n{roi_name}:")
    print(f"  Voxels: {[v+1 for v in voxel_list]}")
    print(f"  Mean Betas:")
    for cond_idx, cond_name in enumerate(conditions):
        print(f"    {cond_name}: {roi_mean_betas[cond_idx]:.4f} ± {roi_std_betas[cond_idx]:.4f}")
    
    # Plot
    ax = axes[roi_idx]
    x_pos = np.arange(len(conditions))
    bars = ax.bar(x_pos, roi_mean_betas, yerr=roi_std_betas, capsize=5, 
                   color=['red', 'green', 'blue'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Average Beta Value', fontsize=11)
    ax.set_title(f'{roi_name}\n(Voxels {voxel_list[0]+1}-{voxel_list[-1]+1})', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('Part3i_roi_analysis.png', dpi=150)
plt.close()
print("\nROI analysis plot saved as 'Part3i_roi_analysis.png'")

# =============================================================================
# PART 3j - Statistical Comparison: Faces vs Objects in ROI 1
# =============================================================================

print("\n" + "-" * 80)
print("PART 3j - ROI 1: Faces vs Objects Comparison")
print("-" * 80)

roi1_voxels = list(roi_definitions['ROI 1'])
roi1_betas = beta_matrix[roi1_voxels, :]

# Extract Faces and Objects betas
faces_betas = roi1_betas[:, 0]
objects_betas = roi1_betas[:, 2]

# Calculate differences
beta_differences = faces_betas - objects_betas

print(f"\nROI 1 (Voxels 1-5): Faces vs Objects")
print(f"{'Voxel':<8} {'Faces':<12} {'Objects':<12} {'Difference (F-O)':<18}")
print("-" * 50)
for i, voxel_idx in enumerate(roi1_voxels):
    print(f"{voxel_idx+1:<8} {faces_betas[i]:>11.4f} {objects_betas[i]:>11.4f} {beta_differences[i]:>17.4f}")

# Summary statistics
mean_difference = np.mean(beta_differences)
std_difference = np.std(beta_differences)

print(f"\nSummary Statistics for Faces vs Objects in ROI 1:")
print(f"  Mean difference (F-O): {mean_difference:.4f}")
print(f"  Std Dev of differences: {std_difference:.4f}")
print(f"  Min difference: {np.min(beta_differences):.4f}")
print(f"  Max difference: {np.max(beta_differences):.4f}")

