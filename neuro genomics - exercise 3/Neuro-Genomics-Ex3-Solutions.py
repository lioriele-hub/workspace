"""
=============================================================================
Neuro-Genomics Exercise 3 - fMRI Signal Analysis with GLM
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

# Load the HRF for Question 2
hrf_data = sio.loadmat('hrf_Q2.mat')
hrf = hrf_data['hrf'].flatten()

print(f"\nHRF loaded: {len(hrf)} time points")
print(f"HRF values:\n{hrf}")

# Plot the HRF
fig, ax = plt.subplots(figsize=(10, 6))
time_hrf = np.arange(len(hrf))  # Assuming 1 second TR
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
print(f"  Peak time: {np.argmax(hrf)} seconds")
print(f"  Minimum (undershoot): {np.min(hrf):.4f}")
print(f"  Undershoot time: {np.argmin(hrf)} seconds")


# =============================================================================
# PART 2 - Convolution of Stimulus with HRF
# =============================================================================

print("\n" + "=" * 80)
print("PART 2 - Convolution of Stimulus with HRF")
print("=" * 80)

# Define experimental parameters
TR = 1  # Repetition time in seconds
total_time = 60  # Total experiment duration in seconds
block_duration = 10  # Duration of each stimulus block in seconds

# Create time vector
time = np.arange(0, total_time, TR)
n_timepoints = len(time)
print(f"\nExperiment duration: {total_time} seconds")
print(f"TR: {TR} second(s)")
print(f"Number of time points: {n_timepoints}")

# -----------------------------------------------------------------------------
# Create boxcar stimulus function
# Pattern: 10s ON, 10s OFF, repeating
# -----------------------------------------------------------------------------
stimulus = np.zeros(n_timepoints)
for i, t in enumerate(time):
    # Determine which 20-second cycle we're in
    cycle_position = t % (2 * block_duration)
    if cycle_position < block_duration:
        stimulus[i] = 1  # Stimulus ON

print(f"\nBoxcar stimulus created:")
print(f"  Block duration: {block_duration} seconds (ON)")
print(f"  Rest duration: {block_duration} seconds (OFF)")
print(f"  Number of stimulus blocks: {int(total_time / (2 * block_duration))}")

# -----------------------------------------------------------------------------
# Convolve stimulus with HRF
# -----------------------------------------------------------------------------
predicted_signal = convolve(stimulus, hrf, mode='full')[:n_timepoints]

# Normalize the predicted signal for visualization
predicted_signal_normalized = (predicted_signal - np.mean(predicted_signal)) / np.std(predicted_signal)

print(f"\nConvolution performed:")
print(f"  Input signal length: {len(stimulus)}")
print(f"  HRF length: {len(hrf)}")
print(f"  Output signal length (after truncation): {len(predicted_signal)}")

# -----------------------------------------------------------------------------
# Plot stimulus and predicted fMRI signal
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Plot 1: Stimulus function (boxcar)
ax1 = axes[0]
ax1.fill_between(time, 0, stimulus, alpha=0.6, color='blue', step='pre')
ax1.plot(time, stimulus, 'b-', linewidth=1, drawstyle='steps-pre')
ax1.set_ylabel('Stimulus', fontsize=12)
ax1.set_title('Boxcar Stimulus Function (10s ON / 10s OFF)', fontsize=13)
ax1.set_ylim(-0.1, 1.3)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['OFF', 'ON'])
ax1.grid(True, alpha=0.3)

# Plot 2: HRF (as reference)
ax2 = axes[1]
hrf_time = np.arange(len(hrf))
ax2.plot(hrf_time, hrf, 'g-o', linewidth=2, markersize=4)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_ylabel('HRF Amplitude', fontsize=12)
ax2.set_title('Hemodynamic Response Function (HRF)', fontsize=13)
ax2.set_xlim(0, 20)
ax2.grid(True, alpha=0.3)

# Plot 3: Predicted fMRI signal
ax3 = axes[2]
ax3.plot(time, predicted_signal, 'r-', linewidth=2, label='Predicted fMRI signal')
ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax3.set_xlabel('Time (seconds)', fontsize=12)
ax3.set_ylabel('Signal Amplitude', fontsize=12)
ax3.set_title('Predicted fMRI Signal (Stimulus ⊗ HRF)', fontsize=13)
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Part2_convolution.png', dpi=150)
plt.close()
print("Convolution plot saved as 'Part2_convolution.png'")

# -----------------------------------------------------------------------------
# Combined plot showing stimulus and predicted signal overlay
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))

# Scale stimulus for visualization
stimulus_scaled = stimulus * np.max(predicted_signal) * 0.8

ax.fill_between(time, 0, stimulus_scaled, alpha=0.3, color='blue', 
                step='pre', label='Stimulus (scaled)')
ax.plot(time, predicted_signal, 'r-', linewidth=2.5, label='Predicted fMRI signal')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Signal Amplitude', fontsize=12)
ax.set_title('Stimulus and Predicted fMRI Response Overlay', fontsize=14)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Part2_stimulus_response_overlay.png', dpi=150)
plt.close()
print("Overlay plot saved as 'Part2_stimulus_response_overlay.png'")


# =============================================================================
# PART 3 - GLM Analysis of FAO fMRI Data
# =============================================================================

print("\n" + "=" * 80)
print("PART 3 - GLM Analysis of FAO fMRI Data")
print("=" * 80)

# Load the FAO data
fmri_data = sio.loadmat('fmri_data_FAO.mat')
data = fmri_data['data']  # Shape: (20 voxels, 164 time points)

# Load the HRF for FAO analysis
hrf_fao_data = sio.loadmat('hrf_FAO_Q3.mat')
hrf_fao = hrf_fao_data['hrf'].flatten()

n_voxels, n_timepoints_fao = data.shape
print(f"\nFAO fMRI Data loaded:")
print(f"  Number of voxels: {n_voxels}")
print(f"  Number of time points: {n_timepoints_fao}")
print(f"  HRF length (FAO): {len(hrf_fao)} points")

# Define experimental parameters for FAO
TR_fao = 2  # TR for FAO experiment (seconds)
time_fao = np.arange(n_timepoints_fao) * TR_fao
total_time_fao = n_timepoints_fao * TR_fao
print(f"  TR: {TR_fao} seconds")
print(f"  Total duration: {total_time_fao} seconds ({total_time_fao/60:.1f} minutes)")

# -----------------------------------------------------------------------------
# Create stimulus function for FAO experiment
# Typical FAO (Famous, Anonymous, Objects) paradigm: block design
# Assuming 16s blocks with 16s rest periods
# -----------------------------------------------------------------------------
block_duration_fao = 16  # seconds
rest_duration_fao = 16  # seconds
cycle_duration_fao = block_duration_fao + rest_duration_fao

# Create stimulus in TR units
stimulus_fao = np.zeros(n_timepoints_fao)
for i in range(n_timepoints_fao):
    t = i * TR_fao  # Time in seconds
    cycle_position = t % cycle_duration_fao
    if cycle_position < block_duration_fao:
        stimulus_fao[i] = 1

print(f"\nFAO Stimulus parameters:")
print(f"  Block duration: {block_duration_fao} seconds")
print(f"  Rest duration: {rest_duration_fao} seconds")
print(f"  Cycle duration: {cycle_duration_fao} seconds")

# -----------------------------------------------------------------------------
# Convolve FAO stimulus with HRF
# -----------------------------------------------------------------------------
predicted_fao = convolve(stimulus_fao, hrf_fao, mode='full')[:n_timepoints_fao]

# Normalize for GLM (z-score)
predicted_fao_z = (predicted_fao - np.mean(predicted_fao)) / np.std(predicted_fao)

print(f"\nPredicted signal created via convolution")

# -----------------------------------------------------------------------------
# Compute correlation between predicted signal and each voxel
# -----------------------------------------------------------------------------
correlations = []
for voxel_idx in range(n_voxels):
    voxel_data = data[voxel_idx, :]
    # Z-score normalize voxel data
    voxel_z = (voxel_data - np.mean(voxel_data)) / np.std(voxel_data)
    # Compute Pearson correlation
    r, p = pearsonr(predicted_fao_z, voxel_z)
    correlations.append({'voxel': voxel_idx + 1, 'r': r, 'p': p})

# Sort by correlation strength
correlations_sorted = sorted(correlations, key=lambda x: abs(x['r']), reverse=True)

print("\n" + "-" * 60)
print("Correlation between predicted signal and voxel responses:")
print("-" * 60)
print(f"{'Voxel':<8} {'Correlation (r)':<18} {'p-value':<12}")
print("-" * 60)
for item in correlations_sorted:
    print(f"{item['voxel']:<8} {item['r']:>12.4f}      {item['p']:>12.2e}")

# -----------------------------------------------------------------------------
# Identify voxels with significant activation
# -----------------------------------------------------------------------------
alpha = 0.05
significant_voxels = [c for c in correlations if c['p'] < alpha]
print(f"\nVoxels with significant correlation (p < {alpha}):")
print(f"  Count: {len(significant_voxels)} out of {n_voxels}")
if significant_voxels:
    print(f"  Voxels: {[c['voxel'] for c in significant_voxels]}")

# -----------------------------------------------------------------------------
# Plot FAO HRF
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
time_hrf_fao = np.arange(len(hrf_fao)) * TR_fao
ax.plot(time_hrf_fao, hrf_fao, 'b-o', linewidth=2, markersize=6)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('HRF Amplitude', fontsize=12)
ax.set_title('Hemodynamic Response Function (HRF) for FAO Experiment', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Part3_HRF_FAO.png', dpi=150)
plt.close()
print("\nFAO HRF plot saved as 'Part3_HRF_FAO.png'")

# -----------------------------------------------------------------------------
# Plot stimulus and predicted signal for FAO
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Stimulus
ax1 = axes[0]
ax1.fill_between(time_fao, 0, stimulus_fao, alpha=0.6, color='blue', step='pre')
ax1.plot(time_fao, stimulus_fao, 'b-', linewidth=1, drawstyle='steps-pre')
ax1.set_ylabel('Stimulus', fontsize=12)
ax1.set_title('FAO Stimulus Function (Block Design)', fontsize=13)
ax1.set_ylim(-0.1, 1.3)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['Rest', 'Task'])
ax1.grid(True, alpha=0.3)

# Predicted signal
ax2 = axes[1]
ax2.plot(time_fao, predicted_fao, 'r-', linewidth=2)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time (seconds)', fontsize=12)
ax2.set_ylabel('Predicted Signal', fontsize=12)
ax2.set_title('Predicted fMRI Signal (Stimulus ⊗ HRF)', fontsize=13)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Part3_FAO_stimulus_prediction.png', dpi=150)
plt.close()
print("FAO stimulus and prediction plot saved as 'Part3_FAO_stimulus_prediction.png'")

# -----------------------------------------------------------------------------
# Plot example voxel responses
# -----------------------------------------------------------------------------
# Select best correlated voxels for visualization
best_voxel = correlations_sorted[0]['voxel'] - 1  # Convert to 0-indexed
worst_voxel = correlations_sorted[-1]['voxel'] - 1

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Best voxel - time series
ax1 = axes[0, 0]
voxel_z_best = (data[best_voxel, :] - np.mean(data[best_voxel, :])) / np.std(data[best_voxel, :])
ax1.plot(time_fao, voxel_z_best, 'b-', linewidth=1.5, alpha=0.7, label=f'Voxel {best_voxel+1}')
ax1.plot(time_fao, predicted_fao_z, 'r-', linewidth=2, label='Predicted signal')
ax1.set_xlabel('Time (seconds)', fontsize=11)
ax1.set_ylabel('Z-scored Signal', fontsize=11)
ax1.set_title(f'Best Correlated Voxel ({best_voxel+1}): r = {correlations_sorted[0]["r"]:.3f}', fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Best voxel - scatter
ax2 = axes[0, 1]
ax2.scatter(predicted_fao_z, voxel_z_best, alpha=0.5, s=30, c='blue')
# Add regression line
z = np.polyfit(predicted_fao_z, voxel_z_best, 1)
p = np.poly1d(z)
x_line = np.linspace(predicted_fao_z.min(), predicted_fao_z.max(), 100)
ax2.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'r = {correlations_sorted[0]["r"]:.3f}')
ax2.set_xlabel('Predicted Signal (z-scored)', fontsize=11)
ax2.set_ylabel(f'Voxel {best_voxel+1} Signal (z-scored)', fontsize=11)
ax2.set_title(f'Scatter: Best Correlated Voxel', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Worst voxel - time series
ax3 = axes[1, 0]
voxel_z_worst = (data[worst_voxel, :] - np.mean(data[worst_voxel, :])) / np.std(data[worst_voxel, :])
ax3.plot(time_fao, voxel_z_worst, 'b-', linewidth=1.5, alpha=0.7, label=f'Voxel {worst_voxel+1}')
ax3.plot(time_fao, predicted_fao_z, 'r-', linewidth=2, label='Predicted signal')
ax3.set_xlabel('Time (seconds)', fontsize=11)
ax3.set_ylabel('Z-scored Signal', fontsize=11)
ax3.set_title(f'Least Correlated Voxel ({worst_voxel+1}): r = {correlations_sorted[-1]["r"]:.3f}', fontsize=12)
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Worst voxel - scatter
ax4 = axes[1, 1]
ax4.scatter(predicted_fao_z, voxel_z_worst, alpha=0.5, s=30, c='blue')
z2 = np.polyfit(predicted_fao_z, voxel_z_worst, 1)
p2 = np.poly1d(z2)
ax4.plot(x_line, p2(x_line), 'r-', linewidth=2, label=f'r = {correlations_sorted[-1]["r"]:.3f}')
ax4.set_xlabel('Predicted Signal (z-scored)', fontsize=11)
ax4.set_ylabel(f'Voxel {worst_voxel+1} Signal (z-scored)', fontsize=11)
ax4.set_title(f'Scatter: Least Correlated Voxel', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('Comparison of Best and Least Correlated Voxels with Predicted Signal', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Part3_voxel_comparison.png', dpi=150)
plt.close()
print("Voxel comparison plot saved as 'Part3_voxel_comparison.png'")

# -----------------------------------------------------------------------------
# Plot all voxel correlations as bar chart
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
voxel_nums = [c['voxel'] for c in correlations]
r_values = [c['r'] for c in correlations]
colors = ['green' if c['p'] < 0.05 else 'gray' for c in correlations]

bars = ax.bar(voxel_nums, r_values, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Voxel Number', fontsize=12)
ax.set_ylabel('Correlation (r)', fontsize=12)
ax.set_title('Correlation Between Predicted Signal and Voxel Responses', fontsize=14)
ax.set_xticks(voxel_nums)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='green', alpha=0.7, edgecolor='black', label='Significant (p < 0.05)'),
                   Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='Not significant')]
ax.legend(handles=legend_elements, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Part3_correlation_bar.png', dpi=150)
plt.close()
print("Correlation bar chart saved as 'Part3_correlation_bar.png'")

# -----------------------------------------------------------------------------
# GLM Analysis: Fit linear model for each voxel
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GLM Analysis Results")
print("=" * 80)

# Create design matrix for GLM
# X = [predicted_signal, constant]
X = np.column_stack([predicted_fao, np.ones(n_timepoints_fao)])

print("\nDesign Matrix shape:", X.shape)
print("  Column 1: Predicted signal (convolved stimulus)")
print("  Column 2: Intercept (constant)")

# Fit GLM for each voxel using ordinary least squares
beta_values = []
residuals_all = []
r_squared_values = []

for voxel_idx in range(n_voxels):
    y = data[voxel_idx, :]
    
    # OLS: beta = (X'X)^(-1) X'y
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    
    # Predicted values
    y_pred = X @ beta
    
    # Residuals
    residuals = y - y_pred
    
    # R-squared
    SS_res = np.sum(residuals ** 2)
    SS_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - SS_res / SS_tot
    
    beta_values.append({'voxel': voxel_idx + 1, 'beta_stimulus': beta[0], 'beta_intercept': beta[1]})
    residuals_all.append(residuals)
    r_squared_values.append({'voxel': voxel_idx + 1, 'R2': R2})

# Sort by beta (stimulus coefficient)
beta_sorted = sorted(beta_values, key=lambda x: abs(x['beta_stimulus']), reverse=True)

print("\n" + "-" * 60)
print("GLM Beta Coefficients (sorted by |beta_stimulus|):")
print("-" * 60)
print(f"{'Voxel':<8} {'Beta (Stimulus)':<18} {'Beta (Intercept)':<18} {'R²':<10}")
print("-" * 60)
for item in beta_sorted[:10]:
    voxel_idx = item['voxel'] - 1
    r2 = r_squared_values[voxel_idx]['R2']
    print(f"{item['voxel']:<8} {item['beta_stimulus']:>12.4f}      {item['beta_intercept']:>12.2f}       {r2:>8.4f}")

# -----------------------------------------------------------------------------
# Plot beta values across voxels
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Beta stimulus values
ax1 = axes[0]
voxel_nums = [b['voxel'] for b in beta_values]
beta_stim = [b['beta_stimulus'] for b in beta_values]
colors_beta = ['red' if b > 0 else 'blue' for b in beta_stim]
ax1.bar(voxel_nums, beta_stim, color=colors_beta, alpha=0.7, edgecolor='black')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_xlabel('Voxel Number', fontsize=12)
ax1.set_ylabel('Beta (Stimulus)', fontsize=12)
ax1.set_title('GLM: Beta Coefficients for Stimulus Regressor', fontsize=13)
ax1.set_xticks(voxel_nums)
ax1.grid(True, alpha=0.3, axis='y')

# R-squared values
ax2 = axes[1]
r2_vals = [r['R2'] for r in r_squared_values]
ax2.bar(voxel_nums, r2_vals, color='purple', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Voxel Number', fontsize=12)
ax2.set_ylabel('R² (Variance Explained)', fontsize=12)
ax2.set_title('GLM: R² Values Across Voxels', fontsize=13)
ax2.set_xticks(voxel_nums)
ax2.set_ylim(0, max(r2_vals) * 1.1)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Part3_GLM_results.png', dpi=150)
plt.close()
print("\nGLM results plot saved as 'Part3_GLM_results.png'")

# -----------------------------------------------------------------------------
# Plot time series of all voxels with predicted signal
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 8))

# Plot predicted signal (highlighted)
ax.plot(time_fao, predicted_fao_z * 3, 'r-', linewidth=3, label='Predicted signal (scaled)', zorder=10)

# Plot all voxel time series (normalized and offset for visibility)
for voxel_idx in range(n_voxels):
    voxel_z = (data[voxel_idx, :] - np.mean(data[voxel_idx, :])) / np.std(data[voxel_idx, :])
    offset = (voxel_idx + 1) * 4
    ax.plot(time_fao, voxel_z + offset, 'b-', linewidth=0.8, alpha=0.5)
    ax.text(time_fao[-1] + 5, offset, f'V{voxel_idx+1}', fontsize=8, va='center')

ax.set_xlabel('Time (seconds)', fontsize=12)
ax.set_ylabel('Z-scored Signal (offset for visibility)', fontsize=12)
ax.set_title('All Voxel Time Series with Predicted Signal', fontsize=14)
ax.legend(loc='upper left')
ax.set_xlim(0, time_fao[-1] + 20)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Part3_all_voxels_timeseries.png', dpi=150)
plt.close()
print("All voxels time series plot saved as 'Part3_all_voxels_timeseries.png'")

# -----------------------------------------------------------------------------
# Summary Statistics
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

mean_r = np.mean([c['r'] for c in correlations])
std_r = np.std([c['r'] for c in correlations])
max_r = max([c['r'] for c in correlations])
min_r = min([c['r'] for c in correlations])

mean_R2 = np.mean(r2_vals)
max_R2 = max(r2_vals)

print(f"\nCorrelation Statistics:")
print(f"  Mean correlation: {mean_r:.4f}")
print(f"  Std correlation:  {std_r:.4f}")
print(f"  Max correlation:  {max_r:.4f}")
print(f"  Min correlation:  {min_r:.4f}")

print(f"\nGLM R² Statistics:")
print(f"  Mean R²: {mean_R2:.4f}")
print(f"  Max R²:  {max_R2:.4f}")

print(f"\nSignificant voxels (p < 0.05): {len(significant_voxels)}/{n_voxels}")

# =============================================================================
# Save results to file
# =============================================================================

with open('Part3_results.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("Neuro-Genomics Exercise 3 - fMRI GLM Analysis Results\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("CORRELATION ANALYSIS\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Voxel':<8} {'Correlation (r)':<18} {'p-value':<15} {'Significant':<12}\n")
    f.write("-" * 70 + "\n")
    for c in correlations:
        sig = "Yes" if c['p'] < 0.05 else "No"
        f.write(f"{c['voxel']:<8} {c['r']:>12.4f}       {c['p']:>12.2e}    {sig:<12}\n")
    
    f.write("\n\nGLM BETA COEFFICIENTS\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Voxel':<8} {'Beta (Stimulus)':<18} {'Beta (Intercept)':<18} {'R²':<10}\n")
    f.write("-" * 70 + "\n")
    for b in beta_values:
        voxel_idx = b['voxel'] - 1
        r2 = r_squared_values[voxel_idx]['R2']
        f.write(f"{b['voxel']:<8} {b['beta_stimulus']:>12.4f}       {b['beta_intercept']:>12.2f}        {r2:>8.4f}\n")
    
    f.write("\n\nSUMMARY STATISTICS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Mean correlation: {mean_r:.4f}\n")
    f.write(f"Std correlation:  {std_r:.4f}\n")
    f.write(f"Max correlation:  {max_r:.4f}\n")
    f.write(f"Min correlation:  {min_r:.4f}\n")
    f.write(f"Mean R²: {mean_R2:.4f}\n")
    f.write(f"Max R²:  {max_R2:.4f}\n")
    f.write(f"Significant voxels (p < 0.05): {len(significant_voxels)}/{n_voxels}\n")

print("\nResults saved to 'Part3_results.txt'")
print("\n" + "=" * 80)
print("Exercise 3 Complete!")
print("=" * 80)
