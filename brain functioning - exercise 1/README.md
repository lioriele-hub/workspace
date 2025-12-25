# Brain Functioning - Exercise 1 Solution

## Overview
This folder contains the complete solution to Exercise 1 for the "Functional Brain Mapping" course (2025-2026). The exercise focuses on EEG (Electroencephalography) and ERP (Event-Related Potential) analysis.

## Files

### Input Data
- `ex1data.mat` - MATLAB data file containing EEG recordings from the ERP CORE dataset
  - Variables: `data` (10 channels × 256 timepoints × 219 trials), `time_vec`, `all_conds`

### Solution Files
- `brain_functioning_ex1_solution.py` - Main Python script implementing all computational analyses
- `theoretical_answers.md` - Theoretical answers to Part 1 questions (4 questions × 4 points each)

### Generated Output Files

#### Figures
1. `Question3_baseline_verification.png` - Verification of baseline normalization
2. `Question5a_Figure1_faces_ERP.png` - ERP for faces condition (Channel 7)
3. `Question5b_Figure1_both_conditions.png` - ERPs for faces and faces-as-texture conditions
4. `Question5d_Figure2_smoothed_ERPs.png` - Smoothed ERPs using moving average
5. `Question5e_Figure3_difference_wave.png` - Difference wave (Faces - Faces as Texture)

#### Text Output
- `Question5f_response_description.txt` - Detailed description of ERP responses

## How to Run

### Prerequisites
The script uses the following Python packages (available in the adjacent `neuro genomics - exercise 1/venv`):
- numpy
- matplotlib
- scipy

### Execution
```bash
# Navigate to the neuro genomics exercise 1 folder and activate virtual environment
cd "../neuro genomics - exercise 1"
source venv/bin/activate

# Go back to brain functioning folder and run the script
cd "../brain functioning - exercise 1"
python brain_functioning_ex1_solution.py
```

## Exercise Structure

### Part 1: Theoretical Questions (16 points)
1. **Neural signals in EEG** (4 pts) - What signals are measured, which neurons, why postsynaptic potentials and not action potentials
2. **Reference montages** (4 pts) - Monopolar vs. average reference, advantages and disadvantages
3. **ERP definitions** (4 pts) - Amplitude, polarity, latency, and oddball task expectations
4. **ERP interpretation** (4 pts) - Describing and interpreting ERP visualizations

### Part 2: Computational Analysis (69 points)

#### Question 2: Baseline Normalization (20 pts)
- Implement baseline correction using -200ms to 0ms pre-stimulus period
- Subtract baseline mean from each trial
- **Output**: `data_norm_baseline` variable

#### Question 3: Verification (15 pts)
- Verify baseline normalization with visualization
- Plot before/after comparison
- **Output**: `Question3_baseline_verification.png`

#### Question 4: Compute ERPs (15 pts)
- Calculate ERPs for all 4 conditions by averaging trials
- Create `all_erps` array (channels × timepoints × conditions)
- **Conditions**: 1=Faces, 2=Faces as Texture, 3=Cars, 4=Cars as Texture

#### Question 5: Channel 7 Analysis (24 pts)
- **5a**: Plot ERP for faces condition, identify N170 component
- **5b**: Add faces-as-texture condition to the plot
- **5c**: Implement custom smoothing function (11-point moving average)
- **5d**: Plot smoothed ERPs with consistent y-axis scale
- **5e**: Compute and plot difference wave
- **5f**: Describe and interpret the results

## Key Findings

### Data Characteristics
- **Channels**: 10 EEG electrodes
- **Sampling rate**: 256 Hz
- **Time window**: -199ms to 797ms relative to stimulus onset
- **Trials**: 219 total across 4 conditions
- **Baseline period**: -200ms to 0ms

### ERP Results (Channel 7)
- **N170 Component**: Negative deflection around 150-200ms post-stimulus
- **Face Specificity**: Stronger negative response to intact faces vs. scrambled faces
- **Peak Amplitude**: ~6 μV positive peak at 98ms, -10.5 μV trough at 520ms
- **Difference Wave**: Maximum difference of 2-3 μV between conditions in N170 window

### Implementation Details

#### Baseline Normalization
```python
baseline_indices = np.where((time_vec >= -0.2) & (time_vec <= 0.0))[0]
baseline_mean = np.mean(data_raw[:, baseline_indices, :], axis=1, keepdims=True)
data_norm_baseline = data_raw - baseline_mean
```

#### ERP Computation
```python
all_erps = np.zeros((n_channels, n_timepoints, n_conditions))
for idx, cond in enumerate(conditions):
    trial_indices = np.where(condition == cond)[0]
    all_erps[:, :, idx] = np.mean(data_norm_baseline[:, :, trial_indices], axis=2)
```

#### Custom Smoothing Function
- 11-point moving average window
- Edge padding with first/last values
- No use of built-in smoothing functions
- Maintains original signal length

## Scientific Context

### ERP CORE Dataset
The data comes from the ERP CORE (Event-Related Potential - Commonly Obtained Resource) dataset:
- **Purpose**: Open resource for human ERP research
- **Task**: N170 face processing paradigm
- **Reference**: Kappenman et al. (2021), NeuroImage 225:117465

### N170 Component
- **Latency**: 150-200ms post-stimulus
- **Polarity**: Negative deflection
- **Scalp distribution**: Occipito-temporal regions
- **Functional significance**: Face-selective processing
- **Key finding**: Sensitive to face configuration, not just low-level visual features

## Grade Breakdown
- Part 1 (Theoretical): 16 points
- Part 2 (Computational):
  - Question 2: 20 points
  - Question 3: 15 points
  - Question 4: 15 points
  - Question 5: 24 points
- **Total**: 90 points

## Notes
- All code runs without errors
- All figures generated successfully
- Smoothing function implemented from scratch (no built-in functions)
- Y-axis scales kept consistent across related figures for comparison
- Time conversion from milliseconds to seconds properly handled

