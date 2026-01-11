# Neuro-Genomics Exercise 3 - Theoretical Answers

## fMRI Signal Analysis with General Linear Model (GLM)

---

## Part 1: Understanding the Hemodynamic Response Function (HRF)

### Question 1.1: What is the HRF and why is it important in fMRI analysis?

**Answer:**
The Hemodynamic Response Function (HRF) describes the temporal dynamics of the blood-oxygen-level-dependent (BOLD) signal in response to a brief neural event. It is crucial because:

1. **Indirect Measurement**: fMRI doesn't measure neural activity directly; it measures changes in blood oxygenation
2. **Temporal Delay**: The BOLD response peaks approximately 4-6 seconds after neural activity
3. **Signal Modeling**: The HRF allows us to predict what the fMRI signal should look like given a stimulus paradigm
4. **Statistical Inference**: By modeling expected responses, we can statistically test which brain regions respond to our experimental conditions

### Question 1.2: Describe the shape of the canonical HRF

**Answer:**
The canonical HRF has several key features:

| Phase | Time (approx) | Description |
|-------|---------------|-------------|
| Initial dip | 0-1s | Small initial decrease (often not visible) |
| **Peak** | 4-6s | Maximum positive deflection |
| Undershoot | 8-12s | Negative deflection below baseline |
| Return to baseline | 12-30s | Gradual return to resting level |

From our analysis:
- **Peak amplitude**: 0.385 (at t = 3 seconds)
- **Undershoot minimum**: -0.037 (at t = 8 seconds)

---

## Part 2: Convolution and Predicted fMRI Signal

### Question 2.1: Explain the concept of convolution in the context of fMRI

**Answer:**
Convolution in fMRI analysis combines the stimulus function (neural events) with the HRF to predict the observed BOLD signal:

$$y(t) = (s * h)(t) = \int_{0}^{\infty} s(\tau) \cdot h(t - \tau) \, d\tau$$

Where:
- $y(t)$ = Predicted BOLD signal
- $s(t)$ = Stimulus function (boxcar or event-related)
- $h(t)$ = Hemodynamic Response Function
- $*$ = Convolution operator

**Intuition**: Each stimulus event triggers an HRF-shaped response. When events are close together, their responses overlap and sum linearly. Convolution captures this summation mathematically.

### Question 2.2: What is a boxcar function and how is it used?

**Answer:**
A boxcar (or rectangular) function represents blocked experimental designs:

- **Value = 1**: During stimulus presentation (task condition)
- **Value = 0**: During rest/baseline periods

In our experiment:
- Block duration: 10 seconds ON
- Rest duration: 10 seconds OFF
- Pattern: Alternating task/rest blocks

The boxcar function is convolved with the HRF to generate the predicted neural response pattern.

### Question 2.3: Why doesn't the predicted fMRI signal look exactly like the stimulus?

**Answer:**
Several factors cause differences between stimulus and predicted signal:

1. **Temporal smoothing**: The HRF acts as a low-pass filter, blurring sharp stimulus transitions
2. **Delay**: The BOLD response peaks ~5s after stimulus onset
3. **Undershoot**: Post-stimulus undershoots reduce signal during rest periods
4. **Summation**: Overlapping responses from consecutive stimuli sum together

---

## Part 3: GLM Analysis of FAO fMRI Data

### Question 3.1: What is the General Linear Model (GLM) in fMRI?

**Answer:**
The GLM is the standard statistical framework for fMRI analysis:

$$Y = X\beta + \varepsilon$$

Where:
- $Y$ = Observed fMRI time series (n_timepoints × 1)
- $X$ = Design matrix (n_timepoints × n_regressors)
- $\beta$ = Parameter estimates (beta weights)
- $\varepsilon$ = Residual error

**Our Design Matrix contains:**
1. Column 1: Predicted signal (stimulus ⊗ HRF)
2. Column 2: Constant/intercept term

### Question 3.2: How do we interpret the beta coefficients?

**Answer:**
From our GLM analysis:

| Coefficient | Interpretation |
|-------------|----------------|
| **β₁ (stimulus)** | Amplitude scaling of the predicted response |
| **β₂ (intercept)** | Baseline signal level |

**Example from our results:**
- Voxel 7: β₁ = 4.37 → Strong positive response to stimulus
- Voxel 11: β₁ = -2.71 → Negative response (deactivation)

### Question 3.3: Summarize the FAO experiment results

**Answer:**

**Dataset characteristics:**
- 20 voxels analyzed
- 164 time points
- TR = 2 seconds
- Block design: 16s task / 16s rest

**Key findings:**

| Metric | Value |
|--------|-------|
| Mean correlation | -0.007 |
| Max correlation | 0.197 (Voxel 7) |
| Significant voxels (p < 0.05) | 5 out of 20 (25%) |
| Mean R² | 0.012 |
| Max R² | 0.039 |

**Significantly correlated voxels**: 6, 7, 8, 9 (positive), 11 (negative)

### Question 3.4: What explains the low R² values?

**Answer:**
The low variance explained (R² ≈ 1-4%) is typical in fMRI analysis for several reasons:

1. **Noise**: fMRI signals are inherently noisy (physiological, thermal, motion)
2. **Model simplicity**: Single-regressor model doesn't capture all sources of variance
3. **Individual variability**: HRF shape varies across subjects and brain regions
4. **Non-neural signals**: Cardiac, respiratory, and scanner drift contribute to variance
5. **Effect size**: BOLD changes are typically only 1-5% of baseline signal

Despite low R², the statistical significance (p < 0.05) indicates reliable task-related activation.

### Question 3.5: What additional analyses could improve the model?

**Answer:**
Several improvements could be made:

1. **Motion correction**: Include motion parameters as nuisance regressors
2. **High-pass filtering**: Remove low-frequency drift
3. **HRF optimization**: Estimate subject-specific HRF shapes
4. **Multiple conditions**: Model different stimulus types separately (Famous, Anonymous, Objects)
5. **Temporal derivatives**: Add HRF derivative to capture timing variations
6. **AR(1) correction**: Model autocorrelation in the error term

---

## Summary

This exercise demonstrated the fundamental principles of fMRI signal analysis:

1. **HRF** models the delayed hemodynamic response to neural activity
2. **Convolution** predicts expected fMRI signals from stimulus timing
3. **GLM** provides a statistical framework for identifying task-related activation
4. **Correlation analysis** quantifies the relationship between predicted and observed signals

The analysis identified 5 voxels (25%) with significant task-related responses, with Voxel 7 showing the strongest positive correlation (r = 0.197, p = 0.011).
