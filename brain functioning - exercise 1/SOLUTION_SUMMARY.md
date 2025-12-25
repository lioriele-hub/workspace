# Brain Functioning Exercise 1 - Solution Summary

## ✅ Exercise Completed Successfully

All parts of the exercise have been solved and tested.

---

## 📁 Files for Submission

### 1. Code File (Required)
**Filename**: `brain_functioning_ex1_solution.py`

**What it does**:
- Loads EEG data from `ex1data.mat`
- Performs baseline normalization (-200ms to 0ms)
- Computes ERPs for all 4 conditions
- Implements custom smoothing function
- Generates all required figures
- Provides detailed console output

**Status**: ✅ Tested and runs without errors

---

### 2. Answers PDF (Required)
**Source file**: `theoretical_answers.md`

**Content**:
- Question 1: What neural signals are measured with EEG? (4 points)
- Question 2: Reference montages - monopolar vs average (4 points)
- Question 3: ERP definitions and oddball task (4 points)
- Question 4: ERP visualization and interpretation (4 points)

**Action needed**: Convert `theoretical_answers.md` to PDF format

**How to convert**:
- Run `./convert_to_pdf.sh` (if pandoc is installed)
- OR use online converter: https://markdown-to-pdf.com
- OR use VS Code with "Markdown PDF" extension
- OR copy to Word/Google Docs and export as PDF

**Before submitting**: Add your name to the PDF!

---

## 📊 Generated Outputs

### Variables Created (in Python script)
- ✅ `data_norm_baseline` - shape (10, 256, 219) - baseline-corrected EEG data
- ✅ `all_erps` - shape (10, 256, 4) - ERPs for all 4 conditions
- ✅ `my_smooth(signal)` - custom smoothing function

### Figures Generated
1. ✅ `Question3_baseline_verification.png` - Shows before/after baseline normalization
2. ✅ `Question5a_Figure1_faces_ERP.png` - ERP for faces (Channel 7)
3. ✅ `Question5b_Figure1_both_conditions.png` - Faces + Faces-as-Texture ERPs
4. ✅ `Question5d_Figure2_smoothed_ERPs.png` - Smoothed versions
5. ✅ `Question5e_Figure3_difference_wave.png` - Difference (Faces - Faces as Texture)

### Text Output
- ✅ `Question5f_response_description.txt` - Interpretation of results

---

## 🔬 Key Results

### Data Overview
- **10 channels** of EEG data
- **256 Hz** sampling rate
- **219 trials** across 4 conditions
- **Time window**: -199ms to 797ms (1 second total)

### Conditions
1. Faces (57 trials)
2. Faces as Texture (59 trials)
3. Cars (54 trials)
4. Cars as Texture (49 trials)

### Main Findings (Channel 7)
- **N170 component** identified around 150-200ms post-stimulus
- **Face selectivity**: Stronger negative response to intact faces vs. scrambled
- **Peak amplitude**: ~6 μV positive, -10.5 μV negative
- **Difference wave**: 2-3 μV maximum difference in N170 window

---

## 🎯 Question Mapping

### Question 2 (20 points): Baseline Normalization
**Implementation**:
```python
# Lines 59-77 in brain_functioning_ex1_solution.py
baseline_indices = np.where((time_vec >= -0.2) & (time_vec <= 0.0))[0]
baseline_mean = np.mean(data_raw[:, baseline_indices, :], axis=1, keepdims=True)
data_norm_baseline = data_raw - baseline_mean
```

**Output**: Variable `data_norm_baseline` with shape (10, 256, 219)

---

### Question 3 (15 points): Verification
**Implementation**: Lines 82-127

**Output**: 
- Figure showing before/after baseline normalization
- Console output confirming baseline mean ≈ 0 after normalization

---

### Question 4 (15 points): Compute ERPs
**Implementation**: Lines 132-170

**Output**: Variable `all_erps` with shape (10, 256, 4)
- Dimension 1: 10 channels
- Dimension 2: 256 time points
- Dimension 3: 4 conditions

---

### Question 5 (24 points): Channel 7 Analysis

#### 5a (Lines 183-228): Plot Faces ERP
- ✅ Figure 1 created
- ✅ Identified peak, trough, N170 component

#### 5b (Lines 233-249): Add Faces-as-Texture
- ✅ Updated Figure 1 with both conditions

#### 5c (Lines 254-294): Smoothing Function
```python
def my_smooth(signal):
    """11-point moving average with edge padding"""
    window_size = 11
    half_window = window_size // 2
    # ... implementation using manual averaging
    return smoothed_signal
```
- ✅ No built-in smoothing functions used
- ✅ Maintains original signal length

#### 5d (Lines 299-325): Smoothed ERPs
- ✅ Figure 2 created
- ✅ Y-axis scale matches Figure 1

#### 5e (Lines 330-354): Difference Wave
- ✅ Figure 3 created
- ✅ Y-axis scale matches Figures 1 and 2

#### 5f (Lines 359-461): Interpretation
- ✅ Detailed description written
- ✅ Saved to text file

---

## 🧪 How to Test

```bash
# Step 1: Activate virtual environment
cd "../neuro genomics - exercise 1"
source venv/bin/activate

# Step 2: Run the solution
cd "../brain functioning - exercise 1"
python brain_functioning_ex1_solution.py

# Expected: Script completes successfully, generates all 5 figures
```

---

## 📚 Documentation Files

### README.md
Complete technical documentation including:
- File descriptions
- How to run instructions
- Exercise structure
- Key findings
- Implementation details
- Scientific context

### SUBMISSION_GUIDE.md
Step-by-step submission instructions including:
- What to submit
- How to convert markdown to PDF
- Pre-submission checklist
- Testing instructions

### This File (SOLUTION_SUMMARY.md)
Quick reference summary of the complete solution

---

## ⚠️ Important Reminders

1. **Add your name** to the theoretical answers PDF before submitting
2. **Test the code** one final time before submission
3. **Deadline**: December 7, 2025 (end of day)
4. **Submit 2 files** to Moodle:
   - `brain_functioning_ex1_solution.py` (code)
   - `theoretical_answers.pdf` (your PDF version)

---

## ✨ Solution Quality

- ✅ All code runs without errors
- ✅ All requirements met
- ✅ Well-documented and commented
- ✅ Follows scientific best practices
- ✅ Figures are publication-quality
- ✅ Complete theoretical answers

---

## 📊 Points Breakdown

| Section | Points |
|---------|--------|
| **Part 1: Theoretical** | |
| Question 1 (EEG signals) | 4 |
| Question 2 (Reference montages) | 4 |
| Question 3 (ERP definitions) | 4 |
| Question 4 (ERP interpretation) | 4 |
| **Part 2: Computational** | |
| Question 2 (Baseline normalization) | 20 |
| Question 3 (Verification) | 15 |
| Question 4 (Compute ERPs) | 15 |
| Question 5 (Channel 7 analysis) | 24 |
| **Total** | **90** |

---

## 🎓 Good Luck with Your Submission!

All work is complete and ready for submission. Just remember to:
1. Convert theoretical answers to PDF
2. Add your name
3. Test the code one more time
4. Submit both files to Moodle

**You're all set!** 🎉

