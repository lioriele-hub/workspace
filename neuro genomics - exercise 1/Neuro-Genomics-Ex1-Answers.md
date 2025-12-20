# Neuro-Genomics Exercise 1 - Written Answers

## Part 1 - General Introduction to Sequencing Data

### Question: Why can we only get an estimation of expression levels and not the actual number of RNA molecules for each gene?

**Answer:**

We can only get an **estimation** of expression levels (not actual RNA molecule counts) due to several sources of distortion during PCR amplification:

1. **PCR Bias**: Different sequences have different amplification efficiencies based on their GC content, secondary structure, and other sequence features. Some sequences are preferentially amplified over others, meaning the final read counts don't proportionally reflect the original RNA amounts.

2. **Stochastic Amplification of Low Copy Number Amplicons**: When starting material is limited (low copy number), PCR amplification introduces random variation. Some molecules may be amplified more than others by chance, leading to distorted representation of the original proportions.

3. **Bridge Amplification Variability**: During Illumina sequencing, bridge amplification creates clusters on the flow cell. The efficiency of cluster formation can vary, affecting read counts.

4. **Library Preparation Losses**: Not all RNA molecules are successfully converted to cDNA and incorporated into the sequencing library.

Therefore, read counts provide a **relative measure** of expression, not absolute molecule counts.

---

## Part 2 - Explore Sequencing Data Using R

### Question: What kind of information is in the cts matrix?

**Answer:**

The matrix contains gene expression count data where:
- **Rows** represent different genes (identified by FlyBase gene IDs like FBgn0000003)
- **Columns** represent different samples (treated1, treated2, treated3, untreated1, etc.)
- **Each cell** contains the number of sequencing reads aligned to that gene in that sample

### Question: Is the sum of reads the same for each sample?

**Answer:**

**No**, the sum of reads is NOT the same for each sample. Different samples have different total read counts due to technical variability in the sequencing process. This is why normalization is necessary for valid comparison between samples.

---

## Part 3 - Basic Statistics of Sequencing Data

### Question: Does the scatter plot of variance vs. mean (log-log) fit a Poisson distribution?

**Answer:**

**No**, the data does NOT follow a Poisson distribution. In a Poisson distribution, the variance equals the mean (the points would lie on the y=x line). Looking at the scatter plot, the variance is consistently **higher** than the mean (points lie above the y=x line).

This phenomenon is called **overdispersion** and is typical of RNA-seq data. The data better fits a **Negative Binomial distribution**, which has an additional dispersion parameter to account for this extra variance. The overdispersion arises from both technical variability and biological variability between samples.

### Question: Can we detect differentially expressed genes with only one sample per condition?

**Answer:**

**No**, we cannot reliably detect differentially expressed genes with only one sample per experimental condition because:

1. We cannot estimate within-group variance (biological + technical noise)
2. We cannot distinguish true biological differences from random variation
3. Statistical tests require replicates to estimate variability
4. Any observed difference could be due to chance, technical artifacts, or biological noise rather than the treatment effect

Replicates are essential for statistical inference in differential expression analysis.

---

## Part 4 - Detecting Differentially Expressed Genes

### Question: What is the meaning of the blue points in the MA plot?

**Answer:**

The blue points represent genes that are **statistically significantly differentially expressed** (adjusted p-value < 0.1 by default in DESeq2). These are genes where the expression change between treated and untreated conditions is unlikely to be due to random chance.

The MA plot shows:
- **X-axis**: Mean of normalized counts (average expression)
- **Y-axis**: Log2 fold change (treated vs untreated)

Blue points are significant, while gray points are not statistically significant.

---

## Part 5 - Detecting Circadian Patterns Using FFT

### Question: What is the time step between measurements?

**Answer:**

The time step is **4 hours** (e.g., 11PM to 3AM = 4 hours).

The time points span 2 days (48 hours total) with measurements at: A_11PM, A_3AM, A_7AM, A_11AM, A_3PM, A_7PM, B_11PM, B_3AM, B_7AM, B_11AM, B_3PM, B_7PM.

### Question: Does the expression of per1a seem circadian? Should it be?

**Answer:**

**Yes**, per1a shows clear circadian expression patterns. **Per1 (Period 1)** is a **core clock gene** that is part of the molecular circadian oscillator. It is one of the best-known circadian genes, showing robust ~24-hour rhythms. The gene is essential for maintaining circadian rhythms in all organisms, so it is expected to show strong circadian oscillation.

### Question: Which experimental design is better for detecting circadian genes?

**(a)** Shorter time step (2 hours) with same 48-hour duration, OR
**(b)** Longer duration (4 days) with same 4-hour time step?

**Answer:**

**Option (b) - increasing the number of days measured (4 days instead of 2)** while keeping the time step the same is BETTER for detecting circadian frequency.

**Reasoning:**

- **Shorter time step (option a)**: This increases the Nyquist frequency (maximum detectable frequency), but doesn't improve resolution at the circadian frequency. We can already detect 24h cycles with 4h sampling, so this doesn't help.

- **Longer measurement period (option b)**: This improves **frequency RESOLUTION** (frequency resolution = 1/(N×Δt)). With more days, we can better distinguish the circadian frequency from nearby frequencies, reducing spectral leakage and providing more precise detection of the 24h period. More cycles of the rhythm also provide more statistical power.

### Question: What are the advantages of detecting circadian genes in the frequency domain versus the time domain?

**Answer:**

1. **Objective quantification**: FFT provides a single numerical value (power at circadian frequency) that can be compared across genes, rather than subjective curve fitting.

2. **Phase-independent**: FFT detects periodicity regardless of phase. A cosine fit would require knowing or estimating the phase of the oscillation.

3. **Noise robustness**: FFT can detect periodic signals even in noisy data by concentrating power at the fundamental frequency.

4. **No assumptions about waveform**: FFT works for any periodic pattern, not just perfect cosines. Real biological rhythms may have non-sinusoidal waveforms.

5. **Computational efficiency**: FFT is O(N log N), making it efficient for analyzing thousands of genes.

6. **Detection of multiple periodicities**: FFT can reveal if a gene has multiple periodic components (e.g., 24h and 12h harmonics).

### Question: Top 10 genes with highest circadian power - known circadian genes?

**Answer:**

Known circadian genes that should appear in the top 10 include:
- **per1a, per1b, per2, per3** (Period genes - core clock components)
- **cry1a, cry1b, cry2** (Cryptochrome genes - core clock components)
- **clock, bmal/arntl** (Clock transcription factors)
- **nr1d1, nr1d2** (Rev-erb genes - clock regulators)

These genes are conserved across species and are essential components of the molecular circadian clock.

---

## Part 6 - Detecting Genes with Variable Expression Levels

### Question: Can you detect known circadian genes among the top 40 variable genes? Is this expected?

**Answer:**

**Yes**, circadian genes are expected to be among the top variable genes, and this is expected because:

1. **By definition**, circadian genes show oscillating expression levels across the 24-hour cycle, which means **high variance** between time points.

2. The **amplitude of circadian oscillations** contributes directly to variance.

3. Therefore, genes with strong circadian rhythms will have **higher-than-expected variance** compared to non-rhythmic genes with similar mean expression.

4. The z-score method specifically identifies genes with **unusually high variance for their expression level**, which describes circadian genes perfectly.

Known circadian genes expected in the top variable genes:
- per1a, per2, per3 (Period genes)
- cry1a, cry1b (Cryptochrome genes)
- bmal, clock (Core clock transcription factors)
- nr1d1 (Rev-erb alpha)

This overlap between "variable genes" and "circadian genes" makes sense because both detection methods are capturing the same underlying biological phenomenon: genes with temporally varying expression patterns.

---

## Summary

This exercise demonstrated key principles of RNA-seq data analysis:

1. **Normalization** is essential for comparing samples with different sequencing depths
2. RNA-seq data is **overdispersed** (variance > mean) and follows a Negative Binomial distribution
3. **Replicates** are necessary for reliable differential expression analysis
4. **FFT** is powerful for detecting periodic gene expression patterns
5. **Variable gene detection** can identify genes with interesting expression dynamics


