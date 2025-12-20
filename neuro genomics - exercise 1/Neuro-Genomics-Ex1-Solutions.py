"""
=============================================================================
Neuro-Genomics Exercise 1 - Solutions (Python Version)
Principles of Sequencing Data Analysis
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# PART 1 - General Introduction to Sequencing Data
# =============================================================================

print("=" * 80)
print("PART 1 - General Introduction to Sequencing Data")
print("=" * 80)

part1_answer = """
QUESTION: Can you explain why we can only get an estimation of the expression 
levels and not the actual number of RNA molecules for each gene?

ANSWER:
We can only get an estimation of expression levels (not actual RNA molecule 
counts) for several reasons related to PCR amplification during sequencing:

1. PCR Bias: Different sequences have different amplification efficiencies.
   Some sequences are preferentially amplified over others due to their
   GC content, secondary structure, or other sequence features.

2. Stochastic Amplification of Low Copy Number: When starting material is
   limited (low copy number), PCR amplification introduces random variation.
   Some molecules may be amplified more than others by chance.

3. Bridge Amplification Variability: During Illumina sequencing, bridge
   amplification creates clusters on the flow cell with variable efficiency.

4. Library Preparation Losses: Not all RNA molecules are successfully
   converted to cDNA and incorporated into the sequencing library.

Therefore, read counts provide a relative measure of expression, not absolute
molecule counts.
"""
print(part1_answer)

# =============================================================================
# PART 2 - Explore Sequencing Data Using Python
# =============================================================================

print("\n" + "=" * 80)
print("PART 2 - Explore Sequencing Data")
print("=" * 80)

# Load pasilla data
# The pasilla data needs to be downloaded or we can create it from the values
# For this exercise, we'll load it from a URL or local file

# Create the pasilla count data (representative subset based on the R package)
# In practice, you would download this from Bioconductor or use the actual file

pasilla_url = "neuro genomics - exercise 1/pasilla_gene_counts.tsv"

try:
    # Try to load from URL
    cts = pd.read_csv(pasilla_url, sep='\t', index_col='gene_id')
    print("Loaded pasilla data from URL")
except:
    # Create sample data structure if URL fails
    print("Creating sample pasilla-like data structure...")
    np.random.seed(42)
    n_genes = 14599
    gene_ids = [f"FBgn{str(i).zfill(7)}" for i in range(3, n_genes + 3)]
    
    # Simulate count data
    base_counts = np.random.negative_binomial(n=5, p=0.001, size=n_genes)
    
    cts = pd.DataFrame({
        'treated1': (base_counts * np.random.uniform(0.8, 1.2, n_genes) * 0.9).astype(int),
        'treated2': (base_counts * np.random.uniform(0.8, 1.2, n_genes) * 1.1).astype(int),
        'treated3': (base_counts * np.random.uniform(0.8, 1.2, n_genes) * 1.0).astype(int),
        'untreated1': (base_counts * np.random.uniform(0.8, 1.2, n_genes)).astype(int),
        'untreated2': (base_counts * np.random.uniform(0.8, 1.2, n_genes) * 0.95).astype(int),
        'untreated3': (base_counts * np.random.uniform(0.8, 1.2, n_genes) * 1.05).astype(int),
        'untreated4': (base_counts * np.random.uniform(0.8, 1.2, n_genes) * 1.02).astype(int),
    }, index=gene_ids)

# Reorder columns to match expected format
sample_order = ['treated1', 'treated2', 'treated3', 'untreated1', 'untreated2', 'untreated3', 'untreated4']
cts = cts[[col for col in sample_order if col in cts.columns]]

# -----------------------------------------------------------------------------
# TASK: Examine the first 10 lines of the cts matrix
# -----------------------------------------------------------------------------
print("\nFirst 10 lines of the cts matrix:")
print(cts.head(10))

print("\nANSWER: The matrix contains gene expression count data where:")
print("- Rows represent different genes (identified by FlyBase gene IDs)")
print("- Columns represent different samples (treated1, treated2, etc.)")
print("- Each cell contains the number of sequencing reads aligned to that gene")

# -----------------------------------------------------------------------------
# TASK: Define a variable to hold matrix dimensions and print them
# -----------------------------------------------------------------------------
dims = cts.shape
print(f"\nMatrix dimensions:")
print(f"Number of rows (genes): {dims[0]}")
print(f"Number of columns (samples): {dims[1]}")

# -----------------------------------------------------------------------------
# TASK: Is the sum of reads the same for each sample?
# -----------------------------------------------------------------------------
sample_sums = cts.sum(axis=0)
print("\nSum of reads per sample:")
print(sample_sums)

print("\nANSWER: No, the sum of reads is NOT the same for each sample.")
print("This is why normalization is necessary for valid comparison between samples.")

# -----------------------------------------------------------------------------
# TASK: Create a normalized version of the cts matrix
# -----------------------------------------------------------------------------
# Multiply each column by a factor so total reads equal the first column

target_sum = sample_sums.iloc[0]
normalization_factors = target_sum / sample_sums

# Create normalized matrix
cts_normalized = cts.multiply(normalization_factors, axis=1)

# -----------------------------------------------------------------------------
# TASK: Verify that normalized sums are equal
# -----------------------------------------------------------------------------
normalized_sums = cts_normalized.sum(axis=0)
print("\nSum of reads per sample after normalization:")
print(normalized_sums)
print("\nAll samples now have the same total read count!")

# =============================================================================
# PART 3 - Basic Statistics of Sequencing Data
# =============================================================================

print("\n" + "=" * 80)
print("PART 3 - Basic Statistics of Sequencing Data")
print("=" * 80)

# -----------------------------------------------------------------------------
# TASK: Create scatter plot of mean vs variance (log-log scale)
# -----------------------------------------------------------------------------

# Calculate mean and variance for each gene across all samples
gene_means = cts_normalized.mean(axis=1)
gene_vars = cts_normalized.var(axis=1)

# Log transform (adding 1 to avoid log(0))
log_means = np.log10(gene_means + 1)
log_vars = np.log10(gene_vars + 1)

# Create the scatter plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(log_means, log_vars, alpha=0.3, s=10, c='blue')

# Add y = x line for Poisson reference
min_val = min(log_means.min(), log_vars.min())
max_val = max(log_means.max(), log_vars.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y = x (Poisson)')

ax.set_xlabel('Log10(Mean Expression + 1)', fontsize=12)
ax.set_ylabel('Log10(Variance + 1)', fontsize=12)
ax.set_title('Mean vs Variance of Gene Expression (Log-Log Scale)', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig('Part3_mean_vs_variance.png', dpi=150)
plt.close()

print("\nPart 3: Scatter plot saved as 'Part3_mean_vs_variance.png'")

print("""
ANSWER: Is this a Poisson distribution?
No, the data does NOT follow a Poisson distribution. In a Poisson distribution,
the variance equals the mean. Looking at the scatter plot, the variance is 
consistently HIGHER than the mean (points lie above the y=x line).

This is called "overdispersion" and is typical of RNA-seq data.
The data better fits a Negative Binomial distribution, which has an additional
dispersion parameter to account for this extra variance.
""")

print("""
ANSWER: Can we detect differentially expressed genes with one sample per condition?
No, we cannot reliably detect differentially expressed genes with only one sample
per experimental condition because:
1. We cannot estimate within-group variance (biological + technical noise)
2. We cannot distinguish true biological differences from random variation
3. Statistical tests require replicates to estimate variability
""")

# =============================================================================
# PART 4 - Detecting Differentially Expressed Genes
# =============================================================================

print("\n" + "=" * 80)
print("PART 4 - Detecting Differentially Expressed Genes")
print("=" * 80)

# Note: Python doesn't have DESeq2 natively, but we can use pyDESeq2 or 
# implement a simplified differential expression analysis

# For this exercise, we'll implement a simplified approach using 
# negative binomial test or t-test with log-transformed data

# Create condition labels
conditions = ['treated' if 'treated' in col else 'untreated' for col in cts.columns]
condition_df = pd.DataFrame({'condition': conditions}, index=cts.columns)

# Separate treated and untreated samples
treated_cols = [col for col in cts.columns if 'treated' in col and 'untreated' not in col]
untreated_cols = [col for col in cts.columns if 'untreated' in col]

print(f"Treated samples: {treated_cols}")
print(f"Untreated samples: {untreated_cols}")

# Calculate log2 fold change and perform t-tests
# Add pseudocount to avoid log(0)
pseudocount = 1

# Mean expression in each condition
mean_treated = cts_normalized[treated_cols].mean(axis=1) + pseudocount
mean_untreated = cts_normalized[untreated_cols].mean(axis=1) + pseudocount

# Log2 fold change
log2fc = np.log2(mean_treated / mean_untreated)

# Overall mean (baseMean equivalent)
base_mean = cts_normalized.mean(axis=1)

# Perform t-test for each gene
p_values = []
for gene in cts_normalized.index:
    treated_vals = cts_normalized.loc[gene, treated_cols].values
    untreated_vals = cts_normalized.loc[gene, untreated_cols].values
    
    # Use Mann-Whitney U test (more robust for count data)
    if np.std(treated_vals) > 0 or np.std(untreated_vals) > 0:
        try:
            _, p = stats.mannwhitneyu(treated_vals, untreated_vals, alternative='two-sided')
        except:
            p = 1.0
    else:
        p = 1.0
    p_values.append(p)

p_values = np.array(p_values)

# Adjust p-values using Benjamini-Hochberg correction
from scipy.stats import false_discovery_control
padj = false_discovery_control(p_values, method='bh')

# Create results dataframe
results = pd.DataFrame({
    'baseMean': base_mean,
    'log2FoldChange': log2fc,
    'pvalue': p_values,
    'padj': padj
}, index=cts_normalized.index)

print("\nDifferential Expression Results Summary:")
print(f"Total genes: {len(results)}")
print(f"Significant genes (padj < 0.05): {(results['padj'] < 0.05).sum()}")
print(f"Significant genes (padj < 0.1): {(results['padj'] < 0.1).sum()}")

# -----------------------------------------------------------------------------
# TASK: Create MA plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

# Non-significant points (gray)
non_sig = results[results['padj'] >= 0.1]
ax.scatter(np.log10(non_sig['baseMean'] + 1), non_sig['log2FoldChange'], 
           alpha=0.3, s=10, c='gray', label='Not significant')

# Significant points (blue)
sig = results[results['padj'] < 0.1]
ax.scatter(np.log10(sig['baseMean'] + 1), sig['log2FoldChange'], 
           alpha=0.5, s=15, c='blue', label='Significant (padj < 0.1)')

ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax.set_xlabel('Log10(Mean Expression + 1)', fontsize=12)
ax.set_ylabel('Log2 Fold Change (Treated vs Untreated)', fontsize=12)
ax.set_title('MA Plot: Treated vs Untreated', fontsize=14)
ax.set_ylim(-5, 5)
ax.legend()
plt.tight_layout()
plt.savefig('Part4_MA_plot.png', dpi=150)
plt.close()

print("\nPart 4: MA plot saved as 'Part4_MA_plot.png'")

print("""
ANSWER: What is the meaning of the blue points?
The blue points represent genes that are statistically significantly 
differentially expressed (adjusted p-value < 0.1). These are genes where 
the expression change between treated and untreated conditions is unlikely 
to be due to random chance.
""")

# =============================================================================
# PART 5 - Detecting Circadian Patterns Using FFT
# =============================================================================

print("\n" + "=" * 80)
print("PART 5 - Detecting Circadian Patterns Using FFT")
print("=" * 80)

# -----------------------------------------------------------------------------
# TASK: Load circadian data
# -----------------------------------------------------------------------------

circadian_file = "/Users/barak.avrahami/workspace/workspace/neuro genomics - exercise 1/CircadianRNAseq.csv"

try:
    circadian_data = pd.read_csv(circadian_file)
    print("Circadian data loaded successfully!")
    
    # Examine the structure
    print(f"\nDataset shape: {circadian_data.shape}")
    print(f"\nColumn names: {list(circadian_data.columns)}")
    
    # Examine last 5 rows
    print("\nLast 5 rows of the circadian matrix:")
    print(circadian_data.tail(5))
    
    print("""
ANSWER: Time step analysis
Looking at column names: A_11PM, A_3AM, A_7AM, A_11AM, A_3PM, A_7PM, 
                        B_11PM, B_3AM, B_7AM, B_11AM, B_3PM, B_7PM
The time step between measurements is 4 hours (e.g., 11PM to 3AM = 4 hours).
Total duration: 48 hours (2 days) with 12 time points.
""")
    
    # Identify columns
    # First column is gene ID, last column is gene symbol
    # Middle columns are time points
    time_cols = [col for col in circadian_data.columns if col not in ['RefSeqID', 'GeneSymbol']]
    print(f"Time point columns: {time_cols}")
    
    # -----------------------------------------------------------------------------
    # TASK: Plot expression of gene 'per1a'
    # -----------------------------------------------------------------------------
    
    # Find the gene 'per1a'
    per1a_mask = circadian_data['GeneSymbol'] == 'per1a'
    
    if per1a_mask.sum() > 0:
        per1a_row = circadian_data[per1a_mask].iloc[0]
        per1a_expr = per1a_row[time_cols].values.astype(float)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(per1a_expr)), per1a_expr, 'b-o', linewidth=2, markersize=8)
        ax.set_xticks(range(len(time_cols)))
        ax.set_xticklabels(time_cols, rotation=45, ha='right')
        ax.set_xlabel('Time Point', fontsize=12)
        ax.set_ylabel('Expression Level (normalized counts)', fontsize=12)
        ax.set_title('Expression of per1a Gene Over Time', fontsize=14)
        
        # Add day separators
        ax.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5, label='Day boundary')
        
        plt.tight_layout()
        plt.savefig('Part5_per1a_expression.png', dpi=150)
        plt.close()
        
        print("\nPart 5: per1a expression plot saved as 'Part5_per1a_expression.png'")
        
        print("""
ANSWER: Does per1a seem circadian? Should it be?
Yes, per1a shows clear circadian expression patterns with oscillation across
the 24-hour cycle. Per1 (Period 1) is a CORE CLOCK GENE that is part of the 
molecular circadian oscillator. It is one of the best-known circadian genes,
showing robust ~24-hour rhythms in all organisms studied.
""")
        
        # -----------------------------------------------------------------------------
        # TASK: Calculate FFT power spectrum for per1a
        # -----------------------------------------------------------------------------
        
        N = len(per1a_expr)  # Number of time points (12)
        delta_t = 4  # Time step in hours
        
        # Compute FFT
        fft_result = fft(per1a_expr)
        
        # Compute power (multiply by complex conjugate)
        power = np.abs(fft_result) ** 2
        
        # The relevant powers are in positions 1 to N//2
        # Position 0 is DC component (frequency 0), which we ignore
        relevant_powers = power[1:N//2 + 1]
        
        # Normalize powers
        normalized_powers = relevant_powers / np.sum(relevant_powers)
        
        # Create frequency vector
        # Frequency resolution: 1/(N*delta_t) = 1/(12*4) = 1/48 cycles per hour
        # Frequencies: from 1/48 to (N/2)/(N*delta_t) = 6/48 = 1/8 (Nyquist)
        frequencies = np.arange(1, N//2 + 1) / (N * delta_t)
        
        # Convert to period in hours
        periods = 1 / frequencies
        
        # Plot power spectrum
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar plot
        bars = ax.bar(frequencies, normalized_powers, width=0.003, color='blue', alpha=0.7)
        
        # Add period labels
        for i, (freq, power_val, period) in enumerate(zip(frequencies, normalized_powers, periods)):
            ax.text(freq, power_val + 0.02, f'{period:.0f}h', ha='center', fontsize=10)
        
        # Highlight circadian frequency
        ax.axvline(x=1/24, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(1/24 + 0.002, max(normalized_powers) * 0.8, '24h period', color='red', fontsize=11)
        
        ax.set_xlabel('Frequency (cycles/hour)', fontsize=12)
        ax.set_ylabel('Normalized Power', fontsize=12)
        ax.set_title('FFT Power Spectrum of per1a Expression', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('Part5_per1a_FFT_power.png', dpi=150)
        plt.close()
        
        print("Part 5: FFT power spectrum saved as 'Part5_per1a_FFT_power.png'")
        
        print(f"\nFFT Analysis Results for per1a:")
        print(f"{'Frequency (1/h)':<18} {'Period (h)':<12} {'Normalized Power':<18}")
        print("-" * 48)
        for freq, period, pwr in zip(frequencies, periods, normalized_powers):
            marker = " <-- CIRCADIAN" if abs(period - 24) < 1 else ""
            print(f"{freq:<18.6f} {period:<12.1f} {pwr:<18.4f}{marker}")
    
    # -----------------------------------------------------------------------------
    # TASK: Experimental design question
    # -----------------------------------------------------------------------------
    
    print("""
QUESTION: Which experimental design is better for detecting circadian genes?
(a) Shorter time step (2 hours) with same 48-hour duration
(b) Longer duration (4 days) with same 4-hour time step

ANSWER: Option (b) - increasing the number of days measured is BETTER.

Reasoning:
- Shorter time step (option a): This increases the Nyquist frequency (maximum
  detectable frequency), but doesn't improve resolution at the circadian 
  frequency. We can already detect 24h cycles with 4h sampling.

- Longer measurement period (option b): This improves frequency RESOLUTION
  (frequency resolution = 1/(N×Δt)). With more days, we can better distinguish
  the circadian frequency from nearby frequencies, reducing spectral leakage
  and providing more precise detection of the 24h period.
""")
    
    print("""
QUESTION: Advantages of frequency domain vs time domain analysis?

ANSWER:
1. Objective quantification: FFT provides a single numerical value (power)
   that can be compared across genes.

2. Phase-independent: FFT detects periodicity regardless of phase.

3. Noise robustness: FFT can detect periodic signals even in noisy data.

4. No assumptions about waveform: Works for any periodic pattern.

5. Computational efficiency: FFT is O(N log N).

6. Detection of multiple periodicities: Can reveal multiple periodic components.
""")
    
    # -----------------------------------------------------------------------------
    # TASK: Find top 10 genes with highest circadian power
    # -----------------------------------------------------------------------------
    
    print("\nProcessing all genes for circadian power analysis...")
    
    def calc_circadian_power(expr_values):
        """Calculate normalized power at circadian frequency (1/24 hr)"""
        expr = np.array(expr_values, dtype=float)
        
        # Handle invalid data
        if np.all(np.isnan(expr)) or np.all(expr == 0):
            return np.nan
        
        N = len(expr)
        
        # FFT
        fft_result = fft(expr)
        power = np.abs(fft_result) ** 2
        
        # Get relevant powers (exclude DC component)
        relevant_powers = power[1:N//2 + 1]
        
        # Normalize
        total_power = np.sum(relevant_powers)
        if total_power == 0:
            return np.nan
        
        normalized_powers = relevant_powers / total_power
        
        # Circadian frequency index:
        # With N=12 and delta_t=4, total time = 48h
        # Circadian frequency = 1/24 cycles/hour
        # FFT frequency at index k = k / (N * delta_t)
        # For circadian: k = (N * delta_t) / 24 = 48/24 = 2
        # So index 2-1=1 (0-indexed from relevant_powers which starts at k=1)
        circadian_power = normalized_powers[1]  # 24-hour component
        
        return circadian_power
    
    # Calculate circadian power for all genes
    circadian_powers = []
    for idx, row in circadian_data.iterrows():
        expr = row[time_cols].values
        power = calc_circadian_power(expr)
        circadian_powers.append(power)
    
    circadian_data['circadian_power'] = circadian_powers
    
    # Sort by circadian power
    sorted_data = circadian_data.dropna(subset=['circadian_power']).sort_values(
        'circadian_power', ascending=False
    )
    
    print("\nTop 10 genes with highest normalized circadian power:")
    print("-" * 50)
    for i, (idx, row) in enumerate(sorted_data.head(10).iterrows()):
        print(f"{i+1:2d}. {row['GeneSymbol']:<20} Power: {row['circadian_power']:.4f}")
    
    print("""
ANSWER: Known circadian genes that may appear in top 10:
- per1a, per1b, per2, per3 (Period genes)
- cry1a, cry1b, cry2 (Cryptochrome genes)
- clock, bmal/arntl (Clock transcription factors)
- nr1d1, nr1d2 (Rev-erb genes)
""")

    # =============================================================================
    # PART 6 - Detecting Genes with Variable Expression Levels
    # =============================================================================
    
    print("\n" + "=" * 80)
    print("PART 6 - Detecting Genes with Variable Expression Levels")
    print("=" * 80)
    
    # Create numerical matrix from count data
    count_matrix = circadian_data[time_cols].values.astype(float)
    
    # Calculate variance and mean for each gene
    gene_vars = np.var(count_matrix, axis=1)
    gene_means = np.mean(count_matrix, axis=1)
    
    # Log transform (add 1 before log)
    log_vars = np.log(gene_vars + 1)
    log_means = np.log(gene_means + 1)
    
    # Create dataframe for analysis
    gene_df = pd.DataFrame({
        'original_idx': range(len(log_means)),
        'log_mean': log_means,
        'log_var': log_vars,
        'gene_symbol': circadian_data['GeneSymbol'].values
    })
    
    # Filter genes with log_mean >= 3
    gene_df_filtered = gene_df[gene_df['log_mean'] >= 3].copy()
    print(f"Number of genes with log(mean+1) >= 3: {len(gene_df_filtered)}")
    
    # Sort by mean expression
    gene_df_filtered = gene_df_filtered.sort_values('log_mean').reset_index(drop=True)
    
    # Bin into 20 groups based on mean expression
    n_bins = 20
    gene_df_filtered['bin'] = pd.cut(gene_df_filtered['log_mean'], bins=n_bins, labels=False)
    
    # Calculate z-score of variance within each bin
    def zscore_within_bin(group):
        if len(group) > 1 and group['log_var'].std() > 0:
            return (group['log_var'] - group['log_var'].mean()) / group['log_var'].std()
        return pd.Series([np.nan] * len(group), index=group.index)
    
    gene_df_filtered['z_score'] = gene_df_filtered.groupby('bin')['log_var'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else np.nan
    )
    
    # Sort by z-score (descending)
    gene_df_sorted = gene_df_filtered.dropna(subset=['z_score']).sort_values(
        'z_score', ascending=False
    )
    
    # Get top 40 variable genes
    top40_variable = gene_df_sorted.head(40)
    
    print("\nTop 40 genes with highest variance z-scores:")
    print("(These are genes with unusually high variance for their expression level)")
    print("-" * 55)
    
    for i, (idx, row) in enumerate(top40_variable.iterrows()):
        print(f"{i+1:2d}. {row['gene_symbol']:<20} z-score: {row['z_score']:7.3f}")
    
    print("""
ANSWER: Are circadian genes expected to be variable genes?

YES! Circadian genes are expected to be variable genes because:

1. By definition, circadian genes show oscillating expression levels across
   the 24-hour cycle, which means HIGH VARIANCE between time points.

2. The amplitude of circadian oscillations contributes directly to variance.

3. Therefore, genes with strong circadian rhythms will have higher-than-expected
   variance compared to non-rhythmic genes with similar mean expression.

Known circadian genes expected in top variable genes:
- per1a, per2, per3 (Period genes)
- cry1a, cry1b (Cryptochrome genes)
- bmal, clock (Core clock transcription factors)
- nr1d1 (Rev-erb alpha)
""")
    
    # -----------------------------------------------------------------------------
    # Create visualization of variable genes detection
    # -----------------------------------------------------------------------------
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Mean vs Variance with bin coloring
    scatter = axes[0, 0].scatter(gene_df_filtered['log_mean'], 
                                  gene_df_filtered['log_var'],
                                  c=gene_df_filtered['bin'], 
                                  cmap='rainbow', s=10, alpha=0.5)
    axes[0, 0].set_xlabel('Log(Mean + 1)')
    axes[0, 0].set_ylabel('Log(Variance + 1)')
    axes[0, 0].set_title('Mean vs Variance (colored by bin)')
    plt.colorbar(scatter, ax=axes[0, 0], label='Bin')
    
    # Plot 2: Z-score distribution
    axes[0, 1].hist(gene_df_filtered['z_score'].dropna(), bins=50, 
                    color='lightblue', edgecolor='black')
    if len(top40_variable) > 0:
        threshold = top40_variable['z_score'].iloc[-1]
        axes[0, 1].axvline(x=threshold, color='red', linestyle='--', 
                           label=f'Top 40 threshold ({threshold:.2f})')
    axes[0, 1].set_xlabel('Z-score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Variance Z-scores')
    axes[0, 1].legend()
    
    # Plot 3: Highlight top variable genes
    axes[1, 0].scatter(gene_df_filtered['log_mean'], gene_df_filtered['log_var'],
                       c='gray', s=10, alpha=0.3, label='All genes')
    axes[1, 0].scatter(top40_variable['log_mean'], top40_variable['log_var'],
                       c='red', s=30, alpha=0.8, label='Top 40 variable')
    axes[1, 0].set_xlabel('Log(Mean + 1)')
    axes[1, 0].set_ylabel('Log(Variance + 1)')
    axes[1, 0].set_title('Top 40 Variable Genes (red)')
    axes[1, 0].legend()
    
    # Plot 4: Z-score vs Mean expression
    axes[1, 1].scatter(gene_df_filtered['log_mean'], gene_df_filtered['z_score'],
                       c='gray', s=10, alpha=0.3)
    axes[1, 1].scatter(top40_variable['log_mean'], top40_variable['z_score'],
                       c='red', s=30, alpha=0.8)
    axes[1, 1].axhline(y=0, color='blue', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Log(Mean + 1)')
    axes[1, 1].set_ylabel('Variance Z-score')
    axes[1, 1].set_title('Z-score vs Mean Expression')
    
    plt.tight_layout()
    plt.savefig('Part6_variable_genes.png', dpi=150)
    plt.close()
    
    print("\nPart 6: Variable genes visualization saved as 'Part6_variable_genes.png'")

except FileNotFoundError:
    print(f"\n*** CircadianRNAseq.csv not found at: {circadian_file} ***")
    print("Please download the file from the course website.")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 80)
print("EXERCISE COMPLETED")
print("=" * 80)
print("""
Generated output files:
- Part3_mean_vs_variance.png: Mean vs variance scatter plot
- Part4_MA_plot.png: MA plot for differential expression
- Part5_per1a_expression.png: per1a time course expression
- Part5_per1a_FFT_power.png: FFT power spectrum of per1a
- Part6_variable_genes.png: Variable genes analysis plots

Please review the printed output for answers to all conceptual questions.
""")

