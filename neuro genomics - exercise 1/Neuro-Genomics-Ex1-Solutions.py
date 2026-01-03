"""
=============================================================================
Neuro-Genomics Exercise 1 - Principles of Sequencing Data Analysis
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import fft
from scipy.optimize import curve_fit
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# PART 2 - Explore Sequencing Data Using Python
# =============================================================================

print("\n" + "=" * 80)
print("PART 2 - Explore Sequencing Data")
print("=" * 80)

# Load pasilla data from local file
cts = pd.read_csv("neuro genomics - exercise 1/pasilla_gene_counts.tsv", sep='\t', index_col='gene_id')

# Examine the first 10 lines of the cts matrix
print("\nFirst 10 lines of the cts matrix:")
print(cts.head(10))

# Define a variable to hold matrix dimensions and print them
dims = cts.shape
print(f"\nMatrix dimensions:")
print(f"Number of rows (genes): {dims[0]}")
print(f"Number of columns (samples): {dims[1]}")

# Sum the number of reads for each sample
sample_sums = cts.sum(axis=0)
print("\nSum of reads per sample:")
print(sample_sums)

# Multiply each column by a factor so total reads equal the first column
target_sum = sample_sums.iloc[0]
normalization_factors = target_sum / sample_sums

# Create normalized cts matrix
cts_normalized = cts.multiply(normalization_factors, axis=1)
print("\nFirst 10 lines of the normalized cts matrix:")
print(cts_normalized.head(10))

# Verify that normalized sums are equal
normalized_sums = cts_normalized.sum(axis=0)
print("\nSum of reads per sample after normalization:")
print(normalized_sums)

# =============================================================================
# PART 3 - Basic Statistics of Sequencing Data
# =============================================================================

print("\n" + "=" * 80)
print("PART 3 - Basic Statistics of Sequencing Data")
print("=" * 80)

# Define untreated and treated sample columns
untreated_cols = [col for col in cts.columns if 'untreated' in col]
treated_cols = [col for col in cts.columns if 'treated' in col and 'untreated' not in col]

# Compute means and variances per group (all, treated, untreated)
means = {}
vars_ = {}

# means['all'] = cts_normalized.mean(axis=1)
# vars_['all'] = cts_normalized.var(axis=1)

means['treated'] = cts_normalized[treated_cols].mean(axis=1)
vars_['treated'] = cts_normalized[treated_cols].var(axis=1)

means['untreated'] = cts_normalized[untreated_cols].mean(axis=1)
vars_['untreated'] = cts_normalized[untreated_cols].var(axis=1)

# Log transform (adding 1 to avoid log(0)) for treated and untreated groups
log_means_treated = np.log10(means['treated'] + 1)
log_vars_treated = np.log10(vars_['treated'] + 1)

log_means_untreated = np.log10(means['untreated'] + 1)
log_vars_untreated = np.log10(vars_['untreated'] + 1)

# Fit dispersion a in variance = mean + a*mean^2 using curve_fit instead of nls function in R
x_t = means['treated'].values
y_t = vars_['treated'].values
popt_t, _ = curve_fit(lambda mu, a: mu + a * (mu ** 2), x_t, y_t, p0=[0.0], maxfev=10000)
a_treated = popt_t[0]

x_u = means['untreated'].values
y_u = vars_['untreated'].values
popt_u, _ = curve_fit(lambda mu, a: mu + a * (mu ** 2), x_u, y_u, p0=[0.0], maxfev=10000)
a_untreated = popt_u[0]

# Build 2x2 subplot: top row Poisson checks, bottom row Negative Binomial fits
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=False)

# Poisson - Untreated (top-left)
ax00 = axes[0, 0]
ax00.scatter(log_means_untreated, log_vars_untreated, alpha=0.35, s=12, c='green', label='Untreated')
min_u = min(log_means_untreated.min(), log_vars_untreated.min())
max_u = max(log_means_untreated.max(), log_vars_untreated.max())
ax00.plot([min_u, max_u], [min_u, max_u], 'r--', linewidth=1, label='y = x')
ax00.set_xlabel('Log10(Mean + 1)', fontsize=12)
ax00.set_ylabel('Log10(Variance + 1)', fontsize=12)
ax00.set_title('Poisson check (Untreated)', fontsize=13)
ax00.legend()

# Poisson - Treated (top-right)
ax01 = axes[0, 1]
ax01.scatter(log_means_treated, log_vars_treated, alpha=0.35, s=12, c='orange', label='Treated')
min_t = min(log_means_treated.min(), log_vars_treated.min())
max_t = max(log_means_treated.max(), log_vars_treated.max())
ax01.plot([min_t, max_t], [min_t, max_t], 'r--', linewidth=1, label='y = x')
ax01.set_xlabel('Log10(Mean + 1)', fontsize=12)
ax01.set_title('Poisson check (Treated)', fontsize=13)
ax01.legend()

# Negative Binomial - Untreated (bottom-left)
ax10 = axes[1, 0]
ax10.scatter(log_means_untreated, log_vars_untreated, alpha=0.35, s=12, c='green', label='Untreated')
xfit = np.linspace(means['untreated'].min(), means['untreated'].max(), 200)
yfit = xfit + a_untreated * (xfit ** 2)
ax10.plot(np.log10(xfit + 1), np.log10(yfit + 1), 'k-', linewidth=1.5, label=f'NB fit (a={a_untreated:.3g})')
ax10.set_xlabel('Log10(Mean + 1)', fontsize=12)
ax10.set_ylabel('Log10(Variance + 1)', fontsize=12)
ax10.set_title('Negative Binomial fit (Untreated)', fontsize=13)
ax10.legend()

# Negative Binomial - Treated (bottom-right)
ax11 = axes[1, 1]
ax11.scatter(log_means_treated, log_vars_treated, alpha=0.35, s=12, c='orange', label='Treated')
xfit = np.linspace(means['treated'].min(), means['treated'].max(), 200)
yfit = xfit + a_treated * (xfit ** 2)
ax11.plot(np.log10(xfit + 1), np.log10(yfit + 1), 'k-', linewidth=1.5, label=f'NB fit (a={a_treated:.3g})')
ax11.set_xlabel('Log10(Mean + 1)', fontsize=12)
ax11.set_title('Negative Binomial fit (Treated)', fontsize=13)
ax11.legend()

plt.suptitle('Mean vs Variance: Poisson (top) and Negative Binomial (bottom)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('Part3_dispersion_poisson_nb.png', dpi=150)
plt.close()

print(f"Dispersion parameter (a) untreated: {a_untreated:.4g}")
print(f"Dispersion parameter (a) treated:   {a_treated:.4g}")


# =============================================================================
# PART 4 - Detecting Differentially Expressed Genes 
# =============================================================================

print("\n" + "=" * 80)
print("PART 4 - Detecting Differentially Expressed Genes")
print("=" * 80)

# Visual inspection: first treated vs first untreated
first_untreated = untreated_cols[0]
first_treated = treated_cols[0]

log_u1 = np.log10(cts[first_untreated] + 1)
log_t1 = np.log10(cts[first_treated] + 1)

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(log_u1, log_t1, alpha=0.35, s=10, c='steelblue')
ax.set_xlabel(f'Log10 counts +1: {first_untreated}', fontsize=11)
ax.set_ylabel(f'Log10 counts +1: {first_treated}', fontsize=11)
ax.set_title('Visual screen: treated_1 vs untreated_1', fontsize=13)
ax.plot([log_u1.min(), log_u1.max()], [log_u1.min(), log_u1.max()], 'r--', linewidth=1)
plt.tight_layout()
plt.savefig('Part4_visual_first_pair.png', dpi=150)
plt.close()

# Grid-like filter to flag striking genes based on visual inspection of the scatter plot
treat_thresh = 2.0  # Log10 scale: genes with counts > 100 in treated
untreat_thresh = 1.0  # Log10 scale: genes with counts < 10 in untreated

grid_mask = ((log_t1 >= treat_thresh) & (log_u1 <= untreat_thresh))
grid_genes = cts.index[grid_mask].tolist()

print("\nGenes visually flagged (treated high, untreated low):")
print(grid_genes)

# Pick one flagged gene for full-profile plot
selected_gene = grid_genes[1]

selected_expr = cts.loc[selected_gene]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(len(cts.columns)), selected_expr.values, 'o-', linewidth=2, markersize=6)
ax.set_xticks(range(len(cts.columns)))
ax.set_xticklabels(cts.columns, rotation=45, ha='right')
ax.set_xlabel('Sample index', fontsize=11)
ax.set_ylabel('Raw counts', fontsize=11)
ax.set_title(f'Expression of {selected_gene} across all samples', fontsize=13)
plt.tight_layout()
plt.savefig('Part4_selected_gene_profile.png', dpi=150)
plt.close()

# The DESeq2 methodology was implemented in Python using the PyDESeq2 package
# Condition labels and metadata
conditions = ['treated' if 'treated' in col and 'untreated' not in col else 'untreated' for col in cts.columns]
metadata = pd.DataFrame({'condition': conditions}, index=cts.columns)

# PyDESeq2 expects samples as rows, genes as columns
counts_t = cts.T.copy()

# Create DeseqDataSet and run DESeq2
dds = DeseqDataSet(
    counts=counts_t,
    metadata=metadata,
    design_factors="condition",
    ref_level=['condition', 'untreated']  # Set untreated as reference
)
dds.deseq2()

# Get results (treated vs untreated)
stats_result = DeseqStats(dds, contrast=['condition', 'treated', 'untreated'])
stats_result.summary()
results = stats_result.results_df

# Sort by adjusted p-value
results_sorted = results.sort_values('padj')

# Get top 10 DE genes
top10_de = results_sorted.head(10)[['log2FoldChange', 'padj']]

print("\nTop 10 differentially expressed genes (sorted by padj):")
print("=" * 60)
print(top10_de)

# Check if visually inspected gene is in top 10
print(f"\nVisually inspected gene: {selected_gene}")
if selected_gene in top10_de.index:
    print(f"{selected_gene} is in the top 10 DE genes!")
    print(f"log2FoldChange: {top10_de.loc[selected_gene, 'log2FoldChange']:.4f}")
    print(f"padj: {top10_de.loc[selected_gene, 'padj']:.4e}")
else:
    print(f"{selected_gene} is not in the top 10 DE genes")
    if selected_gene in results.index:
        rank = list(results_sorted.index).index(selected_gene) + 1
        print(f"(It ranks #{rank} overall with padj = {results.loc[selected_gene, 'padj']:.4e})")


# =============================================================================
# PART 5 - Detecting Circadian Patterns Using FFT
# =============================================================================

print("\n" + "=" * 80)
print("PART 5 - Detecting Circadian Patterns Using FFT")
print("=" * 80)

# Load circadian data
circadian_file = "neuro genomics - exercise 1\CircadianRNAseq.csv"
circadian_data = pd.read_csv(circadian_file)

# Examine the structure
  # print(f"\nDataset shape: {circadian_data.shape}")
  # print(f"\nColumn names: {list(circadian_data.columns)}")

# Examine last 5 rows
print("\nLast 5 rows of the circadian matrix:")
print(circadian_data.tail(5))

# Identify columns
time_cols = [col for col in circadian_data.columns if col not in ['RefSeqID', 'GeneSymbol']]

# Find the gene 'per1a'
per1a_mask = circadian_data['GeneSymbol'] == 'per1a'
per1a_row = circadian_data[per1a_mask].iloc[0]
per1a_expr = per1a_row[time_cols].values.astype(float)

# Plot expression of gene 'per1a'
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(len(per1a_expr)), per1a_expr, 'b-o', linewidth=2, markersize=8)
ax.set_xticks(range(len(time_cols)))
ax.set_xticklabels(time_cols, rotation=45, ha='right')
ax.set_xlabel('Time Point', fontsize=12)
ax.set_ylabel('Expression Level', fontsize=12)
ax.set_title('Expression of per1a Gene Over Time', fontsize=14)
plt.tight_layout()
plt.savefig('Part5_per1a_expression.png', dpi=150)
plt.close()

# TASK: Calculate FFT power spectrum for per1a
# -----------------------------------------------------------------------------

N = len(per1a_expr)  # Number of time points (12)
delta_t = 4  # Time step in hours

# Compute FFT
fft_result = fft(per1a_expr)

# Compute power (multiply by complex conjugate)
power = np.abs(fft_result) ** 2

# The relevant powers are in positions 1 to N//2
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
ax.scatter(frequencies, normalized_powers, s=100, color='black', alpha=0.7, edgecolors='black', linewidth=1)
ax.set_xlabel('Frequency (cycles/hour)', fontsize=12)
ax.set_ylabel('Normalized Power', fontsize=12)
ax.set_title('FFT Power Spectrum of per1a Expression', fontsize=14)
plt.tight_layout()
plt.savefig('Part5_per1a_FFT_power.png', dpi=150)
plt.close()

# TASK: Find top 10 genes with highest circadian power
# -----------------------------------------------------------------------------

# Store all the circadian frequency powers in a vector
circadian_powers = []

N = len(time_cols)  # Number of time points (12)
delta_t = 4  # Time step in hours

for idx, row in circadian_data.iterrows():
    expr = row[time_cols].values.astype(float)
    
    # Compute FFT
    fft_result = fft(expr)
    power = np.abs(fft_result) ** 2
    
    # Get relevant powers (exclude DC component at position 0)
    relevant_powers = power[1:N//2 + 1]
    
    # Normalize powers
    total_power = np.sum(relevant_powers)
    normalized_powers = relevant_powers / total_power
    
    circadian_power = normalized_powers[1]  # 24-hour component
    circadian_powers.append(circadian_power)

# Add circadian powers to dataframe
circadian_data['circadian_power'] = circadian_powers

# Sort by circadian power (NAs will be at the end)
sorted_data = circadian_data.sort_values('circadian_power', ascending=False, na_position='last')

# Get top 10 genes (excluding NAs)
top10_genes = sorted_data.dropna(subset=['circadian_power']).head(10)

print("\nTop 10 genes with highest normalized circadian power:")
print("-" * 60)
for i, (idx, row) in enumerate(top10_genes.iterrows()):
    print(f"{i+1:2d}. {row['GeneSymbol']:<20} Power: {row['circadian_power']:.4f}")

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
gene_df_filtered['bin'] = pd.qcut(gene_df_filtered['log_mean'], q=n_bins, labels=False, duplicates='drop')

# Show the expression range (log_mean) for each bin
  # print("\nExpression range (log_mean) for each bin:")
  # bin_ranges = gene_df_filtered.groupby('bin')['log_mean'].agg(['min', 'max', 'count'])
  # print(bin_ranges)

# Calculate z-score of variance within each bin
gene_df_filtered['z_score'] = gene_df_filtered.groupby('bin')['log_var'].transform(
    lambda x: (x - x.mean()) / x.std())

# Sort by z-score (descending)
gene_df_sorted = gene_df_filtered.sort_values('z_score', ascending=False)

# Get top 40 variable genes
top40_variable = gene_df_sorted.head(40)

print("\nTop 40 genes with highest variance z-scores:")
print("-" * 55)

for i, (idx, row) in enumerate(top40_variable.iterrows()):
    print(f"{i+1:2d}. {row['gene_symbol']:<20} z-score: {row['z_score']:7.3f}")

# =============================================================================