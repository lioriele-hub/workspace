"""
Neuro-Genomics Exercise 2 - Part 2: Circadian Detection with Permutation Analysis

This script detects rhythmical patterns in gene expression using FFT,
and uses permutation analysis to determine the false positive rate.

The "G factor" is the normalized FFT power at the circadian frequency (1/24 hr).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_g_factor(expression_values: np.ndarray) -> float:
    """
    Calculate the G factor (normalized FFT power at circadian frequency 1/24).
    
    Parameters
    ----------
    expression_values : np.ndarray
        Time-course expression values (12 time points, 4-hour intervals)
    
    Returns
    -------
    float
        The G factor (normalized power at 24-hour frequency)
        Value between 0 and 1. Higher = more circadian.
    """
    expr = np.array(expression_values, dtype=float)
    
    # Handle invalid data
    if np.all(np.isnan(expr)) or np.all(expr == 0) or len(expr) != 12:
        return np.nan
    
    N = len(expr)  # Should be 12 time points
    
    # Compute FFT
    fft_result = np.fft.fft(expr)
    
    # Compute power spectrum (squared magnitude)
    power = np.abs(fft_result) ** 2
    
    # Get relevant powers (exclude DC component at index 0)
    # For N=12, we have 6 unique non-DC frequencies (indices 1 to 6)
    relevant_powers = power[1:N//2 + 1]  # indices 1 to 6
    
    # Normalize powers
    total_power = np.sum(relevant_powers)
    if total_power == 0:
        return np.nan
    
    normalized_powers = relevant_powers / total_power
    
    # Circadian frequency calculation:
    # Time step = 4 hours, N = 12 time points, total time = 48 hours
    # FFT frequency at index k = k / (N * delta_t) = k / 48
    # For 24-hour period: frequency = 1/24
    # So k = 48 / 24 = 2
    # In relevant_powers array (starting from k=1), circadian is at index 1 (k=2)
    circadian_power = normalized_powers[1]  # Index 1 corresponds to k=2 (24h period)
    
    return circadian_power


def calculate_all_g_factors(data_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate G factors for all genes in the dataset.
    
    Parameters
    ----------
    data_matrix : np.ndarray
        Matrix of shape (n_genes, 12) with expression values
    
    Returns
    -------
    np.ndarray
        Array of G factors for each gene
    """
    g_factors = np.array([calculate_g_factor(row) for row in data_matrix])
    return g_factors


def count_genes_above_threshold(g_factors: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Count genes with G factor above each threshold.
    
    Parameters
    ----------
    g_factors : np.ndarray
        G factors for all genes
    thresholds : np.ndarray
        Array of thresholds (0 to 0.99)
    
    Returns
    -------
    np.ndarray
        Count of genes above each threshold
    """
    # Remove NaN values for counting
    valid_g = g_factors[~np.isnan(g_factors)]
    counts = np.array([np.sum(valid_g >= t) for t in thresholds])
    return counts


def permute_data(data_matrix: np.ndarray) -> np.ndarray:
    """
    Permute the data to remove biological signal while preserving statistics.
    
    We shuffle COLUMNS (time points) for each gene independently.
    This preserves the expression distribution of each gene but removes
    the temporal pattern (circadian signal).
    
    Why columns (time points) and not rows (genes)?
    - Shuffling genes would keep the time structure intact, preserving circadian patterns
    - Shuffling time points for each gene breaks the temporal correlation
      while preserving each gene's expression statistics
    
    Parameters
    ----------
    data_matrix : np.ndarray
        Original data matrix (n_genes x 12 time points)
    
    Returns
    -------
    np.ndarray
        Permuted data matrix
    """
    permuted = data_matrix.copy()
    
    # For each gene (row), shuffle its time points (columns)
    for i in range(permuted.shape[0]):
        np.random.shuffle(permuted[i, :])
    
    return permuted


def run_permutation_analysis(data_matrix: np.ndarray, n_permutations: int = 100) -> np.ndarray:
    """
    Run permutation analysis to estimate false positive rates.
    
    Parameters
    ----------
    data_matrix : np.ndarray
        Original data matrix (n_genes x 12 time points)
    n_permutations : int
        Number of permutations (default: 100)
    
    Returns
    -------
    np.ndarray
        Average gene counts for each threshold from permuted data
    """
    thresholds = np.arange(0, 1.0, 0.01)
    all_counts = np.zeros((n_permutations, len(thresholds)))
    
    print(f"Running {n_permutations} permutations...")
    
    for i in range(n_permutations):
        if (i + 1) % 10 == 0:
            print(f"  Permutation {i + 1}/{n_permutations}")
        
        # Permute the data
        permuted_data = permute_data(data_matrix)
        
        # Calculate G factors for permuted data
        g_factors_perm = calculate_all_g_factors(permuted_data)
        
        # Count genes above each threshold
        all_counts[i, :] = count_genes_above_threshold(g_factors_perm, thresholds)
    
    # Average across all permutations
    avg_counts = np.mean(all_counts, axis=0)
    
    return avg_counts


# =============================================================================
# Main Script
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Part 2: Circadian Detection with Permutation Analysis")
    print("=" * 70)
    
    # Load circadian data
    circadian_file = "neuro genomics - exercise 2/CircadianRNAseq.csv"
    
    try:
        circadian_df = pd.read_csv(circadian_file)
        print(f"Loaded circadian data: {circadian_df.shape}")
    except FileNotFoundError:
        try:
            circadian_df = pd.read_csv("CircadianRNAseq.csv")
            print(f"Loaded circadian data from current directory: {circadian_df.shape}")
        except:
            print("Error: Could not find CircadianRNAseq.csv")
            print("Please place it in the current directory or exercise 1 folder")
            exit(1)
    
    # Identify time point columns (all except first and last)
    all_cols = circadian_df.columns.tolist()
    time_cols = [col for col in all_cols if col not in ['RefSeqID', 'GeneSymbol']]
    print(f"Time point columns: {time_cols}")
    
    # Extract numerical data matrix
    data_matrix = circadian_df[time_cols].values.astype(float)
    gene_symbols = circadian_df['GeneSymbol'].values
    
    print(f"Data matrix shape: {data_matrix.shape}")
    print(f"Number of genes: {len(gene_symbols)}")
    
    # ==========================================================================
    # Step 1: Calculate G factors for real data
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Calculating G factors for real data...")
    print("=" * 70)
    
    g_factors_real = calculate_all_g_factors(data_matrix)
    print(f"Calculated G factors for {np.sum(~np.isnan(g_factors_real))} valid genes")
    
    # Count genes above each threshold
    thresholds = np.arange(0, 1.0, 0.01)
    counts_real = count_genes_above_threshold(g_factors_real, thresholds)
    
    print(f"\nSample counts from real data:")
    print(f"  G >= 0.00: {counts_real[0]:,} genes")
    print(f"  G >= 0.30: {counts_real[30]:,} genes")
    print(f"  G >= 0.50: {counts_real[50]:,} genes")
    print(f"  G >= 0.70: {counts_real[70]:,} genes")
    
    # ==========================================================================
    # Steps 2-4: Run permutation analysis (100 permutations)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Steps 2-4: Running permutation analysis...")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    counts_permuted = run_permutation_analysis(data_matrix, n_permutations=100)
    
    print(f"\nSample counts from permuted data (averaged over 100 permutations):")
    print(f"  G >= 0.00: {counts_permuted[0]:,.1f} genes")
    print(f"  G >= 0.30: {counts_permuted[30]:,.1f} genes")
    print(f"  G >= 0.50: {counts_permuted[50]:,.1f} genes")
    print(f"  G >= 0.70: {counts_permuted[70]:,.1f} genes")
    
    # ==========================================================================
    # Step 5: Plot real vs permuted counts
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Plotting results...")
    print("=" * 70)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, counts_real, 'b-', linewidth=2, label='Real data')
    ax.plot(thresholds, counts_permuted, 'r--', linewidth=2, label='Permuted data (avg of 100)')
    
    ax.set_xlabel('G Factor Cutoff', fontsize=12)
    ax.set_ylabel('Number of Genes Detected', fontsize=12)
    ax.set_title('Genes Detected vs G Factor Cutoff', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Part2_genes_detected.png', dpi=150)
    plt.close()
    print("Plot saved as 'Part2_genes_detected.png'")
    
    # ==========================================================================
    # Step 6 & 7: Calculate and plot true positive rate
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Steps 6-7: Calculating true positive rates...")
    print("=" * 70)
    
    # True positive rate = (real - permuted) / real
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tp_rate = (counts_real - counts_permuted) / counts_real
        tp_rate = np.where(counts_real == 0, 0, tp_rate)
        tp_rate = np.clip(tp_rate, 0, 1)  # Ensure values are between 0 and 1
    
    # Plot true positive rate
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, tp_rate, 'g-', linewidth=2)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% true positive rate')
    
    ax.set_xlabel('G Factor Cutoff', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('True Positive Rate vs G Factor Cutoff', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Part2_true_positive_rate.png', dpi=150)
    plt.close()
    print("Plot saved as 'Part2_true_positive_rate.png'")
    
    # Find threshold for 80% true positive rate
    tp_80_idx = np.where(tp_rate >= 0.8)[0]
    if len(tp_80_idx) > 0:
        g_cutoff_80 = thresholds[tp_80_idx[0]]
        genes_at_80 = int(counts_real[tp_80_idx[0]])
    else:
        g_cutoff_80 = thresholds[-1]
        genes_at_80 = int(counts_real[-1])
    
    print(f"\nResults:")
    print(f"  G factor cutoff for 80% true positive rate: {g_cutoff_80:.2f}")
    print(f"  Number of circadian genes at this cutoff: {genes_at_80}")
    
    # List the circadian genes at 80% TP rate
    circadian_mask = g_factors_real >= g_cutoff_80
    circadian_genes = gene_symbols[circadian_mask]
    circadian_g_values = g_factors_real[circadian_mask]
    
    # Sort by G factor
    sort_idx = np.argsort(circadian_g_values)[::-1]
    circadian_genes = circadian_genes[sort_idx]
    circadian_g_values = circadian_g_values[sort_idx]
    
    print(f"\nTop 20 circadian genes (G factor >= {g_cutoff_80:.2f}):")
    print("-" * 40)
    for i, (gene, g_val) in enumerate(zip(circadian_genes[:20], circadian_g_values[:20])):
        print(f"  {i+1:2d}. {gene:<20} G = {g_val:.4f}")
    
    # Save circadian genes for Part 3
    circadian_df_out = pd.DataFrame({
        'GeneSymbol': circadian_genes,
        'G_factor': circadian_g_values
    })
    circadian_df_out.to_csv('circadian_genes_80tp.csv', index=False)
    print(f"\nCircadian genes saved to 'circadian_genes_80tp.csv'")
    
    # Also save the expression data for these genes (needed for Part 3)
    circadian_indices = np.where(circadian_mask)[0]
    circadian_expression = data_matrix[circadian_indices, :]
    
    circadian_expr_df = pd.DataFrame(
        circadian_expression,
        columns=time_cols,
        index=circadian_genes
    )
    circadian_expr_df.to_csv('circadian_genes_expression.csv')
    print(f"Circadian gene expression data saved to 'circadian_genes_expression.csv'")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total genes analyzed: {len(gene_symbols)}")
    print(f"Genes with valid G factor: {np.sum(~np.isnan(g_factors_real))}")
    print(f"Circadian genes (80% TP rate): {len(circadian_genes)}")
    print(f"G factor cutoff used: {g_cutoff_80:.2f}")

