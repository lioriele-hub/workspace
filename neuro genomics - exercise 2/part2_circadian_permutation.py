"""
Neuro-Genomics Exercise 2 - Part 2: Circadian Detection with Permutation Analysis
This script detects rhythmical patterns in gene expression using FFT, and uses permutation analysis to determine the false positive rate.
The "G factor" is the normalized FFT power at the circadian frequency (1/24 hr).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Main execution - Run on circadian data
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
        print(f"Error: Could not find {circadian_file}")
        exit(1)

    # Identify time point columns (all except RefSeqID and GeneSymbol)
    all_cols = circadian_df.columns.tolist()
    time_cols = [col for col in all_cols if col not in ['RefSeqID', 'GeneSymbol']]
    print(f"Time point columns: {time_cols}")

    # Extract numerical data matrix
    data_matrix = circadian_df[time_cols].values.astype(float)
    gene_symbols = circadian_df['GeneSymbol'].values

    print(f"Data matrix shape: {data_matrix.shape}")
    print(f"Number of genes: {len(gene_symbols)}")

    # -------------------------------------------------------------------------
    # Step 1: Calculate G factors for real data 
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 1: Calculating G factors for real data")
    print("=" * 70)

    g_list = []
    for row in data_matrix:
        expr = np.array(row, dtype=float)
        
        N = len(expr)
        fft_result = np.fft.fft(expr)
        power = np.abs(fft_result) ** 2
        relevant_powers = power[1:N // 2 + 1]
        total_power = np.sum(relevant_powers)
        if total_power == 0:
            g_list.append(0.0)
            continue
        normalized_powers = relevant_powers / total_power
        # circadian at k=2 -> index 1 in relevant_powers
        circadian_power = normalized_powers[1]
        g_list.append(circadian_power)

    g_factors_real = np.array(g_list)
    print(f"Calculated G factors for {len(g_factors_real)} genes")

    # Count genes above each threshold
    thresholds = np.arange(0, 1.0, 0.01)
    counts_real = np.array([np.sum(g_factors_real >= t) for t in thresholds])

    print(f"\nSample counts from real data:")
    print(f"  G >= 0.00: {counts_real[0]:,} genes")
    print(f"  G >= 0.30: {counts_real[30]:,} genes")
    print(f"  G >= 0.50: {counts_real[50]:,} genes")
    print(f"  G >= 0.70: {counts_real[70]:,} genes")

    # -------------------------------------------------------------------------
    # Steps 2-4: Run permutation analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Steps 2-4: Running permutation analysis")
    print("=" * 70)

    # np.random.seed(42)
    n_permutations = 100
    all_counts = np.zeros((n_permutations, len(thresholds)))
    print(f"Running {n_permutations} permutations...")

    for i in range(n_permutations):
        # Permute data by shuffling time points for each gene
        permuted = data_matrix.copy()
        for j in range(permuted.shape[0]):
            np.random.shuffle(permuted[j, :])

        # Calculate G factors for permuted data 
        g_perm_list = []
        for row in permuted:
            expr = np.array(row, dtype=float)
            N = len(expr)
            fft_result = np.fft.fft(expr)
            power = np.abs(fft_result) ** 2
            relevant_powers = power[1:N // 2 + 1]
            total_power = np.sum(relevant_powers)
            if total_power == 0:
                g_perm_list.append(0.0)
                continue
            normalized_powers = relevant_powers / total_power
            g_perm_list.append(normalized_powers[1])

        g_factors_perm = np.array(g_perm_list)
        all_counts[i, :] = np.array([np.sum(g_factors_perm >= t) for t in thresholds])

    counts_permuted = np.mean(all_counts, axis=0)

    print(f"\nSample counts from permuted data (averaged over {n_permutations} permutations):")
    print(f"  G >= 0.00: {counts_permuted[0]:,.1f} genes")
    print(f"  G >= 0.30: {counts_permuted[30]:,.1f} genes")
    print(f"  G >= 0.50: {counts_permuted[50]:,.1f} genes")
    print(f"  G >= 0.70: {counts_permuted[70]:,.1f} genes")

    # -------------------------------------------------------------------------
    # Step 5: Plot real vs permuted counts
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 5: Plotting results")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, counts_real, 'b-', linewidth=2, label='Real data')
    ax.plot(thresholds, counts_permuted, 'r--', linewidth=2, label=f'Permuted data (avg of {n_permutations})')
    ax.set_xlabel('G Factor Cutoff', fontsize=12)
    ax.set_ylabel('Number of Genes Detected', fontsize=12)
    ax.set_title('Genes Detected vs G Factor Cutoff', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Part2_genes_detected.png', dpi=150)
    plt.close()
    print("Plot saved as 'Part2_genes_detected.png'")

    # -------------------------------------------------------------------------
    # Steps 6-7: Calculate and plot true positive rate
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Steps 6-7: Calculating true positive rates")
    print("=" * 70)

    with np.errstate(divide='ignore', invalid='ignore'):
        tp_rate = (counts_real - counts_permuted) / counts_real
        tp_rate = np.where(counts_real == 0, 0, tp_rate)
        tp_rate = np.clip(tp_rate, 0, 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, tp_rate, 'g-', linewidth=2)
    # ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% true positive rate')
    ax.set_xlabel('G Factor Cutoff', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('True Positive Rate vs G Factor Cutoff', fontsize=14)
    ax.set_ylim(0, 1.05)
    # ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Part2_true_positive_rate.png', dpi=150)
    plt.close()
    print("Plot saved as 'Part2_true_positive_rate.png'")


 # -------------------------------------------------------------------------
    # Analysis for Part 3: True positive rate
 # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Analysis for Part 3: True positive rate")
    print("=" * 70) 
    
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

    # Select genes for Part 3 (those with G >= cutoff) and save expression matrix
    circadian_indices = np.where(g_factors_real >= g_cutoff_80)[0]
    circadian_gene_names = gene_symbols[circadian_indices]
    circadian_expression = data_matrix[circadian_indices, :]
    circadian_expr_df = pd.DataFrame(
        circadian_expression,
        columns=time_cols,
        index=circadian_gene_names
    )
    circadian_expr_df.to_csv('circadian_genes_expression.csv')
    print(f"Circadian gene expression data saved to 'circadian_genes_expression.csv'")

   

