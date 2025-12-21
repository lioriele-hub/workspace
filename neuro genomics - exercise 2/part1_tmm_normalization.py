"""
Neuro-Genomics Exercise 2 - Part 1: TMM Normalization

This module implements TMM (Trimmed Mean of M-values) normalization
for RNA-seq data as described in the exercise instructions.

DO NOT use built-in normalization functions - this is a custom implementation.
"""

import numpy as np
import matplotlib.pyplot as plt


def tmm_normalize(X: np.ndarray, Y: np.ndarray, D: float = 0.3) -> tuple:
    """
    Perform TMM normalization on expression vector Y using X as reference.
    
    Parameters
    ----------
    X : np.ndarray
        Reference expression vector (will not be modified)
    Y : np.ndarray
        Expression vector to be normalized
    D : float
        Trimming parameter (0 to 1). D/2 fraction trimmed from each end.
        Default is 0.3 (30% total trimming, 15% from each tail)
    
    Returns
    -------
    tuple
        (X, Y_corrected, correction_factor)
        - X: original reference vector (unchanged)
        - Y_corrected: normalized Y vector
        - correction_factor: the factor used for correction (exp(TMM))
    
    Notes
    -----
    TMM normalization steps:
    1. Compute M-values (log fold changes): M_i = log(Y_i / X_i)
    2. Compute weights based on Poisson variance: W_i = sqrt((X_i + Y_i) / 2)
    3. Trim the extreme D/2 fraction from each end of M-values
    4. Compute weighted mean of trimmed M-values
    5. Correct Y by dividing by exp(TMM)
    """
    # Ensure inputs are numpy arrays
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    
    # Validate inputs
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length")
    if not 0 <= D <= 1:
        raise ValueError("D must be between 0 and 1")
    
    # Step 2: Compute M folds (log fold changes)
    # Only compute for genes where both X_i and Y_i are > 0
    valid_mask = (X > 0) & (Y > 0)
    
    X_valid = X[valid_mask]
    Y_valid = Y[valid_mask]
    
    # M_i = log(Y_i / X_i) using natural logarithm
    M = np.log(Y_valid / X_valid)
    
    # Step 3: Compute weights based on Poisson assumption
    # W_i = sqrt((X_i + Y_i) / 2)
    W = np.sqrt((X_valid + Y_valid) / 2)
    
    # Step 4: Sort M folds and trim D/2 from each end
    n_valid = len(M)
    sorted_indices = np.argsort(M)
    
    # Calculate how many to trim from each end
    trim_count = int(np.floor(n_valid * D / 2))
    
    # Keep only the middle portion after trimming
    if trim_count > 0:
        keep_indices = sorted_indices[trim_count:-trim_count]
    else:
        keep_indices = sorted_indices
    
    # Get trimmed M values and corresponding weights
    M_trimmed = M[keep_indices]
    W_trimmed = W[keep_indices]
    
    # Step 5: Compute TMM as weighted mean of trimmed M values
    # TMM = sum(W_i * M_i) / sum(W_i)
    TMM = np.sum(W_trimmed * M_trimmed) / np.sum(W_trimmed)
    
    # Step 6: Compute correction factor
    correction_factor = np.exp(TMM)
    
    # Step 7: Correct Y by dividing by the factor
    Y_corrected = Y / correction_factor
    
    # Step 8: Generate plots
    _plot_histograms(X, Y, Y_corrected)
    
    return X, Y_corrected, correction_factor


def _plot_histograms(X: np.ndarray, Y: np.ndarray, Y_corrected: np.ndarray):
    """
    Plot histograms before and after normalization.
    
    Parameters
    ----------
    X : np.ndarray
        Reference vector
    Y : np.ndarray
        Original vector (before normalization)
    Y_corrected : np.ndarray
        Corrected vector (after normalization)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Filter out zeros for better visualization (log scale)
    X_nonzero = X[X > 0]
    Y_nonzero = Y[Y > 0]
    Y_corr_nonzero = Y_corrected[Y_corrected > 0]
    
    # Use log-transformed values for better visualization
    log_X = np.log10(X_nonzero + 1)
    log_Y = np.log10(Y_nonzero + 1)
    log_Y_corr = np.log10(Y_corr_nonzero + 1)
    
    # Determine common bin edges
    all_values = np.concatenate([log_X, log_Y, log_Y_corr])
    bins = np.linspace(np.min(all_values), np.max(all_values), 50)
    
    # Plot 1: Before normalization
    axes[0].hist(log_X, bins=bins, alpha=0.6, color='blue', label='X (reference)', density=True)
    axes[0].hist(log_Y, bins=bins, alpha=0.6, color='red', label='Y (original)', density=True)
    axes[0].set_xlabel('Log10(Expression + 1)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Vectors histogram before normalization')
    axes[0].legend()
    
    # Plot 2: After normalization
    axes[1].hist(log_X, bins=bins, alpha=0.6, color='blue', label='X (reference)', density=True)
    axes[1].hist(log_Y_corr, bins=bins, alpha=0.6, color='green', label="Y' (normalized)", density=True)
    axes[1].set_xlabel('Log10(Expression + 1)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Vectors histogram after normalization')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('Part1_TMM_normalization.png', dpi=150)
    plt.close()
    print("Plot saved as 'Part1_TMM_normalization.png'")


# =============================================================================
# Main execution - Run on pasilla data
# =============================================================================

if __name__ == "__main__":
    import pandas as pd
    
    print("=" * 70)
    print("Part 1: TMM Normalization")
    print("=" * 70)
    
    # Load pasilla data
    # Try to load from URL first, then from local file
    pasilla_url = "neuro genomics - exercise 2/pasilla_gene_counts.tsv"
    
    try:
        cts = pd.read_csv(pasilla_url, sep='\t', index_col='gene_id')
        print("Loaded pasilla data from URL")
    except:
        # Try local file
        try:
            cts = pd.read_csv("pasilla_gene_counts.tsv", sep='\t', index_col='gene_id')
            print("Loaded pasilla data from local file")
        except:
            print("Error: Could not load pasilla data.")
            print("Please download from: https://github.com/Bioconductor/pasilla")
            exit(1)
    
    print(f"\nData shape: {cts.shape}")
    print(f"Samples: {list(cts.columns)}")
    
    # Select two vectors for normalization
    # Using 'untreated1' as reference (X) and 'treated1' as the vector to normalize (Y)
    X = cts['untreated1'].values
    Y = cts['treated1'].values
    
    print(f"\nReference vector (X): untreated1")
    print(f"Vector to normalize (Y): treated1")
    print(f"Number of genes: {len(X)}")
    
    # Run TMM normalization with D=0.3 (30% trimming)
    D = 0.3
    print(f"\nRunning TMM normalization with D = {D}")
    
    X_out, Y_corrected, factor = tmm_normalize(X, Y, D=D)
    
    print(f"\nResults:")
    print(f"  Correction factor (exp(TMM)): {factor:.6f}")
    print(f"  TMM value: {np.log(factor):.6f}")
    
    # Compare sums before and after
    print(f"\nSum comparison:")
    print(f"  Sum of X (reference):     {np.sum(X):,.0f}")
    print(f"  Sum of Y (original):      {np.sum(Y):,.0f}")
    print(f"  Sum of Y' (normalized):   {np.sum(Y_corrected):,.0f}")
    
    # Show first few values
    print(f"\nFirst 5 gene values comparison:")
    print(f"{'Gene':<15} {'X':<12} {'Y (orig)':<12} {'Y (norm)':<12}")
    print("-" * 51)
    for i, gene in enumerate(cts.index[:5]):
        print(f"{gene:<15} {X[i]:<12.1f} {Y[i]:<12.1f} {Y_corrected[i]:<12.1f}")

