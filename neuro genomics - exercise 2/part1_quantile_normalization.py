"""
Neuro-Genomics Exercise 2 - Part 1: Quantile Normalization

This module implements Quantile normalization for RNA-seq data
as described in the exercise instructions.

DO NOT use built-in normalization functions - this is a custom implementation.
"""

import numpy as np
import matplotlib.pyplot as plt


def quantile_normalize(X: np.ndarray, Y: np.ndarray) -> tuple:
    """
    Perform quantile normalization on two expression vectors.
    
    Quantile normalization forces the distributions of two (or more) vectors
    to be identical by replacing values with the average of the corresponding
    quantile values.
    
    Parameters
    ----------
    X : np.ndarray
        First expression vector (reference)
    Y : np.ndarray
        Second expression vector
    
    Returns
    -------
    tuple
        (X_corrected, Y_corrected)
        Both vectors normalized to have the same distribution
    
    Notes
    -----
    Quantile normalization algorithm:
    1. Sort each vector independently
    2. Calculate the mean across vectors for each rank position
    3. Assign the mean values back to the original positions
    
    Assumption: Each vector contains only unique values (as per instructions)
    """
    # Ensure inputs are numpy arrays
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    
    # Validate inputs
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length")
    
    n = len(X)
    
    # Step 1: Get the ranks (sorted order) for each vector
    # argsort gives the indices that would sort the array
    X_rank_indices = np.argsort(X)  # Indices to sort X
    Y_rank_indices = np.argsort(Y)  # Indices to sort Y
    
    # Get sorted values
    X_sorted = X[X_rank_indices]
    Y_sorted = Y[Y_rank_indices]
    
    # Step 2: Calculate the mean value at each rank position
    # This creates the "target distribution"
    mean_values = (X_sorted + Y_sorted) / 2
    
    # Step 3: Assign the mean values back to original positions
    # We need to "unsort" - put mean values back in original order
    
    # Create output arrays
    X_corrected = np.zeros(n)
    Y_corrected = np.zeros(n)
    
    # For X: the i-th smallest value in X should be replaced by mean_values[i]
    # X_rank_indices[i] is the original position of the i-th smallest value
    X_corrected[X_rank_indices] = mean_values
    
    # For Y: same logic
    Y_corrected[Y_rank_indices] = mean_values
    
    # Generate plots
    _plot_histograms(X, Y, X_corrected, Y_corrected)
    
    return X_corrected, Y_corrected


def _plot_histograms(X: np.ndarray, Y: np.ndarray, 
                     X_corrected: np.ndarray, Y_corrected: np.ndarray):
    """
    Plot histograms before and after quantile normalization.
    
    Parameters
    ----------
    X : np.ndarray
        Original X vector
    Y : np.ndarray
        Original Y vector
    X_corrected : np.ndarray
        Normalized X vector
    Y_corrected : np.ndarray
        Normalized Y vector
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Filter out zeros for better visualization (log scale)
    X_nonzero = X[X > 0]
    Y_nonzero = Y[Y > 0]
    X_corr_nonzero = X_corrected[X_corrected > 0]
    Y_corr_nonzero = Y_corrected[Y_corrected > 0]
    
    # Use log-transformed values for better visualization
    log_X = np.log10(X_nonzero + 1)
    log_Y = np.log10(Y_nonzero + 1)
    log_X_corr = np.log10(X_corr_nonzero + 1)
    log_Y_corr = np.log10(Y_corr_nonzero + 1)
    
    # Determine common bin edges
    all_values = np.concatenate([log_X, log_Y, log_X_corr, log_Y_corr])
    bins = np.linspace(np.min(all_values), np.max(all_values), 50)
    
    # Plot 1: Before normalization
    axes[0].hist(log_X, bins=bins, alpha=0.6, color='blue', label='X (original)', density=True)
    axes[0].hist(log_Y, bins=bins, alpha=0.6, color='red', label='Y (original)', density=True)
    axes[0].set_xlabel('Log10(Expression + 1)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Vectors histogram before normalization')
    axes[0].legend()
    
    # Plot 2: After normalization
    axes[1].hist(log_X_corr, bins=bins, alpha=0.6, color='blue', label="X' (normalized)", density=True)
    axes[1].hist(log_Y_corr, bins=bins, alpha=0.6, color='green', label="Y' (normalized)", density=True)
    axes[1].set_xlabel('Log10(Expression + 1)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Vectors histogram after normalization')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('Part1_Quantile_normalization.png', dpi=150)
    plt.close()
    print("Plot saved as 'Part1_Quantile_normalization.png'")


# =============================================================================
# Main execution - Run on pasilla data
# =============================================================================

if __name__ == "__main__":
    import pandas as pd
    
    print("=" * 70)
    print("Part 1: Quantile Normalization")
    print("=" * 70)
    
    # Load pasilla data
    pasilla_url = "neuro genomics - exercise 2/pasilla_gene_counts.tsv"
    
    try:
        cts = pd.read_csv(pasilla_url, sep='\t', index_col='gene_id')
        print("Loaded pasilla data from URL")
    except:
        try:
            cts = pd.read_csv("pasilla_gene_counts.tsv", sep='\t', index_col='gene_id')
            print("Loaded pasilla data from local file")
        except:
            print("Error: Could not load pasilla data.")
            exit(1)
    
    print(f"\nData shape: {cts.shape}")
    print(f"Samples: {list(cts.columns)}")
    
    # Select the SAME two vectors as in TMM normalization
    # Using 'untreated1' as X and 'treated1' as Y
    X = cts['untreated1'].values
    Y = cts['treated1'].values
    
    print(f"\nVector X: untreated1")
    print(f"Vector Y: treated1")
    print(f"Number of genes: {len(X)}")
    
    # Run quantile normalization
    print(f"\nRunning Quantile normalization...")
    
    X_corrected, Y_corrected = quantile_normalize(X, Y)
    
    print(f"\nResults:")
    
    # Compare sums before and after
    print(f"\nSum comparison:")
    print(f"  Sum of X (original):      {np.sum(X):,.0f}")
    print(f"  Sum of Y (original):      {np.sum(Y):,.0f}")
    print(f"  Sum of X' (normalized):   {np.sum(X_corrected):,.0f}")
    print(f"  Sum of Y' (normalized):   {np.sum(Y_corrected):,.0f}")
    
    # Verify distributions are the same
    print(f"\nDistribution comparison (should be identical after normalization):")
    print(f"  Mean of X':  {np.mean(X_corrected):.2f}")
    print(f"  Mean of Y':  {np.mean(Y_corrected):.2f}")
    print(f"  Std of X':   {np.std(X_corrected):.2f}")
    print(f"  Std of Y':   {np.std(Y_corrected):.2f}")
    
    # Show first few values
    print(f"\nFirst 5 gene values comparison:")
    print(f"{'Gene':<15} {'X (orig)':<12} {'Y (orig)':<12} {'X (norm)':<12} {'Y (norm)':<12}")
    print("-" * 63)
    for i, gene in enumerate(cts.index[:5]):
        print(f"{gene:<15} {X[i]:<12.1f} {Y[i]:<12.1f} {X_corrected[i]:<12.1f} {Y_corrected[i]:<12.1f}")

