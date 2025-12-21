"""
Neuro-Genomics Exercise 2 - Part 3: Hierarchical Clustering

This module implements hierarchical clustering using average linkage (UPGMA).
DO NOT use built-in clustering functions - this is a custom implementation.

The only built-in function allowed is for Pearson correlation calculation.
"""

import numpy as np
import pandas as pd


def hierarchical_clustering(expression_matrix: np.ndarray, 
                           gene_names: list, 
                           K: int) -> dict:
    """
    Perform hierarchical clustering on gene expression data.
    
    Parameters
    ----------
    expression_matrix : np.ndarray
        Matrix where each row is a gene and each column is an experimental condition.
        Shape: (N genes, M conditions)
    gene_names : list
        List of gene names, in the same order as rows in the matrix.
        Length: N
    K : int
        Desired number of clusters.
    
    Returns
    -------
    dict
        Dictionary mapping cluster ID to list of gene names in that cluster.
    
    Raises
    ------
    ValueError
        If K >= N (number of genes)
    
    Notes
    -----
    Algorithm (Average Linkage / UPGMA):
    
    1. Preparation: Compute distance matrix d using Pearson correlation.
       d_ij = 1 - R_ij (where R is Pearson correlation)
       
    2. Iteration:
       a. Find the two closest clusters i and j
       b. Merge them into a new cluster (ij)
       c. Update distances using weighted average:
          D_(ij),k = (n_i * D_i,k + n_j * D_j,k) / (n_i + n_j)
       d. Repeat until K clusters remain
    """
    N = len(gene_names)
    
    # Validate inputs
    if K >= N:
        raise ValueError(f"Error: K ({K}) must be less than N ({N})")
    if K < 1:
        raise ValueError(f"Error: K must be at least 1")
    if expression_matrix.shape[0] != N:
        raise ValueError(f"Error: Matrix rows ({expression_matrix.shape[0]}) must match gene_names length ({N})")
    
    print(f"Starting hierarchical clustering...")
    print(f"  Number of genes (N): {N}")
    print(f"  Number of conditions: {expression_matrix.shape[1]}")
    print(f"  Target clusters (K): {K}")
    
    # =========================================================================
    # PREPARATION: Compute initial distance matrix using Pearson correlation
    # =========================================================================
    print("\nComputing distance matrix using Pearson correlation...")
    
    # Initialize distance matrix
    d = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i + 1, N):
            # Compute Pearson correlation between gene i and gene j
            # Using numpy's corrcoef (allowed per instructions)
            corr_matrix = np.corrcoef(expression_matrix[i, :], expression_matrix[j, :])
            R = corr_matrix[0, 1]
            
            # Handle NaN (can occur if variance is zero)
            if np.isnan(R):
                R = 0
            
            # Distance = 1 - R
            # R = 1 (identical) -> d = 0
            # R = 0 (uncorrelated) -> d = 1
            # R = -1 (anti-correlated) -> d = 2
            distance = 1 - R
            
            d[i, j] = distance
            d[j, i] = distance
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    # Each gene starts as its own cluster
    # clusters[cluster_id] = list of gene indices in that cluster
    clusters = {i: [i] for i in range(N)}
    
    # cluster_sizes[cluster_id] = number of genes in cluster
    cluster_sizes = {i: 1 for i in range(N)}
    
    # Current distance matrix D (initially same as d)
    # D will be updated as clusters merge
    # We'll use a dictionary to track distances between active clusters
    D = {}
    active_clusters = set(range(N))
    
    for i in range(N):
        for j in range(i + 1, N):
            D[(i, j)] = d[i, j]
            D[(j, i)] = d[j, i]
    
    # Track the original distance matrix for UPGMA calculation
    original_d = d.copy()
    
    # Next available cluster ID for merged clusters
    next_cluster_id = N
    
    # =========================================================================
    # ITERATION: Merge clusters until K remain
    # =========================================================================
    
    n_iterations = N - K
    print(f"\nPerforming {n_iterations} merge iterations...")
    
    for iteration in range(n_iterations):
        if (iteration + 1) % 50 == 0 or iteration == 0:
            print(f"  Iteration {iteration + 1}/{n_iterations}, clusters remaining: {len(active_clusters)}")
        
        # Step 1: Find the two closest clusters
        min_dist = np.inf
        merge_i, merge_j = None, None
        
        for i in active_clusters:
            for j in active_clusters:
                if i >= j:
                    continue
                
                key = (i, j) if (i, j) in D else (j, i)
                if key in D and D[key] < min_dist:
                    min_dist = D[key]
                    merge_i, merge_j = i, j
        
        if merge_i is None:
            print(f"Warning: Could not find clusters to merge at iteration {iteration}")
            break
        
        # Step 2: Create new merged cluster
        new_cluster_id = next_cluster_id
        next_cluster_id += 1
        
        # Combine gene indices from both clusters
        clusters[new_cluster_id] = clusters[merge_i] + clusters[merge_j]
        cluster_sizes[new_cluster_id] = cluster_sizes[merge_i] + cluster_sizes[merge_j]
        
        # Step 3: Compute distances from new cluster to all other clusters
        n_i = cluster_sizes[merge_i]
        n_j = cluster_sizes[merge_j]
        
        for k in active_clusters:
            if k == merge_i or k == merge_j:
                continue
            
            # Get distances D_i,k and D_j,k
            key_ik = (merge_i, k) if (merge_i, k) in D else (k, merge_i)
            key_jk = (merge_j, k) if (merge_j, k) in D else (k, merge_j)
            
            D_ik = D.get(key_ik, 0)
            D_jk = D.get(key_jk, 0)
            
            # Weighted average (UPGMA formula)
            # D_(ij),k = (n_i * D_i,k + n_j * D_j,k) / (n_i + n_j)
            D_new_k = (n_i * D_ik + n_j * D_jk) / (n_i + n_j)
            
            D[(new_cluster_id, k)] = D_new_k
            D[(k, new_cluster_id)] = D_new_k
        
        # Step 4: Remove old clusters and add new one
        active_clusters.remove(merge_i)
        active_clusters.remove(merge_j)
        active_clusters.add(new_cluster_id)
        
        # Clean up old cluster entries (optional, for memory efficiency)
        del clusters[merge_i]
        del clusters[merge_j]
        del cluster_sizes[merge_i]
        del cluster_sizes[merge_j]
    
    # =========================================================================
    # OUTPUT: Create result dictionary with gene names
    # =========================================================================
    
    print(f"\nClustering complete. {len(active_clusters)} clusters formed.")
    
    result = {}
    for idx, cluster_id in enumerate(sorted(active_clusters)):
        gene_indices = clusters[cluster_id]
        cluster_gene_names = [gene_names[i] for i in gene_indices]
        result[f"Cluster_{idx + 1}"] = cluster_gene_names
    
    # Print results
    print("\n" + "=" * 70)
    print(f"CLUSTERING RESULTS (K = {K} clusters)")
    print("=" * 70)
    
    for cluster_name, genes in result.items():
        print(f"\n{cluster_name} ({len(genes)} genes):")
        print("-" * 40)
        for gene in genes:
            print(f"  - {gene}")
    
    return result


# =============================================================================
# Main execution - Test on circadian genes
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Part 3: Hierarchical Clustering")
    print("=" * 70)
    
    # Try to load circadian genes from Part 2 output
    try:
        # Load expression data
        expr_df = pd.read_csv('circadian_genes_expression.csv', index_col=0)
        print(f"Loaded circadian gene expression data: {expr_df.shape}")
        
        expression_matrix = expr_df.values
        gene_names = list(expr_df.index)
        
    except FileNotFoundError:
        print("Circadian gene data not found. Running Part 2 first...")
        print("Please run part2_circadian_permutation.py first to generate the data.")
        
        # Create sample data for testing
        print("\nUsing sample data for demonstration...")
        np.random.seed(42)
        
        # Create 20 genes with 12 time points
        n_genes = 20
        n_timepoints = 12
        
        # Create some correlated patterns
        base_pattern = np.sin(np.linspace(0, 4*np.pi, n_timepoints))
        
        expression_matrix = np.zeros((n_genes, n_timepoints))
        gene_names = [f"Gene_{i+1}" for i in range(n_genes)]
        
        for i in range(n_genes):
            # Add variation to base pattern
            phase_shift = np.random.uniform(0, np.pi)
            amplitude = np.random.uniform(0.5, 2)
            noise = np.random.normal(0, 0.2, n_timepoints)
            
            expression_matrix[i, :] = amplitude * np.sin(np.linspace(0, 4*np.pi, n_timepoints) + phase_shift) + noise + 5
    
    print(f"\nExpression matrix shape: {expression_matrix.shape}")
    print(f"Number of genes: {len(gene_names)}")
    
    # Run hierarchical clustering with K=6
    K = 6
    print(f"\nRunning hierarchical clustering with K = {K}...")
    
    try:
        result = hierarchical_clustering(expression_matrix, gene_names, K)
        
        # Save results
        print("\n" + "=" * 70)
        print("Saving results...")
        print("=" * 70)
        
        # Save to file
        with open('Part3_clustering_results.txt', 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"HIERARCHICAL CLUSTERING RESULTS (K = {K})\n")
            f.write("=" * 70 + "\n\n")
            
            for cluster_name, genes in result.items():
                f.write(f"\n{cluster_name} ({len(genes)} genes):\n")
                f.write("-" * 40 + "\n")
                for gene in genes:
                    f.write(f"  - {gene}\n")
        
        print("Results saved to 'Part3_clustering_results.txt'")
        
        # Also save as CSV for easier processing
        rows = []
        for cluster_name, genes in result.items():
            for gene in genes:
                rows.append({'Cluster': cluster_name, 'Gene': gene})
        
        result_df = pd.DataFrame(rows)
        result_df.to_csv('Part3_clustering_results.csv', index=False)
        print("Results also saved to 'Part3_clustering_results.csv'")
        
    except ValueError as e:
        print(f"Error: {e}")

