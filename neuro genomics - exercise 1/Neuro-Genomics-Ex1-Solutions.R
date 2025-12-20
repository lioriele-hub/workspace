# =============================================================================
# Neuro-Genomics Exercise 1 - Solutions
# Principles of Sequencing Data Analysis
# =============================================================================

# =============================================================================
# PART 1 - General Introduction to Sequencing Data
# =============================================================================

# QUESTION: Can you explain why we can only get an estimation of the expression 
# levels and not the actual number of RNA molecules for each gene?
#
# ANSWER:
# We can only get an estimation of expression levels (not actual RNA molecule 
# counts) for several reasons related to PCR amplification during sequencing:
#
# 1. PCR Bias: Different sequences have different amplification efficiencies.
#    Some sequences are preferentially amplified over others due to their
#    GC content, secondary structure, or other sequence features. This means
#    the final read counts don't proportionally reflect the original RNA amounts.
#
# 2. Stochastic Amplification of Low Copy Number: When starting material is
#    limited (low copy number), PCR amplification introduces random variation.
#    Some molecules may be amplified more than others by chance, leading to
#    distorted representation of the original proportions.
#
# 3. Bridge Amplification Variability: During Illumina sequencing, bridge
#    amplification creates clusters on the flow cell. The efficiency of cluster
#    formation can vary, affecting read counts.
#
# 4. Library Preparation Losses: Not all RNA molecules are successfully
#    converted to cDNA and incorporated into the sequencing library.
#
# Therefore, read counts provide a relative measure of expression, not absolute
# molecule counts.

# =============================================================================
# PART 2 - Explore Sequencing Data Using R
# =============================================================================

# Install and load the pasilla package
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

# Install pasilla if not already installed
if (!requireNamespace("pasilla", quietly = TRUE))
  BiocManager::install("pasilla")

library("pasilla")

# Load the actual data from the pasilla package
pasCts <- system.file("extdata",
                      "pasilla_gene_counts.tsv",
                      package="pasilla", mustWork=TRUE)
pasAnno <- system.file("extdata",
                       "pasilla_sample_annotation.csv",
                       package="pasilla", mustWork=TRUE)
cts <- as.matrix(read.csv(pasCts, sep="\t", row.names="gene_id"))
coldata <- read.csv(pasAnno, row.names=1)
coldata <- coldata[,c("condition","type")]
rownames(coldata) <- sub("fb", "", rownames(coldata))
cts <- cts[, rownames(coldata)]

# -----------------------------------------------------------------------------
# TASK: Examine the first 10 lines of the cts matrix
# -----------------------------------------------------------------------------
cat("First 10 lines of the cts matrix:\n")
print(head(cts, 10))

# ANSWER: The matrix contains gene expression count data where:
# - Rows represent different genes (identified by FlyBase gene IDs like FBgn0000003)
# - Columns represent different samples (treated1, treated2, treated3, untreated1, etc.)
# - Each cell contains the number of sequencing reads aligned to that gene in that sample

# -----------------------------------------------------------------------------
# TASK: Define a variable to hold matrix dimensions and print them
# -----------------------------------------------------------------------------
dims <- dim(cts)
cat("\nMatrix dimensions:\n")
cat("Number of rows (genes):", dims[1], "\n")
cat("Number of columns (samples):", dims[2], "\n")

# -----------------------------------------------------------------------------
# TASK: Is the sum of reads the same for each sample?
# -----------------------------------------------------------------------------
sample_sums <- colSums(cts)
cat("\nSum of reads per sample:\n")
print(sample_sums)

# ANSWER: No, the sum of reads is NOT the same for each sample.
# Different samples have different total read counts, which is why normalization
# is necessary for valid comparison between samples.

# -----------------------------------------------------------------------------
# TASK: Create a normalized version of the cts matrix
# -----------------------------------------------------------------------------
# Multiply each column by a factor so that total reads equal the first column

# Get the target sum (first column's sum)
target_sum <- sample_sums[1]

# Calculate normalization factors for each column
normalization_factors <- target_sum / sample_sums

# Create normalized matrix
cts_normalized <- sweep(cts, 2, normalization_factors, "*")

# Alternative method (more explicit):
# cts_normalized <- cts
# for(i in 1:ncol(cts)) {
#   cts_normalized[,i] <- cts[,i] * (target_sum / sample_sums[i])
# }

# -----------------------------------------------------------------------------
# TASK: Verify that normalized sums are equal
# -----------------------------------------------------------------------------
normalized_sums <- colSums(cts_normalized)
cat("\nSum of reads per sample after normalization:\n")
print(normalized_sums)

# ANSWER: All samples now have the same total read count (equal to the first sample)

# =============================================================================
# PART 3 - Basic Statistics of Sequencing Data
# =============================================================================

# -----------------------------------------------------------------------------
# TASK: Create scatter plot of mean vs variance (log-log scale)
# -----------------------------------------------------------------------------

# Calculate mean and variance for each gene across all samples
gene_means <- apply(cts_normalized, 1, mean)
gene_vars <- apply(cts_normalized, 1, var)

# Log transform (adding 1 to avoid log(0))
log_means <- log10(gene_means + 1)
log_vars <- log10(gene_vars + 1)

# Create the scatter plot
png("Part3_mean_vs_variance.png", width=800, height=600)
plot(log_means, log_vars,
     xlab = "Log10(Mean Expression + 1)",
     ylab = "Log10(Variance + 1)",
     main = "Mean vs Variance of Gene Expression (Log-Log Scale)",
     pch = 20, col = rgb(0, 0, 1, 0.3))
abline(0, 1, col = "red", lty = 2, lwd = 2)  # y = x line for reference
legend("topleft", "y = x", col = "red", lty = 2)
dev.off()

cat("\nPart 3: Scatter plot saved as 'Part3_mean_vs_variance.png'\n")

# ANSWER: Is this a Poisson distribution?
# No, the data does NOT follow a Poisson distribution. In a Poisson distribution,
# the variance equals the mean. Looking at the scatter plot, the variance is 
# consistently HIGHER than the mean (points lie above the y=x line).
# This is called "overdispersion" and is typical of RNA-seq data.
# The data better fits a Negative Binomial distribution, which has an additional
# dispersion parameter to account for this extra variance.

# -----------------------------------------------------------------------------
# TASK: Can we detect differentially expressed genes with one sample per condition?
# -----------------------------------------------------------------------------
# ANSWER: No, we cannot reliably detect differentially expressed genes with only
# one sample per experimental condition. This is because:
# 1. We cannot estimate within-group variance (biological + technical noise)
# 2. We cannot distinguish true biological differences from random variation
# 3. Statistical tests require replicates to estimate variability
# 4. Any observed difference could be due to chance, technical artifacts, or
#    biological noise rather than the treatment effect

# =============================================================================
# PART 4 - Detecting Differentially Expressed Genes
# =============================================================================

# Install and load DESeq2 if not already installed
if (!requireNamespace("DESeq2", quietly = TRUE))
  BiocManager::install("DESeq2")

library("DESeq2")

# -----------------------------------------------------------------------------
# TASK: Create DESeqDataSet object and run differential expression analysis
# -----------------------------------------------------------------------------

# Create DESeqDataSet from matrix
dds <- DESeqDataSetFromMatrix(countData = cts,
                              colData = coldata,
                              design = ~ condition)

# Run the DESeq2 analysis pipeline
dds <- DESeq(dds)

# Get results
res <- results(dds)

cat("\nDESeq2 Results Summary:\n")
summary(res)

# -----------------------------------------------------------------------------
# TASK: Create MA plot
# -----------------------------------------------------------------------------
png("Part4_MA_plot.png", width=800, height=600)
plotMA(res, main = "MA Plot: Treated vs Untreated", ylim = c(-5, 5))
dev.off()

cat("\nPart 4: MA plot saved as 'Part4_MA_plot.png'\n")

# ANSWER: What is the meaning of the blue points?
# The blue points represent genes that are statistically significantly 
# differentially expressed (adjusted p-value < 0.1 by default in DESeq2).
# These are genes where the expression change between treated and untreated
# conditions is unlikely to be due to random chance.

# -----------------------------------------------------------------------------
# TASK: Examine specific genes
# -----------------------------------------------------------------------------

# Check the status of specific genes
genes_to_check <- c("FBgn0039155", "FBgn0025111", "FBgn0029167")

cat("\nResults for specific genes:\n")
for (gene in genes_to_check) {
  if (gene %in% rownames(res)) {
    gene_res <- res[gene, ]
    cat("\n", gene, ":\n")
    cat("  Log2 Fold Change:", round(gene_res$log2FoldChange, 3), "\n")
    cat("  Adjusted p-value:", format(gene_res$padj, scientific = TRUE), "\n")
    cat("  Differentially expressed:", ifelse(gene_res$padj < 0.05, "YES", "NO"), "\n")
  }
}

# ANSWER: 
# FBgn0039155 - This gene shows significant differential expression (pasilla itself)
# FBgn0025111 - Check the adjusted p-value to determine significance
# FBgn0029167 - Check the adjusted p-value to determine significance
# Genes with adjusted p-value < 0.05 (or 0.1) are considered differentially expressed

# =============================================================================
# PART 5 - Detecting Circadian Patterns Using FFT
# =============================================================================

# -----------------------------------------------------------------------------
# TASK: Load circadian data
# -----------------------------------------------------------------------------

# Note: Download CircadianRNAseq.csv from the course website first
# For this solution, we'll check if the file exists

circadian_file <- "CircadianRNAseq.csv"

if (file.exists(circadian_file)) {
  circadian_data <- as.matrix(read.csv(circadian_file, row.names = 1))
  
  cat("\nCircadian data loaded successfully.\n")
  
  # Examine last 5 rows
  cat("\nLast 5 rows of the circadian matrix:\n")
  print(tail(circadian_data, 5))
  
  # ANSWER: Time step analysis
  # Looking at column names: A_11PM, A_3AM, A_7AM, A_11AM, A_3PM, A_7PM, 
  #                         B_11PM, B_3AM, B_7AM, B_11AM, B_3PM, B_7PM
  # The time step between measurements is 4 hours (e.g., 11PM to 3AM = 4 hours)
  
  # -----------------------------------------------------------------------------
  # TASK: Plot expression of gene 'per1a'
  # -----------------------------------------------------------------------------
  
  # Find the gene 'per1a'
  gene_col <- ncol(circadian_data)  # GeneSymbol is last column
  per1a_idx <- which(circadian_data[, gene_col] == "per1a")
  
  if (length(per1a_idx) > 0) {
    # Get expression values (columns 2 to 13, which are the time points)
    # Note: Adjust indices based on actual data structure
    time_cols <- 1:(ncol(circadian_data) - 1)  # All columns except GeneSymbol
    
    per1a_expr <- as.numeric(circadian_data[per1a_idx[1], time_cols])
    time_labels <- colnames(circadian_data)[time_cols]
    
    # Plot
    png("Part5_per1a_expression.png", width=1000, height=600)
    plot(1:length(per1a_expr), per1a_expr, 
         type = "b", pch = 19, col = "blue",
         xaxt = "n",
         xlab = "Time Point",
         ylab = "Expression Level (normalized counts)",
         main = "Expression of per1a Gene Over Time")
    axis(1, at = 1:length(time_labels), labels = time_labels, las = 2)
    dev.off()
    
    cat("\nPart 5: per1a expression plot saved as 'Part5_per1a_expression.png'\n")
    
    # ANSWER: Does per1a seem circadian?
    # Yes, per1a should show circadian expression patterns. Per1 (Period 1) is a 
    # core clock gene that is part of the molecular circadian oscillator. It is
    # one of the best-known circadian genes, showing robust ~24-hour rhythms.
    
    # -----------------------------------------------------------------------------
    # TASK: Calculate FFT power spectrum for per1a
    # -----------------------------------------------------------------------------
    
    # Number of time points
    N <- length(per1a_expr)
    
    # Time step in hours
    delta_t <- 4
    
    # Compute FFT
    fft_result <- fft(per1a_expr)
    
    # Compute power (multiply by complex conjugate)
    power <- fft_result * Conj(fft_result)
    power <- Re(power)  # Convert to real numbers
    
    # The relevant powers are in positions 2:(N/2+1)
    # Position 1 is DC component (frequency 0), which we ignore
    # Positions 2 to N/2+1 represent the unique frequencies
    
    relevant_powers <- power[2:(N/2 + 1)]
    
    # Normalize powers
    normalized_powers <- relevant_powers / sum(relevant_powers)
    
    # Create frequency vector
    # Sampling rate: fs = 1/delta_t = 1/4 cycles per hour
    # Frequency resolution: 1/(N*delta_t) = 1/(12*4) = 1/48 cycles per hour
    # Frequencies: from 1/48 to (N/2)/(N*delta_t) = 6/48 = 1/8 (Nyquist)
    
    fs <- 1 / delta_t
    freq_resolution <- fs / N
    frequencies <- seq(from = freq_resolution, 
                       by = freq_resolution, 
                       length.out = N/2)
    
    # Convert to period in hours for easier interpretation
    periods <- 1 / frequencies
    
    # Plot power spectrum
    png("Part5_per1a_FFT_power.png", width=800, height=600)
    plot(frequencies, normalized_powers,
         type = "h", lwd = 3, col = "blue",
         xlab = "Frequency (cycles/hour)",
         ylab = "Normalized Power",
         main = "FFT Power Spectrum of per1a Expression")
    
    # Add text labels for period
    text(frequencies, normalized_powers + 0.02, 
         labels = paste0(round(periods, 1), "h"),
         cex = 0.8)
    
    # Highlight circadian frequency (1/24)
    abline(v = 1/24, col = "red", lty = 2)
    text(1/24, max(normalized_powers) * 0.9, "24h period", col = "red", pos = 4)
    dev.off()
    
    cat("Part 5: FFT power spectrum saved as 'Part5_per1a_FFT_power.png'\n")
    
    # ANSWER: The power spectrum clearly shows the highest power at frequency 1/24
    # (24-hour period), confirming circadian expression of per1a.
  }
  
  # -----------------------------------------------------------------------------
  # TASK: Which experimental design is better for detecting circadian genes?
  # -----------------------------------------------------------------------------
  
  # ANSWER:
  # Option (b) - increasing the number of days measured (4 days instead of 2) 
  # while keeping the time step the same is BETTER for detecting circadian frequency.
  #
  # Reasoning:
  # - Shorter time step (option a): This increases the Nyquist frequency (maximum
  #   detectable frequency), but doesn't improve resolution at the circadian 
  #   frequency. We can already detect 24h cycles with 4h sampling.
  #
  # - Longer measurement period (option b): This improves frequency RESOLUTION
  #   (frequency resolution = 1/(N*Δt)). With more days, we can better distinguish
  #   the circadian frequency from nearby frequencies, reducing spectral leakage
  #   and providing more precise detection of the 24h period. More cycles of the
  #   rhythm also provide more statistical power.
  
  # -----------------------------------------------------------------------------
  # TASK: Advantages of frequency domain analysis
  # -----------------------------------------------------------------------------
  
  # ANSWER:
  # Advantages of detecting circadian genes in frequency domain vs time domain:
  #
  # 1. Objective quantification: FFT provides a single numerical value (power at
  #    circadian frequency) that can be compared across genes, rather than
  #    subjective curve fitting.
  #
  # 2. Phase-independent: FFT detects periodicity regardless of phase. A cosine
  #    fit would require knowing or estimating the phase of the oscillation.
  #
  # 3. Noise robustness: FFT can detect periodic signals even in noisy data by
  #    concentrating power at the fundamental frequency.
  #
  # 4. No assumptions about waveform: FFT works for any periodic pattern, not
  #    just perfect cosines. Real biological rhythms may have non-sinusoidal
  #    waveforms.
  #
  # 5. Computational efficiency: FFT is O(N log N), making it efficient for
  #    analyzing thousands of genes.
  #
  # 6. Detection of multiple periodicities: FFT can reveal if a gene has
  #    multiple periodic components (e.g., 24h and 12h harmonics).
  
  # -----------------------------------------------------------------------------
  # TASK: Find top 10 genes with highest circadian power
  # -----------------------------------------------------------------------------
  
  # Get numerical data (exclude RefSeqID and GeneSymbol columns)
  # Adjust based on actual column structure
  num_cols <- 1:(ncol(circadian_data) - 1)
  
  # Function to calculate normalized circadian power for a gene
  calc_circadian_power <- function(expr_row) {
    expr <- as.numeric(expr_row)
    
    # Handle NA or all-zero rows
    if (all(is.na(expr)) || all(expr == 0)) {
      return(NA)
    }
    
    N <- length(expr)
    
    # FFT
    fft_result <- fft(expr)
    power <- Re(fft_result * Conj(fft_result))
    
    # Get relevant powers (exclude DC component)
    relevant_powers <- power[2:(N/2 + 1)]
    
    # Normalize
    total_power <- sum(relevant_powers)
    if (total_power == 0) return(NA)
    
    normalized_powers <- relevant_powers / total_power
    
    # Circadian frequency is 1/24 = 1/(N*delta_t) * k
    # With N=12 and delta_t=4, circadian frequency index k = N*delta_t / 24 = 48/24 = 2
    # So circadian power is at position 2 of relevant_powers (index 1 in 0-based)
    circadian_power <- normalized_powers[2]  # 24-hour component
    
    return(circadian_power)
  }
  
  # Calculate circadian power for all genes
  circadian_powers <- apply(circadian_data[, num_cols], 1, calc_circadian_power)
  
  # Sort by circadian power (descending) with NA last
  sorted_idx <- order(abs(circadian_powers), decreasing = TRUE, na.last = TRUE)
  
  # Get top 10 gene symbols
  gene_symbols <- circadian_data[, ncol(circadian_data)]
  top10_genes <- gene_symbols[sorted_idx[1:10]]
  top10_powers <- circadian_powers[sorted_idx[1:10]]
  
  cat("\nTop 10 genes with highest normalized circadian power:\n")
  for (i in 1:10) {
    cat(i, ".", top10_genes[i], "- Power:", round(top10_powers[i], 4), "\n")
  }
  
  # ANSWER: Known circadian genes that may appear in top 10:
  # - per1a, per1b, per2, per3 (Period genes)
  # - cry1a, cry1b, cry2 (Cryptochrome genes)
  # - clock, bmal (Clock genes)
  # - nr1d1, nr1d2 (Rev-erb genes)
  # - arntl (Bmal1)
  
} else {
  cat("\n*** CircadianRNAseq.csv not found. ***\n")
  cat("Please download the file from the course website and place it in the working directory.\n")
  cat("The file should be at:", getwd(), "/CircadianRNAseq.csv\n")
}

# =============================================================================
# PART 6 - Detecting Genes with Variable Expression Levels
# =============================================================================

if (file.exists(circadian_file)) {
  
  cat("\n\n=== PART 6: Variable Genes Detection ===\n")
  
  # -----------------------------------------------------------------------------
  # TASK: Identify variable genes using z-score binning method
  # -----------------------------------------------------------------------------
  
  # Create numerical matrix from count data
  num_cols <- 1:(ncol(circadian_data) - 1)
  count_matrix <- circadian_data[, num_cols]
  count_matrix <- matrix(as.numeric(count_matrix), 
                         nrow = nrow(circadian_data),
                         ncol = length(num_cols))
  
  # Calculate variance and mean for each gene
  gene_vars <- apply(count_matrix, 1, var, na.rm = TRUE)
  gene_means <- apply(count_matrix, 1, mean, na.rm = TRUE)
  
  # Log transform (add 1 before log)
  log_vars <- log(gene_vars + 1)
  log_means <- log(gene_means + 1)
  
  # Create data frame with original index for tracking
  gene_data <- data.frame(
    original_idx = 1:length(log_means),
    log_mean = log_means,
    log_var = log_vars,
    gene_symbol = circadian_data[, ncol(circadian_data)]
  )
  
  # Sort by mean expression (ascending)
  gene_data <- gene_data[order(gene_data$log_mean), ]
  
  # Filter genes with log_mean >= 3 (approximately mean expression >= exp(3)-1 ≈ 19)
  # Actually, log(mean+1) >= 3 means mean >= exp(3) - 1 ≈ 19
  gene_data_filtered <- gene_data[gene_data$log_mean >= 3, ]
  
  cat("Number of genes with log(mean+1) >= 3:", nrow(gene_data_filtered), "\n")
  
  # Bin into 20 groups based on mean expression
  n_bins <- 20
  gene_data_filtered$bin <- cut(gene_data_filtered$log_mean, 
                                breaks = n_bins, 
                                labels = FALSE)
  
  # Calculate z-score of variance within each bin
  gene_data_filtered$z_score <- NA
  
  for (b in 1:n_bins) {
    bin_idx <- which(gene_data_filtered$bin == b)
    if (length(bin_idx) > 1) {
      bin_vars <- gene_data_filtered$log_var[bin_idx]
      bin_mean_var <- mean(bin_vars, na.rm = TRUE)
      bin_sd_var <- sd(bin_vars, na.rm = TRUE)
      
      if (!is.na(bin_sd_var) && bin_sd_var > 0) {
        gene_data_filtered$z_score[bin_idx] <- (bin_vars - bin_mean_var) / bin_sd_var
      }
    }
  }
  
  # Sort by z-score (descending) with NA last
  gene_data_sorted <- gene_data_filtered[order(-gene_data_filtered$z_score, na.last = TRUE), ]
  
  # Get top 40 variable genes
  top40_variable <- gene_data_sorted[1:40, ]
  
  cat("\nTop 40 genes with highest variance z-scores:\n")
  cat("(These are genes with unusually high variance for their expression level)\n\n")
  
  for (i in 1:40) {
    cat(sprintf("%2d. %-15s z-score: %6.3f\n", 
                i, 
                as.character(top40_variable$gene_symbol[i]),
                top40_variable$z_score[i]))
  }
  
  # ANSWER: Are circadian genes expected to be variable genes?
  # Yes! Circadian genes are expected to be variable genes because:
  # 1. By definition, circadian genes show oscillating expression levels across
  #    the 24-hour cycle, which means high variance between time points.
  # 2. The amplitude of circadian oscillations contributes directly to variance.
  # 3. Therefore, genes with strong circadian rhythms will have higher-than-expected
  #    variance compared to non-rhythmic genes with similar mean expression.
  #
  # Known circadian genes expected in the top variable genes:
  # - per1a, per2, per3 (Period genes)
  # - cry1a, cry1b (Cryptochrome genes)
  # - bmal, clock (Core clock transcription factors)
  # - nr1d1 (Rev-erb alpha)
  
  # -----------------------------------------------------------------------------
  # Create visualization of variable genes detection
  # -----------------------------------------------------------------------------
  
  png("Part6_variable_genes.png", width=1000, height=800)
  par(mfrow = c(2, 2))
  
  # Plot 1: Mean vs Variance with bin coloring
  plot(gene_data_filtered$log_mean, gene_data_filtered$log_var,
       col = rainbow(n_bins)[gene_data_filtered$bin],
       pch = 20, cex = 0.5,
       xlab = "Log(Mean + 1)",
       ylab = "Log(Variance + 1)",
       main = "Mean vs Variance (colored by bin)")
  
  # Plot 2: Z-score distribution
  hist(gene_data_filtered$z_score, breaks = 50,
       main = "Distribution of Variance Z-scores",
       xlab = "Z-score",
       col = "lightblue")
  abline(v = gene_data_sorted$z_score[40], col = "red", lty = 2)
  legend("topright", "Top 40 threshold", col = "red", lty = 2)
  
  # Plot 3: Highlight top variable genes
  plot(gene_data_filtered$log_mean, gene_data_filtered$log_var,
       col = "gray80", pch = 20, cex = 0.5,
       xlab = "Log(Mean + 1)",
       ylab = "Log(Variance + 1)",
       main = "Top 40 Variable Genes (red)")
  points(top40_variable$log_mean, top40_variable$log_var,
         col = "red", pch = 19, cex = 1)
  
  # Plot 4: Z-score vs Mean expression
  plot(gene_data_filtered$log_mean, gene_data_filtered$z_score,
       col = "gray50", pch = 20, cex = 0.5,
       xlab = "Log(Mean + 1)",
       ylab = "Variance Z-score",
       main = "Z-score vs Mean Expression")
  abline(h = 0, col = "blue", lty = 2)
  points(top40_variable$log_mean, top40_variable$z_score,
         col = "red", pch = 19, cex = 1)
  
  dev.off()
  
  cat("\nPart 6: Variable genes visualization saved as 'Part6_variable_genes.png'\n")
}

# =============================================================================
# Summary
# =============================================================================

cat("\n\n")
cat("=============================================================================\n")
cat("EXERCISE COMPLETED\n")
cat("=============================================================================\n")
cat("\nGenerated output files:\n")
cat("- Part3_mean_vs_variance.png: Mean vs variance scatter plot\n")
cat("- Part4_MA_plot.png: DESeq2 MA plot\n")
if (file.exists(circadian_file)) {
  cat("- Part5_per1a_expression.png: per1a time course expression\n")
  cat("- Part5_per1a_FFT_power.png: FFT power spectrum of per1a\n")
  cat("- Part6_variable_genes.png: Variable genes analysis plots\n")
}
cat("\nPlease review the code comments for answers to conceptual questions.\n")


