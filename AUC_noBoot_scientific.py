import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats

# Function to compute AUC and p-values (no bootstrapping)
def compute_auc_and_pvalues(y_true, y_scores, baseline_values=[0.5, 0.9636, 1.0]): #Random classifier, Diossy et al. value, and perfect classifier
    # Compute the AUC
    auc = roc_auc_score(y_true, y_scores)
    
    # Compute the standard error of the AUC (estimated via the formula for standard error of AUC)
    n = len(y_true)
    # Calculate standard error for AUC
    auc_se = np.sqrt((auc * (1 - auc)) / n)
    
    # Perform t-tests for each baseline
    p_values = {}
    for baseline in baseline_values:
        t_stat = (auc - baseline) / auc_se  # Calculate t-statistic
        df = n - 1  # Degrees of freedom
        p_value = stats.t.sf(np.abs(t_stat), df) * 2  # Two-tailed p-value
        p_values[baseline] = p_value
    
    return auc, auc_se, p_values

# Function to format p-value in scientific notation
def format_scientific_pvalue(value):
    if value < 1e-10:
        return "{:.1e}".format(value)  # format as scientific notation for very small values
    else:
        return f"{value:.6f}"  # For larger p-values, keep the regular float format

# Example datasets
y_true_1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #FFPE HRDetect
y_scores_1 = [26, 27, 19, 14, 23, 13, 2, 3, 4, 14, 0, 3, 2, 1, 8, 0, 1, 5, 6] #FFPE ShallowHRD Score

y_true_2 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #Fresh Frozen 5X HRDetect
y_scores_2 = [24, 26, 26, 15, 32, 11, 4, 8, 13, 17, 4, 6, 3, 0, 13, 12, 5, 24, 7] #Fresh Frozen 5X ShallowHRD Score

y_true_3 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #Fresh Frozen 50X HRDetect
y_scores_3 = [24, 22, 23, 10, 43, 24, 4, 8, 12, 11, 6, 7, 3, 0, 16, 6, 3, 28, 6] #Fresh Frozen 50X ShallowHRD Score

# Baseline AUC values to compare to
baseline_values = [0.5, 0.9636, 1.0] #Random classifier, Diossy et al. value, and perfect classifier

# Perform AUC and p-value calculations for each set
auc_1, se_1, pvals_1 = compute_auc_and_pvalues(y_true_1, y_scores_1, baseline_values) #FFPE
auc_2, se_2, pvals_2 = compute_auc_and_pvalues(y_true_2, y_scores_2, baseline_values) #Fresh Frozen 5X
auc_3, se_3, pvals_3 = compute_auc_and_pvalues(y_true_3, y_scores_3, baseline_values) #Fresh Frozen 50X

# Print results
print(f"Set FFPE - AUC: {auc_1:.4f} ± {se_1:.4f}")
print(f"P-values for Set FFPE (vs baseline values):")
for baseline, p_value in pvals_1.items():
    print(f"  vs {baseline}: {format_scientific_pvalue(p_value)}")

print(f"\nSet Frozen 5X - AUC: {auc_2:.4f} ± {se_2:.4f}")
print(f"P-values for Set Frozen 5X (vs baseline values):")
for baseline, p_value in pvals_2.items():
    print(f"  vs {baseline}: {format_scientific_pvalue(p_value)}")

print(f"\nSet Frozen 50X - AUC: {auc_3:.4f} ± {se_3:.4f}")
print(f"P-values for Set Frozen 50X (vs baseline values):")
for baseline, p_value in pvals_3.items():
    print(f"  vs {baseline}: {format_scientific_pvalue(p_value)}")
