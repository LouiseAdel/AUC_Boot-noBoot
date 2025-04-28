import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import scipy.stats as stats

# Function to compute AUC, standard error, and p-values using bootstrapping
def compute_auc_with_se_and_pvalues(y_true, y_scores, baseline_values=[0.5, 0.9636, 1.0], n_iterations=1000):
    # Ensure inputs are of the same length
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length")
    
    aucs = []  # Single list to store all bootstrap AUC values
    
    # Perform bootstrapping
    for i in range(n_iterations):
        # Resample with replacement
        X_resampled, y_resampled = resample(y_scores, y_true, replace=True, n_samples=len(y_true), random_state=i)
        # Only calculate AUC if both classes are present
        if len(np.unique(y_resampled)) > 1:
            auc = roc_auc_score(y_resampled, X_resampled)
            aucs.append(auc)
    
    # Calculate mean AUC and standard error
    auc_mean = np.mean(aucs)
    auc_se = np.std(aucs) / np.sqrt(len(aucs))  # Standard error calculation
    
    # Calculate p-values using t-test for each baseline value
    p_values = {}
    for baseline in baseline_values:
        t_stat, p_value = stats.ttest_1samp(aucs, baseline)
        p_values[baseline] = p_value
    
    return auc_mean, auc_se, p_values, np.array(aucs)  # Return the AUC array for additional calculations

# Function to get a readable representation of extremely small p-values
def format_extreme_pvalue(aucs, baseline):
    n = len(aucs)
    mean = np.mean(aucs)
    std = np.std(aucs, ddof=1)
    t_stat = (mean - baseline) / (std / np.sqrt(n))
    
    # Handle extremely large t-statistic values
    if abs(t_stat) > 1e15:
        # For extremely large t-statistics, compute a rough approximation
        df = n - 1
        # For large abs(t), t-distribution approaches exp(-t²/2)·t^(-df-1)
        # We'll just return a very large negative exponent
        return f"~10^-{min(1000, int(abs(t_stat)))}"
    
    try:
        # Try to use logarithm of the survival function
        log_p = stats.t.logsf(abs(t_stat), df=n-1) + np.log(2)  # Two-tailed test
        
        # If log_p is -inf, handle it specially
        if np.isinf(log_p):
            return "< 10^-324"  # Smallest representable double precision number
        
        exp = int(log_p / np.log(10))
        mantissa = 10 ** (log_p - exp * np.log(10))
        return f"{mantissa:.2f}×10^{exp}"
    except:
        # If any calculation fails, return a fallback message
        return "extremely small (< 10^-300)"

# Example datasets
y_true_1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
y_scores_1 = [26, 27, 19, 14, 23, 13, 2, 3, 4, 14, 0, 3, 2, 1, 8, 0, 1, 5, 6]

y_true_2 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
y_scores_2 = [24, 26, 26, 15, 32, 11, 4, 8, 13, 17, 4, 6, 3, 0, 13, 12, 5, 24, 7]

y_true_3 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
y_scores_3 = [24, 22, 23, 10, 43, 24, 4, 8, 12, 11, 6, 7, 3, 0, 16, 6, 3, 28, 6]

# Baseline AUC values to compare to
baseline_values = [0.5, 0.9636, 1.0]

# Perform bootstrapping for each set
auc_1, se_1, pvals_1, aucs_1 = compute_auc_with_se_and_pvalues(y_true_1, y_scores_1, baseline_values)
auc_2, se_2, pvals_2, aucs_2 = compute_auc_with_se_and_pvalues(y_true_2, y_scores_2, baseline_values)
auc_3, se_3, pvals_3, aucs_3 = compute_auc_with_se_and_pvalues(y_true_3, y_scores_3, baseline_values)

# Print results with scientific notation for very small p-values
print(f"Set FFPE - AUC: {auc_1:.4f} ± {se_1:.4f}")
print(f"P-values for Set FFPE (vs baseline values):")
for baseline in baseline_values:
    p_value = pvals_1[baseline]
    if p_value < 1e-10:  # Use precise calculation for very small p-values
        formatted_p = format_extreme_pvalue(aucs_1, baseline)
        print(f"  vs {baseline}: {formatted_p}")
    else:
        print(f"  vs {baseline}: {p_value:.6f}")

print(f"\nSet Frozen 5X - AUC: {auc_2:.4f} ± {se_2:.4f}")
print(f"P-values for Set Frozen 5X (vs baseline values):")
for baseline in baseline_values:
    p_value = pvals_2[baseline]
    if p_value < 1e-10:  # Use precise calculation for very small p-values
        formatted_p = format_extreme_pvalue(aucs_2, baseline)
        print(f"  vs {baseline}: {formatted_p}")
    else:
        print(f"  vs {baseline}: {p_value:.6f}")

print(f"\nSet Frozen 50X - AUC: {auc_3:.4f} ± {se_3:.4f}")
print(f"P-values for Set Frozen 50X (vs baseline values):")
for baseline in baseline_values:
    p_value = pvals_3[baseline]
    if p_value < 1e-10:  # Use precise calculation for very small p-values
        formatted_p = format_extreme_pvalue(aucs_3, baseline)
        print(f"  vs {baseline}: {formatted_p}")
    else:
        print(f"  vs {baseline}: {p_value:.6f}")