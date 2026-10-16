> **Note added October 2026.** This file was written by the original notebook
> and is kept unchanged below. The quantitative follow-up in the
> [README](../README.md#key-findings) qualifies insights 2-4. The runtime
> saving comes from using fewer background rows: a random 20-row background is
> just as fast as 20 centroids. At equal size, the equal-weight K-means
> background was usually *further* from exact SHAP values than a random
> subsample (46 of 54 settings). Weighting centroids by cluster size fixed this
> on Breast Cancer and MAGIC, and for logistic regression on all three
> datasets, but not for the tree models on Spambase.

# Interpretability Insights from SHAP vs C-SHAP Analysis

This project compares standard SHAP with a C-SHAP-style approximation using K-means clustering to reduce the size of the background dataset.

## Key Interpretability Insights

### 1. Standard SHAP provides detailed explanations but can be slow
Standard SHAP produces informative feature attributions, but explanation generation may become computationally expensive when the background dataset is large.

### 2. K-means clustering can reduce explanation runtime
Using K-means centroids as background representatives reduces the number of reference points used by SHAP, making explanation generation faster.

### 3. C-SHAP can preserve much of the global explanation structure
Despite using a compressed background, the C-SHAP-style method can still produce feature importance rankings that remain highly correlated with standard SHAP.

### 4. Clustering offers a practical trade-off between speed and fidelity
The results suggest that background compression through clustering can improve efficiency without severely sacrificing interpretability quality.

### 5. This approach is useful for scalable interpretability
For larger tabular datasets, C-SHAP-style approximation may provide a more practical interpretability solution when explanation time is important.

## Research Value

This experiment demonstrates a simple but meaningful approximation strategy for scalable explainability. It shows how clustering can be integrated into SHAP-based workflows to balance runtime efficiency and explanation fidelity.
