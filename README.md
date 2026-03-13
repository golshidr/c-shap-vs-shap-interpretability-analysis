# C-SHAP vs SHAP Interpretability Analysis

This repository presents an interpretability experiment comparing **standard SHAP explanations** with a **C-SHAP–inspired approximation** that uses **K-means clustering** to compress the SHAP background dataset.

The goal of this project is to investigate whether clustering-based background reduction can **significantly reduce explanation runtime while maintaining explanation fidelity**.

---

# Overview

SHAP (SHapley Additive Explanations) is one of the most widely used methods for interpreting machine learning models. However, generating SHAP explanations can be computationally expensive when the background dataset is large.

This project explores a simplified **C-SHAP-style approach** where:

- The training dataset is compressed using **K-means clustering**
- The cluster **centroids** are used as SHAP background representatives
- Model explanations are generated using the reduced background set
- The results are compared with **standard SHAP explanations**

The experiment evaluates both **efficiency and interpretability quality**.

---

# Research Motivation

This experiment is inspired by the **C-SHAP method**, which introduces clustering-based background compression to accelerate SHAP explanations.

📄 Paper:

Ranjbaran, G., Roy, C. K., & Schneider, K. A. (2025).  
**C-SHAP: A Clustering-Boosted SHAP Method for Faster Model Interpretability.**  
Applied Sciences, 15(2), 672.  
https://www.mdpi.com/2076-3417/15/2/672

---

# Dataset

The experiment uses the **Breast Cancer Wisconsin dataset** from `scikit-learn`.

Dataset characteristics:

- Binary classification problem
- 569 samples
- 30 numerical features

---

# Methodology

The workflow implemented in this project follows these steps:

1. Load and preprocess the dataset  
2. Train a **Random Forest classifier**  
3. Generate explanations using **standard SHAP**  
4. Generate explanations using a **C-SHAP-style clustered background**  
5. Compare runtime and explanation similarity  

---

# Results

The experiment demonstrates that:

- Standard SHAP provides detailed explanations but can be slow
- Clustering the background dataset reduces explanation runtime
- C-SHAP-style explanations preserve much of the global feature importance structure
- Clustering provides a practical trade-off between **speed and interpretability fidelity**

---
# Methodology and Tools

This project was implemented using:

- Python
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Matplotlib

The workflow includes:
1. Data preprocessing
2. Random Forest model training
3. Standard SHAP explanations
4. C-SHAP style background clustering using K-means
5. Runtime and explanation comparison
