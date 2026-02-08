# Retail Fraud Detection (Data Mining Cup 2019)

This repository contains an advanced, cost-sensitive machine learning approach to detect retail checkout fraud, developed for the Data Mining Cup (DMC) 2019 dataset.

## The Business Problem
In retail self-checkout systems, detecting fraud is highly imbalanced and cost-sensitive. 
Based on the DMC 2019 evaluation matrix:
- **True Positive (Caught Fraud):** +5 €
- **False Negative (Missed Fraud):** -5 €
- **False Positive (False Accusation):** -25 € *(Massive penalty for accusing an innocent customer)*

Standard models optimizing for pure "Accuracy" or "ROC-AUC" will bankrupt the business by generating too many False Positives.

## The Solution: "Hydra Ensemble & Dual-Gate Strategy"
To maximize actual business profit, this project implements a specialized architecture:

1. **Systematic Feature Engineering:** Created behavioral interaction features (e.g., scan speed per item, void ratios, trust-level interactions).
2. **The Hydra Ensemble:** A Soft-Voting Classifier combining:
   - **XGBoost:** For capturing complex non-linear patterns.
   - **Random Forest:** For reducing variance and overfitting.
   - **Logistic Regression:** For baseline probability calibration.
3. **Dual-Gate Thresholding:** Instead of a global probability threshold, we optimized distinct decision boundaries for different customer risk segments using Brute-Force Grid Search (e.g., Threshold `0.85` for Risk Level 1, `0.75` for Risk Level 2).

## Results
By minimizing the severe €25 False Positive penalty while maximizing detection in high-risk zones, the model successfully turned a baseline negative profit into a massive gain.

* **Final Profit:** **€ 20,070**

## Tech Stack
* Python
* Pandas, NumPy
* Scikit-Learn (Random Forest, Logistic Regression, Voting Classifier)
* XGBoost
* Imbalanced-learn
* Matplotlib, Seaborn
