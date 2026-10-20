# Model Card: Churn-ML

**Task:** binary classification, probability that a subscriber churns.
**Model:** best of {Logistic Regression, Random Forest, HistGradientBoosting} by CV ROC-AUC, wrapped in probability calibration. Chosen model and metrics are written to `artifacts/metadata.json` on every run.

**Data:** synthetic telecom-style customers (10 raw features: tenure, charges, products, support calls, late payments, senior flag, usage, contract, internet service, payment method). ~22% positive rate, ~3% missing values in two columns.

**Evaluation:** stratified 80/20 split; 5-fold CV for selection; threshold tuned on out-of-fold training predictions; metrics reported on the untouched test set (see README).

**Intended use:** demonstration of ML engineering practice; prioritizing retention outreach.
**Out-of-scope:** any real decision-making about people without re-training and validating on real data.

**Limitations & risks**
- Trained on synthetic data; performance on real populations is unknown.
- `senior_citizen` is a sensitive attribute: audit fairness (e.g. equalized recall across groups) before any real deployment, and consider removing it.
- Predicts churn *propensity*, not whether an intervention will work (see uplift modelling in the roadmap).
- No drift monitoring is included; retrain periodically.
