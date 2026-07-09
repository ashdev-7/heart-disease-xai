# Clinical XAI Reliability Evaluation

## Paper

**Evaluating the Reliability and Clinical Consistency of Explainability Methods in Deep Learning–Based Heart Disease Prediction**

## Datasets

- Framingham Heart Study 
- CDC BRFSS 2015 

## Models

- Logistic Regression (LR)
- Random Forest (RF)
- Multi-Layer Perceptron (MLP)

## Explainability Methods

- SHAP (LinearExplainer, TreeExplainer, KernelExplainer)
- Integrated Gradients (IG)

## Evaluation Metrics

- **Clinical Alignment:** Kendall's τ (against ACC/AHA 2019 guidelines)
- **Instance-Level Stability:** Mean Rank Perturbation (under 1%, 2%, and 5% input noise)
- **Cross-Seed Consistency:** Spearman rank correlation (ρ)
- **Cross-Model Consistency:** Jaccard similarity

## Reproduce

Run xai-reliability-pipeline
