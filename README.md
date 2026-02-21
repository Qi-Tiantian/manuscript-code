# Code for: Interpretable Predictive Model for Listed Companies ESG Greenwashing Based on XGBoost and SHAP

This repository contains the complete source code to reproduce all results from the manuscript titled "Interpretable Predictive Model for Listed Companies ESG Greenwashing Based on XGBoost and SHAP" (Submitted to *Scientific Reports*). The code implements a machine learning model for binary classification, encompassing data preprocessing, handling class imbalance, hyperparameter tuning using the XGBoost algorithm, model performance evaluation, and interpretability analysis using the SHAP method.

## Repository DOI (Permanent Archive)
[![DOI](https://zenodo.org/badge/DOI/)](https://doi.org/)

*Note: Replace the above example DOI with your actual one from Zenodo after archiving.*

## Files in This Repository
- `Model Prediction Results.ipynb`: The primary Jupyter Notebook containing the results of the corporate characteristics model and the external pressure characteristics model.
- `Model Prediction Efficacy and Further Analysis.ipynb`: The Jupyter Notebook containing the results of model prediction efficacy and further analysis.
- `requirements.txt`: List of all Python packages and their specific versions.
- `LICENSE`: MIT License file.
- `README.md`: This file.

## Quick Start: How to Reproduce the Results

Follow these steps to replicate the computational environment and run the analysis:

1.  **Obtain the Code**: Clone this repository or download it as a ZIP file.
2.  **Navigate to the Directory**: Open a terminal/command prompt and change to the project directory.
3.  **(Recommended) Create a Virtual Environment**:
bash
python -m venv venv
# Activate it:
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate
4.  **Install Dependencies**:
bash
pip install -r requirements.txt
This installs key packages including `pandas`, `numpy`, `xgboost`, `matplotlib`, `seaborn`, `shap`, and so on.
5.  **Prepare the Data**:
- Since the raw data consists of multiple Excel files, we use the file named `Data1.xlsx` as an example.
- **Due to data licensing, this data file is NOT included in the repository. All data supporting the findings of this study are available within the paper and its Supplementary Information.**
- Users must obtain the data independently and place the `Data1.xlsx` file in the same folder as `Model Prediction Results.ipynb`.
6.  **Execute the Analysis**:
bash
jupyter notebook
- In the browser, open `Model Prediction Results.ipynb`.
- **Critical First Step**: Before running, modify the `excel_file` variable in the first code cell to point to your local `Data1.xlsx` path (e.g., `excel_file = "./Data1.xlsx"`).
- Run all cells (`Cell -> Run All`). This will execute the entire pipeline and generate all outputs.

## Detailed Workflow & Outputs

The `Model Prediction Results.ipynb` and `Model Prediction Efficacy and Further Analysis.ipynb` notebooks execute the following sequence:

1.  **Data Loading**: Reads data files and extracts pre-defined features and the target label.
2.  **Class Imbalance Handling**: Applies the SMOTEENN algorithm.
3.  **Feature Preprocessing**: Performs winsorization (1st/99th percentiles) and Z-score standardization.
4.  **Model Training & Tuning**: Uses a 5-fold cross-validated Grid Search to optimize XGBoost hyperparameters.
5.  **Model Evaluation**: Calculates Accuracy, Precision, Recall, F1-Score, AUC, and plots a confusion matrix and ROC curve.
6.  **Model Interpretation**: Calculates feature importance and SHAP values, generating summary plots, dependence plots, and individual force plots.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use or build upon this code, please cite the associated manuscript and this code archive:

> Zhang Jianfeng, Qi Tiantian. (2026). Interpretable Predictive Model for Listed Companies ESG Greenwashing Based on XGBoost and SHAP. *Scientific Reports* (Under Review). Code available at: https://doi.org/[Your-Actual-DOI]
