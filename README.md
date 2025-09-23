# Income Classification Project

## 📌 Overview
This project predicts whether an individual earns **>50K or ≤50K** using demographic, employment, and financial attributes.  
The dataset contains ~50,000 labeled entries and 25,000 unlabeled entries.  
The goal was to preprocess the data, compare different models, and generate predictions for the unlabeled set.

---

## 📊 Data & Preprocessing
- **Target:** `Income` (binary classification).
- **Weights:** `Weighting` column used for weighted evaluation.
- **Missing values:**
  - `Employment_Type` → imputed with most frequent value (mode).
  - `Employment_Area` & `Country_Of_Birth` → imputed with a DecisionTreeClassifier.
- **Transformations:**
  - Log transformation on skewed numeric variables (`Financial_Assets_Gains`, `Financial_Assets_Loses`, `Weighting`).
  - Standard scaling for numeric features.
  - One-hot encoding for categorical features.

---

## 🔎 Exploratory Data Analysis
- **Imbalances:** Majority categories dominate (e.g., *Private* in Employment_Type, *White* in Ethnicity, *Male* in Gender).
- **Numeric trends:**
  - High earners are typically older and work more hours weekly.
  - Financial assets show heavy skew → log transformation applied.
- **Correlation heatmap:**
  - Moderate positive correlation of `Income` with `Age` (0.23), `Weekly_Working_Time` (0.24), and `Financial_Assets_Gains` (0.22).
  - No strong multicollinearity among numeric features.

---

## 🤖 Models & Tuning
- **Logistic Regression**
  - Baseline.
- **Decision Tree Classifier**
  - Tuned `max_depth`, `min_samples_split`, `min_samples_leaf`.
- **Random Forest Classifier**
  - Tuned `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`.

---

## 📈 Evaluation Metrics
- **PR-AUC** (main metric) → better for imbalanced classification.
- **ROC-AUC** (secondary metric).
- **Weighted evaluation** using `Weighting` column.
- **Threshold tuning:**
  - 0.3 → high recall, lower precision.
  - 0.6 → precision ≈ 0.80, recall ≈ 0.50 → chosen for final model to prioritize **precision**.

---

## 🏆 Results (Comparison Table)

| Model                         | CV ROC-AUC | CV PR-AUC | Holdout ROC-AUC | Holdout PR-AUC |
|-------------------------------|------------|-----------|-----------------|----------------|
| Logistic Regression (Baseline)| 0.893      | 0.723     | 0.914           | 0.777          |
| Decision Tree (Default)       | 0.737      | 0.458     | 0.729           | 0.418          |
| Decision Tree (Tuned)         | 0.874      | 0.721     | 0.876           | 0.723          |
| Random Forest (Tree Prep)     | 0.891      | 0.734     | 0.895           | 0.700          |
| **Random Forest (Tuned)**     | **0.908**  | **0.783** | **0.922**       | **0.804**      |

👉 **Final model:** Random Forest (Tuned).

---

## 📂 Deliverables
- **Cleaned & Imputed Dataset** → ready for regression/correlation analyses.
- **Final Model** → Random Forest (Tuned), retrained on all labeled data.
- **Predictions for 25k Unlabeled Data** →  
  Output file: `income_predictions_unlabeled.csv`  
  Contains:  
  - `Predicted_Label` (`>50K` / `<=50K`)  
  - `Predicted_Prob` (probability of >50K)

---

## 📌 Key Takeaways
1. Preprocessing (imputation, log transforms, scaling) was critical for stable modeling.
2. Random Forest outperformed Logistic Regression and Decision Tree in both PR-AUC and ROC-AUC.
3. Threshold tuning (0.6) allowed prioritizing precision, as required.
4. Predictions for 25k unlabeled entries were generated for downstream analyses.

---
