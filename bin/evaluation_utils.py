from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from sklearn.model_selection import cross_validate
import numpy as np

# Global results dictionary
results_summary = {}

# -------------------------------------------------------
# Cross-validation helper
# -------------------------------------------------------
def evaluate_with_cv(model, X_train, y_train, w_train, model_name="Model", cv=None):
    """Run cross-validation with ROC-AUC and PR-AUC, print results,
    and store mean values in results_summary."""
    
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision"
    }
    
    cv_results = cross_validate(
        model,
        X_train, y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        params={"model__sample_weight": w_train.to_numpy()},
        return_train_score=False
    )
    
    # Mean ± std
    cv_roc_auc_mean = cv_results["test_roc_auc"].mean()
    cv_roc_auc_std  = cv_results["test_roc_auc"].std()
    cv_pr_auc_mean  = cv_results["test_pr_auc"].mean()
    cv_pr_auc_std   = cv_results["test_pr_auc"].std()
    
    # Print nicely
    print(f"\n=== {model_name} (Cross-Validation) ===")
    print(f"CV ROC-AUC: {cv_roc_auc_mean:.3f} ± {cv_roc_auc_std:.3f}")
    print(f"CV PR-AUC : {cv_pr_auc_mean:.3f} ± {cv_pr_auc_std:.3f}")
    
    # Update or extend results_summary
    if model_name not in results_summary:
        results_summary[model_name] = {}
        
    # Store only means in results_summary
    results_summary[model_name].update({
        "CV ROC-AUC": cv_roc_auc_mean,
        "CV PR-AUC": cv_pr_auc_mean
    })
    
    return cv_results, results_summary

# -------------------------------------------------------
# Holdout evaluation helper
# -------------------------------------------------------
def evaluate_on_holdout(model, X_test, y_test, w_test, model_name="Model"):
    """Evaluate a fitted model on holdout set with sample weights,
    print metrics, and add to results_summary."""
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob, sample_weight=w_test)
    pr_auc  = average_precision_score(y_test, y_prob, sample_weight=w_test)

    print(f"\n=== {model_name} (Holdout) ===")
    print(classification_report(y_test, y_pred, sample_weight=w_test))
    print("ROC-AUC:", round(roc_auc, 3))
    print("PR-AUC :", round(pr_auc, 3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, sample_weight=w_test))

    # Update or extend results_summary
    if model_name not in results_summary:
        results_summary[model_name] = {}
    
    results_summary[model_name].update({
        "Holdout ROC-AUC": roc_auc,
        "Holdout PR-AUC": pr_auc
    })
    
    return results_summary, y_prob
