from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def fairness_metrics(y_true, y_pred, sensitive):
    return {
        "dp_diff": demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive),
        "eo_diff": equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive),
    }
