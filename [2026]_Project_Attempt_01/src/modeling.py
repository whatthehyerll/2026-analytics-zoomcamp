"""
Step 3 - Temporal split, model tuning and predictions.

Models (from the lectures): DecisionTreeClassifier, RandomForestClassifier.
Selection metric: precision on VALIDATION (as in Modules 3-4), because a buy
signal matters only when the model says "buy".
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier

TARGET = "is_outperform_6m"


# ---------------------------------------------------------------------------
# Temporal split with an embargo
# ---------------------------------------------------------------------------
def add_split(monthly: pd.DataFrame, train_end: str, valid_end: str) -> pd.DataFrame:
    ym = monthly["ym"]
    monthly["split"] = np.where(ym <= pd.Period(train_end, "M"), "train",
                       np.where(ym <= pd.Period(valid_end, "M"), "validation", "test"))
    return monthly


def fit_mask(monthly, upto_split: str, embargo_months: int, train_end, valid_end):
    """
    Rows usable for fitting a model that will be evaluated on the NEXT split.

    A 6-month label starting less than 6 months before the next split overlaps
    that split's period, so those rows are dropped (embargo) to avoid leakage.
    """
    boundary = pd.Period(train_end if upto_split == "train" else valid_end, "M")
    allowed = ["train"] if upto_split == "train" else ["train", "validation"]
    return (monthly["split"].isin(allowed)
            & (monthly["ym"] <= boundary - embargo_months)
            & monthly[TARGET].notna())


def make_X(monthly: pd.DataFrame, features) -> pd.DataFrame:
    """Same cleaning as the course: inf -> NaN -> 0."""
    return monthly[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")


# ---------------------------------------------------------------------------
# Hyperparameter tuning on VALIDATION
# ---------------------------------------------------------------------------
def _make_model(kind: str, depth: int, seed: int):
    if kind == "DecisionTree":
        return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=10, random_state=seed)
    return RandomForestClassifier(n_estimators=300, max_depth=depth, min_samples_leaf=10,
                                  n_jobs=-1, random_state=seed)


def tune_models(monthly, features, cfg, threshold: float = 0.5, min_positive: int = 20):
    """
    Grid over model type x max_depth. Fit on TRAIN (with embargo), score on VALIDATION.
    Returns a results table sorted by validation precision.
    """
    fit_rows = fit_mask(monthly, "train", cfg.EMBARGO_MONTHS, cfg.TRAIN_END, cfg.VALID_END)
    val_rows = (monthly["split"] == "validation") & monthly[TARGET].notna()
    X_fit, y_fit = make_X(monthly[fit_rows], features), monthly.loc[fit_rows, TARGET]
    X_val, y_val = make_X(monthly[val_rows], features), monthly.loc[val_rows, TARGET]

    rows = []
    for kind in ["DecisionTree", "RandomForest"]:
        for depth in range(2, 9):
            model = _make_model(kind, depth, cfg.RANDOM_STATE).fit(X_fit, y_fit)
            pred = (model.predict_proba(X_val)[:, 1] >= threshold).astype(int)
            n_pos = int(pred.sum())
            prec = precision_score(y_val, pred, zero_division=0) if n_pos >= min_positive else np.nan
            rows.append({"model": kind, "max_depth": depth, "val_precision": prec, "val_n_positive": n_pos})

    res = pd.DataFrame(rows).sort_values("val_precision", ascending=False, na_position="last")
    print(f"Base rate (validation): {y_val.mean():.3f}  | train rows {len(y_fit)}, validation rows {len(y_val)}")
    return res.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Final model and predictions for every row (incl. the latest, unlabeled month)
# ---------------------------------------------------------------------------
def fit_final_and_predict(monthly, features, cfg, kind: str, depth: int):
    rows = fit_mask(monthly, "validation", cfg.EMBARGO_MONTHS, cfg.TRAIN_END, cfg.VALID_END)
    model = _make_model(kind, depth, cfg.RANDOM_STATE).fit(make_X(monthly[rows], features),
                                                          monthly.loc[rows, TARGET])
    monthly["proba"] = model.predict_proba(make_X(monthly, features))[:, 1]
    return model, monthly


def test_report(monthly, thresholds=(0.5, 0.55, 0.6)) -> pd.DataFrame:
    """
    Precision and number of signals on TEST for several decision thresholds.

    n_months matters: all assets are compared with the SAME benchmark, so in a
    month when the benchmark does badly almost every asset "outperforms".
    Signals concentrated in a few months can show high precision by luck
    (observed in a synthetic-data smoke test: 0.80 precision on pure noise).
    """
    t = monthly[(monthly["split"] == "test") & monthly[TARGET].notna()]
    out = [{"threshold": "base rate", "n_signals": len(t), "n_months": t["ym"].nunique(),
            "precision": t[TARGET].mean()}]
    for thr in thresholds:
        sel = t[t["proba"] >= thr]
        out.append({"threshold": thr, "n_signals": len(sel), "n_months": sel["ym"].nunique(),
                    "precision": sel[TARGET].mean() if len(sel) else np.nan})
    return pd.DataFrame(out)


def feature_importance(model, features, top: int = 15) -> pd.Series:
    return pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(top)
