from __future__ import annotations
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone

try:
    from utils.logger import log
except Exception:
    def log(x): print(x)

# === Пътища ===
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models")); MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR = Path(os.getenv("DATASET_DIR", "data/training")); DATASET_DIR.mkdir(parents=True, exist_ok=True)
CONTEXT_FILE = Path(os.getenv("CONTEXT_FILE", "data/context_dataset.csv"))
CONTEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
TARGET_COL = "target"

@dataclass
class TrainConfig:
    test_frac: float = float(os.getenv("TRAIN_TEST_FRAC", "0.2"))
    random_state: int = int(os.getenv("TRAIN_SEED", "42"))
    max_iter: int = int(os.getenv("TRAIN_MAX_ITER", "1000"))

def _get_features_list(df_cols: List[str]) -> List[str]:
    try:
        from utils.feature_config import get_feature_columns
    except Exception:
        from utils.feature_config import get_feature_columns
    return [c for c in get_feature_columns() if c in df_cols]

class ModelWithContext:
    def __init__(self, name: str = "baseline"):
        self.name = name
        self.path = MODELS_DIR / f"{self.name}.joblib"
        self.pipeline: Optional[Pipeline] = None
        self._feature_cols: List[str] = []
        try:
            if self.path.exists():
                self.pipeline = joblib.load(self.path)
                log(f"[model] loaded: {self.path}")
        except Exception as e:
            log(f"[model] load failed: {e}")

    def _symbol_path(self, symbol: str) -> Path:
        return DATASET_DIR / f"{symbol.replace('/', '_')}.parquet"

    def append_labeled_rows(self, symbol: str, df: pd.DataFrame) -> int:
        if df is None or df.empty or TARGET_COL not in df.columns:
            return 0
        feat_cols = _get_features_list(df.columns.tolist())
        keep = feat_cols + [TARGET_COL]
        out = df[keep].dropna()

        p = self._symbol_path(symbol)
        if p.exists():
            old = pd.read_parquet(p)
            out = pd.concat([old, out], ignore_index=True)
        out = out.drop_duplicates()
        out.to_parquet(p, index=False)

        try:
            if CONTEXT_FILE.exists():
                ctx = pd.read_csv(CONTEXT_FILE, on_bad_lines='skip')
                out = pd.concat([ctx, out], ignore_index=True)
            out.drop_duplicates().to_csv(CONTEXT_FILE, index=False)
        except Exception as e:
            log(f"[context] append failed: {e}")

        return len(out)

    def _load_all_dataset(self) -> pd.DataFrame:
        try:
            if CONTEXT_FILE.exists():
                return pd.read_csv(CONTEXT_FILE, on_bad_lines='skip').dropna()
        except Exception as e:
            log(f"[context] read failed: {e}")

        parts = []
        for p in DATASET_DIR.glob("*.parquet"):
            try:
                parts.append(pd.read_parquet(p))
            except Exception as e:
                log(f"[model] read failed {p}: {e}")
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def retrain_from_context(self, cfg: Optional[TrainConfig] = None) -> dict:
        """
        Train a predictive model using the accumulated context dataset.  This method
        will attempt to select the best performing model out of a set of
        candidate algorithms using time‐series cross‐validation.  The list of
        candidate models can be configured via the environment variable
        ``MODEL_TYPES`` (comma separated list of model identifiers: ``gbm`` for
        GradientBoostingClassifier, ``rf`` for RandomForestClassifier, ``lr``
        for LogisticRegression).  For example, ``MODEL_TYPES=gbm,rf,lr`` will
        evaluate these three models and pick the one with the highest average
        ROC AUC.

        If the dataset is too small to perform cross validation, a single
        train/test split will be used with the first candidate model.

        Parameters
        ----------
        cfg : TrainConfig, optional
            Training hyperparameters controlling split fraction, random state
            and maximum iterations for logistic regression.

        Returns
        -------
        dict
            Information about the training process, including number of
            training/test samples and the chosen model's cross validated AUC.
        """
        cfg = cfg or TrainConfig()
        data = self._load_all_dataset()
        if data.empty:
            raise RuntimeError("No training data found.")

        # Determine which features to use based on current columns
        feat_cols = _get_features_list(data.columns.tolist())
        if not feat_cols:
            log("[model] No valid feature columns found in dataset.")
            return {}
        X = data[feat_cols].values
        y = data[TARGET_COL].values.astype(int)
        n = len(X)
        # Require a minimum number of samples to avoid overfitting
        # Require a minimum number of samples to avoid overfitting.  Previously this
        # threshold was 200, which often prevented any model from being trained on
        # sparse datasets and resulted in constant 0.5 predictions.  Reduce this
        # requirement to 100 samples so that the model can still learn on more
        # limited data and produce non‑trivial probabilities.
        if n < 100:
            log("[model] Not enough data to retrain.")
            return {}

        # Build candidate models list based on env configuration
        requested = os.getenv("MODEL_TYPES", "gbm").split(',')
        # sanitize
        requested = [t.strip().lower() for t in requested if t.strip()]
        candidates = {}
        for t in requested:
            if t == "gbm":
                candidates[t] = Pipeline([
                    ("clf", GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        random_state=cfg.random_state
                    ))
                ])
            elif t == "rf":
                candidates[t] = Pipeline([
                    ("clf", RandomForestClassifier(
                        n_estimators=200,
                        max_depth=None,
                        n_jobs=-1,
                        random_state=cfg.random_state
                    ))
                ])
            elif t == "lr":
                # Standardize inputs for logistic regression.  Do not set n_jobs here
                # because some solvers (e.g. lbfgs) ignore it or raise an error.
                candidates[t] = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(
                        max_iter=cfg.max_iter,
                        random_state=cfg.random_state,
                        solver='lbfgs'
                    ))
                ])

        # If no candidates remain, fallback to a basic GradientBoosting model
        if not candidates:
            candidates["gbm"] = Pipeline([
                ("clf", GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=cfg.random_state
                ))
            ])

        # Decide whether to use time series cross validation
        n_splits = 5
        use_cv = n > n_splits * 2

        best_name: Optional[str] = None
        best_auc: float = -float('inf')
        best_model: Optional[Pipeline] = None

        results = {}

        if use_cv:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            # Evaluate each candidate using time series CV
            for name, pipe in candidates.items():
                fold_aucs: List[float] = []
                try:
                    for train_idx, test_idx in tscv.split(X):
                        Xtr, Xte = X[train_idx], X[test_idx]
                        ytr, yte = y[train_idx], y[test_idx]
                        # Clone the pipeline to ensure fresh model each fold
                        model = clone(pipe)
                        model.fit(Xtr, ytr)
                        proba = model.predict_proba(Xte)[:, 1]
                        # compute AUC if both classes present, else NaN
                        if len(np.unique(yte)) > 1:
                            auc = roc_auc_score(yte, proba)
                            fold_aucs.append(auc)
                    # compute mean AUC across folds
                    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else float('nan')
                    results[name] = mean_auc
                    if mean_auc > best_auc:
                        best_auc = mean_auc
                        best_name = name
                        best_model = pipe
                except Exception as e:
                    log(f"[model] cv evaluation failed for {name}: {e}")
        else:
            # Fallback to a single holdout split (train/test)
            split = int(n * (1 - cfg.test_frac))
            Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
            # Evaluate each candidate
            for name, pipe in candidates.items():
                try:
                    model = clone(pipe)
                    model.fit(Xtr, ytr)
                    proba = model.predict_proba(Xte)[:, 1]
                    if len(np.unique(yte)) > 1:
                        auc = roc_auc_score(yte, proba)
                    else:
                        auc = float('nan')
                    results[name] = float(auc)
                    if auc > best_auc:
                        best_auc = float(auc)
                        best_name = name
                        best_model = pipe
                except Exception as e:
                    log(f"[model] holdout evaluation failed for {name}: {e}")

        # If for some reason no model was selected, choose the first candidate
        if best_model is None:
            best_name, best_model = next(iter(candidates.items()))
            log(f"[model] Defaulting to model {best_name} due to evaluation failure.")

        # Fit the best model on the entire dataset
        try:
            final_model = clone(best_model)
            final_model.fit(X, y)
            self.pipeline = final_model
            self._feature_cols = feat_cols
            # Save the pipeline to disk
            try:
                joblib.dump(self.pipeline, self.path)
                log(f"[model] saved: {self.path}")
            except Exception as e:
                log(f"[model] save failed: {e}")
        except Exception as e:
            log(f"[model] final training failed: {e}")
            return {}

        return {
            "n_samples": int(n),
            "auc_by_model": results,
            "best_model": best_name,
            "best_auc": float(best_auc)
        }

    def predict_proba(self, feat_df: pd.DataFrame) -> float:
        if self.pipeline is None or feat_df is None or feat_df.empty:
            return 0.5

        if not self._feature_cols:
            self._feature_cols = _get_features_list(feat_df.columns.tolist())

        missing = [c for c in self._feature_cols if c not in feat_df.columns]
        for m in missing:
            feat_df[m] = 0.0

        last = feat_df[self._feature_cols].tail(1).values
        try:
            return float(self.pipeline.predict_proba(last)[0, 1])
        except Exception as e:
            log(f"[model] predict error: {e}")
            return 0.5
