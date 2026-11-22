"""
hybrid_models.py
Custom hybrid model used in the notebook and loaded by the dashboard.

RFANNHybrid (Random Forest -> ANN cascade)
  1. A Random Forest is trained on the features (log1p target).
  2. Its out-of-bag (OOB) predictions are added as an extra input column.
     OOB predictions are honest: each training row is predicted only by
     trees that never saw it, so the ANN does not learn from leaked values.
  3. A multilayer perceptron (ANN) learns the final prediction from the
     original features + the RF opinion. The ANN can correct smooth,
     non-linear patterns that the step-shaped RF output misses.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dengue_features import CAT_FEATURES, NUM_FEATURES


def make_preprocessor(scale: bool = False) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [("num", num, NUM_FEATURES),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES)],
        verbose_feature_names_out=False,
    )


class RFANNHybrid(RegressorMixin, BaseEstimator):
    """Works in log1p space internally; predict() returns case counts."""

    def __init__(self, rf_params=None, ann_params=None, random_state=42):
        self.rf_params = rf_params
        self.ann_params = ann_params
        self.random_state = random_state

    def fit(self, X, y):
        y_log = np.log1p(np.asarray(y, dtype=float))
        rf_params = dict(n_estimators=150, min_samples_leaf=3, max_features=0.5)
        rf_params.update(self.rf_params or {})
        self.rf_ = Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("model", RandomForestRegressor(oob_score=True, n_jobs=-1,
                                            random_state=self.random_state, **rf_params)),
        ]).fit(X, y_log)
        oob = self.rf_.named_steps["model"].oob_prediction_

        ann_params = dict(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3,
                          max_iter=300, early_stopping=True, n_iter_no_change=15)
        ann_params.update(self.ann_params or {})
        self.prep_ann_ = make_preprocessor(scale=True).fit(X)
        Z = self._stack(self.prep_ann_.transform(X), oob)
        self.rf_scale_ = (oob.mean(), oob.std())
        Z[:, -1] = (oob - self.rf_scale_[0]) / self.rf_scale_[1]
        self.ann_ = MLPRegressor(random_state=self.random_state, **ann_params).fit(Z, y_log)
        return self

    @staticmethod
    def _stack(A, col):
        return np.column_stack([A, col])

    def predict(self, X):
        rf_log = self.rf_.predict(X)
        Z = self._stack(self.prep_ann_.transform(X),
                        (rf_log - self.rf_scale_[0]) / self.rf_scale_[1])
        return np.clip(np.expm1(self.ann_.predict(Z)), 0, None)


class CatBoostNative(RegressorMixin, BaseEstimator):
    """Thin sklearn wrapper so CatBoost (with native text categories) can be
    cloned by sklearn tools such as StackingRegressor and TransformedTargetRegressor."""

    def __init__(self, iterations=800, depth=6, learning_rate=0.05, random_seed=42, thread_count=2):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.thread_count = thread_count

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        self.model_ = CatBoostRegressor(iterations=self.iterations, depth=self.depth,
                                        learning_rate=self.learning_rate, loss_function="RMSE",
                                        cat_features=CAT_FEATURES, random_seed=self.random_seed,
                                        thread_count=self.thread_count, verbose=0)
        self.model_.fit(X, y)
        self.feature_importances_ = self.model_.get_feature_importance()
        return self

    def predict(self, X):
        return self.model_.predict(X)
