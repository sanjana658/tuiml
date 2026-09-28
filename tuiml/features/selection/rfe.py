
"""Recursive feature elimination selectors."""

from typing import Any

import numpy as np

from tuiml.base.features import FeatureSelector, feature_selector
from tuiml.evaluation.splitting import cross_val_score

from ._base import SelectorMixin, _ensure_numpy


def _get_feature_importances(estimator: Any) -> np.ndarray:
    """Extract one importance value for each feature.

    Parameters
    ----------
    estimator : object
        Fitted estimator exposing ``feature_importances_`` or ``coef_``.

    Returns
    -------
    importances : ndarray of shape (n_features,)
        Non-negative feature importance values.

    Raises
    ------
    ValueError
        If the estimator does not expose a supported importance attribute.
    """
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        importances = np.abs(np.asarray(estimator.coef_))
        if importances.ndim > 1:
            importances = np.max(importances, axis=0)
    else:
        raise ValueError(
            "Estimator must expose either 'feature_importances_' or "
            "'coef_' after fitting."
        )

    if importances.ndim != 1:
        raise ValueError(
            "Feature importance must be one-dimensional after extraction."
        )

    return np.abs(importances)

@feature_selector(
    tags=["wrapper", "recursive", "feature-selection"],
    version="1.0.0",
)
class RFE(FeatureSelector, SelectorMixin):
    """Recursive feature elimination.

    Parameters
    ----------
    estimator : object
        A supervised estimator exposing ``fit`` and either
        ``feature_importances_`` or ``coef_`` after fitting.
    n_features_to_select : int, optional
        Number of features to retain. If ``None``, half of the
        input features are retained.
    step : int or float, default=1
        Number of features to remove at each iteration, or the
        fraction of features to remove when a float is supplied.

    Attributes
    ----------
    n_features_to_select_ : int
        Number of features selected after fitting.
    support_ : ndarray of bool
        Boolean mask indicating selected features.
    ranking_ : ndarray of int
        Feature ranking. Selected features have rank 1.
    """

    def __init__(
        self,
        estimator: Any = None,
        n_features_to_select: int | None = None,
        step: float = 1,
    ):
        super().__init__()
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.n_features_to_select_: int | None = None
        self.support_: np.ndarray | None = None
        self.ranking_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "RFE":
        """Fit RFE and recursively eliminate features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RFE
            The fitted selector.
        """
        if y is None:
            raise ValueError("RFE requires target values (y)")
        if self.estimator is None:
            raise ValueError("estimator must be provided")

        X = _ensure_numpy(X)
        y = _ensure_numpy(y)

        n_features = X.shape[1]
        self._n_features_in = n_features

        if self.n_features_to_select is None:
            target = max(1, n_features // 2)
        else:
            target = int(self.n_features_to_select)

        if target < 1 or target > n_features:
            raise ValueError(
                "n_features_to_select must be between 1 and "
                "the number of features."
            )

        if isinstance(self.step, bool):
            raise TypeError("step must be an int or float.")
        if isinstance(self.step, float):
            if not 0 < self.step < 1:
                raise ValueError("step as a float must be between 0 and 1.")
        elif isinstance(self.step, int):
            if self.step < 1:
                raise ValueError("step must be at least 1.")
        else:
            raise TypeError("step must be an int or float.")

        support = np.ones(n_features, dtype=bool)
        ranking = np.ones(n_features, dtype=int)
        current_rank = 1

        while support.sum() > target:
            active = np.flatnonzero(support)
            X_current = X[:, active]

            estimator = self._clone_estimator()
            estimator.fit(X_current, y)

            importances = _get_feature_importances(estimator)

            if importances.shape[0] != active.size:
                raise ValueError(
                    "Estimator feature importance does not match "
                    "the number of input features."
                )

            if isinstance(self.step, float):
                n_remove = max(1, int(active.size * self.step))
            else:
                n_remove = self.step

            n_remove = min(n_remove, active.size - target)

            order = np.argsort(importances, kind="stable")
            remove_local = order[:n_remove]
            remove_global = active[remove_local]

            ranking[remove_global] = current_rank + 1
            support[remove_global] = False
            current_rank += 1

        self._selected_indices = np.flatnonzero(support)
        self.n_features_to_select_ = int(support.sum())
        self.support_ = support
        self.ranking_ = ranking
        self._feature_scores = support.astype(float)
        self._is_fitted = True

        return self

    def _clone_estimator(self) -> Any:
        """Create a fresh estimator with the same parameters."""
        if hasattr(self.estimator, "get_params"):
            params = self.estimator.get_params()
            return self.estimator.__class__(**params)
        return self.estimator.__class__()

    def _compute_scores(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Return binary scores for selected features."""
        return self._feature_scores

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return the parameter schema for RFE."""
        return {
            "estimator": {
                "type": "object",
                "default": None,
                "description": "Estimator used for recursive elimination.",
            },
            "n_features_to_select": {
                "type": "integer",
                "default": None,
                "description": "Number of features to retain.",
            },
            "step": {
                "type": ["integer", "number"],
                "default": 1,
                "description": (
                    "Number or fraction of features removed per iteration."
                ),
            },
        }

class _RFECVEstimator:
    """Estimator that performs RFE during each cross-validation fit.

    Parameters
    ----------
    estimator : object
        Base estimator used for recursive feature elimination.
    n_features_to_select : int
        Number of features to retain.
    step : int or float
        Number or fraction of features removed at each iteration.
    """

    def __init__(
        self,
        estimator: Any,
        n_features_to_select: int,
        step: float,
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.selector: RFE | None = None
        self.fitted_estimator: Any = None

    def get_params(self) -> dict:
        """Return parameters used to clone the estimator."""
        return {
            "estimator": self.estimator,
            "n_features_to_select": self.n_features_to_select,
            "step": self.step,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RFECVEstimator":
        """Fit RFE and the wrapped estimator on a training fold."""
        self.selector = RFE(
            estimator=self.estimator,
            n_features_to_select=self.n_features_to_select,
            step=self.step,
        )
        self.selector.fit(X, y)

        self.fitted_estimator = self.selector._clone_estimator()
        self.fitted_estimator.fit(X[:, self.selector.support_], y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using features selected during fitting."""
        if self.selector is None or self.fitted_estimator is None:
            raise ValueError("Estimator must be fitted before prediction.")

        return self.fitted_estimator.predict(X[:, self.selector.support_])


@feature_selector(
    tags=["wrapper", "recursive", "cross-validation"],
    version="1.0.0",
)
class RFECV(RFE):
    """Recursive feature elimination with cross-validation.

    Parameters
    ----------
    estimator : object
        A supervised estimator exposing ``fit`` and ``predict``.
    step : int or float, default=1
        Number of features to remove at each iteration, or the
        fraction of features to remove when a float is supplied.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str or callable, optional
        Scoring strategy passed to :func:`cross_val_score`.
    random_state : int, optional
        Random seed used when ``cv`` is an integer.

    Attributes
    ----------
    n_features_to_select_ : int
        Number of features selected by cross-validation.
    support_ : ndarray of bool
        Boolean mask indicating selected features.
    ranking_ : ndarray of int
        Feature ranking.
    cv_results_ : dict
        Cross-validation results for each tested feature count.
    """

    def __init__(
        self,
        estimator: Any = None,
        step: float = 1,
        cv: int = 5,
        scoring: Any | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            estimator=estimator,
            n_features_to_select=None,
            step=step,
        )
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.cv_results_: dict | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "RFECV":
        """Fit RFECV and select the feature count with the best CV score.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RFECV
            The fitted selector.
        """
        if y is None:
            raise ValueError("RFECV requires target values (y)")
        if self.estimator is None:
            raise ValueError("estimator must be provided")

        X = _ensure_numpy(X)
        y = _ensure_numpy(y)

        n_features = X.shape[1]
        self._n_features_in = n_features

        if isinstance(self.cv, bool) or not isinstance(self.cv, int):
            raise TypeError("cv must be an integer.")
        if self.cv < 2:
            raise ValueError("cv must be at least 2.")

        if isinstance(self.step, bool):
            raise TypeError("step must be an int or float.")
        if isinstance(self.step, float) and not 0 < self.step < 1:
            raise ValueError("step as a float must be between 0 and 1.")
        if isinstance(self.step, int) and self.step < 1:
            raise ValueError("step must be at least 1.")

        feature_counts = [n_features]
        current = n_features

        while current > 1:
            if isinstance(self.step, float):
                n_remove = max(1, int(current * self.step))
            else:
                n_remove = self.step

            current = max(1, current - n_remove)

            if current not in feature_counts:
                feature_counts.append(current)

        scores = []
        for count in feature_counts:
            cv_estimator = _RFECVEstimator(
                estimator=self.estimator,
                n_features_to_select=count,
                step=self.step,
            )
            cv_scores = cross_val_score(
                cv_estimator,
                X,
                y,
                cv=self.cv,
                scoring=self.scoring,
                random_state=self.random_state,
            )
            scores.append(float(np.mean(cv_scores)))
        scores_array = np.asarray(scores)
        best_index = int(np.argmax(scores_array))
        best_count = feature_counts[best_index]

        final_selector = RFE(
            estimator=self.estimator,
            n_features_to_select=best_count,
            step=self.step,
        )
        final_selector.fit(X, y)

        self._selected_indices = final_selector._selected_indices
        self.n_features_to_select_ = final_selector.n_features_to_select_
        self.support_ = final_selector.support_
        self.ranking_ = final_selector.ranking_
        self._feature_scores = final_selector._feature_scores
        self._is_fitted = True

        self.cv_results_ = {
            "n_features": np.asarray(feature_counts),
            "mean_test_score": scores_array,
        }

        return self

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return the parameter schema for RFECV."""
        return {
            "estimator": {
                "type": "object",
                "default": None,
                "description": "Estimator used for recursive elimination.",
            },
            "step": {
                "type": ["integer", "number"],
                "default": 1,
                "description": (
                    "Number or fraction of features removed per iteration."
                ),
            },
            "cv": {
                "type": "integer",
                "default": 5,
                "description": "Number of cross-validation folds.",
            },
            "scoring": {
                "type": ["string", "object"],
                "default": None,
                "description": "Scoring strategy for cross-validation.",
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for cross-validation.",
            },
        }
