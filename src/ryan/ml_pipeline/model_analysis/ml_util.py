"""
This module hosts a utility to run validation for ML modles
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, Union

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from . import features


def run_kfold_cv(
    X_numerical: List[str],
    X_category: List[str],
    df_raw: pd.DataFrame,
    model: RegressorMixin,
    target: str,
    n_splits: int = 10,
    random_state: int = 27,
) -> Tuple[float, float, float, float]:
    """
    Run K-fold cross-validation for a single target and return summary metrics.

    Args:
        X_numerical:
            List of numerical feature column names.
        X_category:
            List of categorical feature column names.
        df_raw:
            DataFrame containing both feature and target columns.
        model:
            Any regressor that follows the scikit-learn interface
            (has fit and predict). This can be a bare model or a
            Pipeline with preprocessing.
        target:
            Name of the target column to predict.
        n_splits:
            Number of folds for KFold cross-validation.
        random_state:
            Random seed for the KFold splitter.

    Returns:
        Tuple containing:
            r2_mean:
                Mean R^2 across folds.
            r2_std:
                Standard deviation of R^2 across folds.
            rmse_mean:
                Mean RMSE across folds.
            rmse_std:
                Standard deviation of RMSE across folds.
    """
    feature_columns = X_numerical + X_category
    X = df_raw[feature_columns].copy()
    y = df_raw[target]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    base_estimator = clone(model)

    r2_scores: List[float] = []
    rmse_scores: List[float] = []

    for train_index, test_index in kf.split(X):
        X_train_fold = X.iloc[train_index]
        X_test_fold = X.iloc[test_index]
        y_train_fold = y.iloc[train_index]
        y_test_fold = y.iloc[test_index]

        fold_model = clone(base_estimator)
        fold_model.fit(X_train_fold, y_train_fold)

        y_pred_fold = fold_model.predict(X_test_fold)

        r2_scores.append(r2_score(y_test_fold, y_pred_fold))
        rmse_scores.append(root_mean_squared_error(y_test_fold, y_pred_fold))

    r2_mean = float(np.mean(r2_scores))
    r2_std = float(np.std(r2_scores))
    rmse_mean = float(np.mean(rmse_scores))
    rmse_std = float(np.std(rmse_scores))

    return r2_mean, r2_std, rmse_mean, rmse_std


def make_regression_pipeline(
    X_numerical: List[str],
    X_category: List[str],
    model_cls: Type[RegressorMixin],
    default_params: Mapping | None = None,
    **override_params,
) -> Pipeline:
    """
    Create a preprocessing + regression Pipeline for an arbitrary model class.

    Args:
        X_numerical:
            List of numerical feature column names.
        X_category:
            List of categorical feature column names.
        model_cls:
            Regressor class (e.g. XGBRegressor, RandomForestRegressor, MLPRegressor).
        default_params:
            Mapping of default hyperparameters for this model.
        override_params:
            Extra keyword arguments that override entries in default_params.

    Returns:
        A scikit-learn Pipeline with preprocessing and the instantiated model.
    """
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), X_numerical),
            ("cat", OneHotEncoder(handle_unknown="ignore"), X_category),
        ]
    )

    # Merge default parameters with overrides
    params = dict(default_params or {})
    params.update(override_params)

    # Instantiate model
    model = model_cls(**params)

    # Build pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def make_suggest_params_fn(search_space: Dict[str, tuple]):
    """
    Turn a simple search-space dict into a Optuna suggest_params_fn(trial).

    Expected patterns:
        "param": ("int", low, high)
        "param": ("int_log", low, high)
        "param": ("float", low, high)
        "param": ("float_log", low, high)
        "param": ("categorical", [choices])
    """

    def _suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
        params: Dict[str, Any] = {}

        for name, spec in search_space.items():
            kind = spec[0]

            if kind == "int":
                _, low, high = spec
                params[name] = trial.suggest_int(name, low, high)

            elif kind == "int_log":
                _, low, high = spec
                params[name] = trial.suggest_int(name, low, high, log=True)

            elif kind == "float":
                _, low, high = spec
                params[name] = trial.suggest_float(name, low, high)

            elif kind == "float_log":
                _, low, high = spec
                params[name] = trial.suggest_float(name, low, high, log=True)

            elif kind == "categorical":
                _, choices = spec
                params[name] = trial.suggest_categorical(name, choices)

            else:
                raise ValueError(f"Unknown search space type '{kind}' for '{name}'.")

        return params

    return _suggest_params


def optimize_regressor_with_optuna(
    X_numerical: List[str],
    X_category: List[str],
    df_raw: pd.DataFrame,
    target: str,
    model_cls: Type[RegressorMixin],
    default_params: Optional[Mapping[str, Any]],
    suggest_params_fn: Callable[[optuna.Trial], Dict[str, Any]],
    n_splits: int = 10,
    random_state: int = 27,
    n_trials: int = 50,
    study_name: Optional[str] = None,
    print_best_params: bool = True,
    save_best_params: bool = True,
    best_params_filename: Optional[Path] = None,
) -> Tuple[optuna.Study, Pipeline]:
    """
    Optimize a regression model using Optuna with K-fold cross-validation.

    Args:
        X_numerical:
            List of numerical feature column names.
        X_category:
            List of categorical feature column names.
        df_raw:
            DataFrame containing both feature and target columns.
        target:
            Name of the target column for prediction.
        model_cls:
            Regressor class to instantiate (e.g. XGBRegressor, RandomForestRegressor).
        default_params:
            Mapping containing default hyperparameters for the model.
        suggest_params_fn:
            Function that takes an Optuna trial and returns a dict of suggested hyperparameters.
        n_splits:
            Number of folds for K-fold cross-validation.
        random_state:
            Seed for reproducible CV splits.
        n_trials:
            Number of Optuna trials to run.
        study_name:
            Optional name for the Optuna study.
        print_best_params:
            Whether to print best parameters at the end.
        save_best_params:
            Whether to save the best parameters to a file.
        best_params_filename:
            Optional filename for storing best parameters.

    Returns:
        Tuple containing:
            - The Optuna study object.
            - The best-performing regression Pipeline instantiated with the best parameters.
    """

    def _generate_default_filename() -> Path:
        algo = model_cls.__name__
        slug = str(target).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(f"log_optuna_{algo}_{slug}_{ts}.txt")

    def _format_params_with_commas(params: Dict[str, Any]) -> str:
        base = json.dumps(params, indent=4)
        lines = base.splitlines()
        out = []
        for line in lines:
            t = line.rstrip()
            if ":" in t and not t.endswith(("{", "[", "}", "]", ",")):
                t += ","
            out.append(t)
        return "\n".join(out)

    if save_best_params:
        if best_params_filename is None:
            resolved_path = _generate_default_filename()
        else:
            resolved_path = (
                best_params_filename if isinstance(best_params_filename, Path) else Path(best_params_filename)
            )
        if resolved_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {resolved_path}")
    else:
        resolved_path = None

    log_file = open(resolved_path, "a", encoding="utf-8") if resolved_path else None

    def _log(text: str):
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
        if log_file:
            log_file.write(text + "\n")
            log_file.flush()

    class _OptunaFileHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = self.format(record)
            _log(msg)

    optuna_handler = None
    if log_file:
        optuna_handler = _OptunaFileHandler()
        optuna_handler.setFormatter(logging.Formatter("%(message)s"))
        optuna_logger = optuna.logging.get_logger("optuna")
        optuna_logger.addHandler(optuna_handler)

    base_params = dict(default_params or {})

    def _objective(trial: optuna.Trial) -> float:
        suggested = suggest_params_fn(trial)
        model_params = {**base_params, **suggested}

        model = make_regression_pipeline(
            X_numerical=X_numerical,
            X_category=X_category,
            model_cls=model_cls,
            default_params=model_params,
        )

        r2_mean, r2_std, rmse_mean, rmse_std = run_kfold_cv(
            X_numerical=X_numerical,
            X_category=X_category,
            df_raw=df_raw,
            model=model,
            target=target,
            n_splits=n_splits,
            random_state=random_state,
        )

        trial.set_user_attr("r2_std", r2_std)
        trial.set_user_attr("rmse_mean", rmse_mean)
        trial.set_user_attr("rmse_std", rmse_std)

        return r2_mean

    try:
        study = optuna.create_study(direction="maximize", study_name=study_name)
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

        best_params = {**base_params, **study.best_params}

        if print_best_params or save_best_params:
            formatted = _format_params_with_commas(best_params)

            _log("\nBest parameters (merged with defaults):")
            _log(formatted)

            if save_best_params and resolved_path:
                _log(f"\nSaved best parameters to: {resolved_path}")

            r2_best = study.best_value
            _log(f"\nBest mean R^2: {r2_best}")

            _log("Best hyperparameters:")
            for k, v in study.best_params.items():
                _log(f"  {k}: {v}")

    finally:
        if optuna_handler is not None:
            optuna.logging.get_logger("optuna").removeHandler(optuna_handler)
        if log_file:
            log_file.close()

    best_pipeline = make_regression_pipeline(
        X_numerical=X_numerical,
        X_category=X_category,
        model_cls=model_cls,
        default_params=best_params,
    )

    return study, best_pipeline


def build_model_pipeline(
    df: pd.DataFrame,
    target_column: str,
    algo: str,
    best_params: Dict,
    pkl_path: Union[str, Path],
    numerical_columns: List[str] = features.X_ALL_NUMERICAL,
    categorical_columns: List[str] = features.X_METADATA_CATEGORICAL,
) -> Pipeline:
    """
    Build, fit, and save a regression Pipeline consisting of preprocessing and a selected model.

    The preprocessing step applies:
        - StandardScaler to numerical features.
        - OneHotEncoder to categorical features.

    The model step uses the algorithm specified by ``algo``, instantiated with ``best_params``.

    Args:
        df:
            DataFrame containing both features and the regression target.
        target_column:
            Name of the target column to be predicted.
        numerical_columns:
            List of numerical feature column names.
        categorical_columns:
            List of categorical feature column names.
        algo:
            Name of the regression algorithm to use.
            Supported values:
            'random_forest', 'svm', 'mlp', 'xgboost', 'xbg', 'lightgbm', 'catboost'.
        best_params:
            Mapping containing hyperparameters selected from the HPO campaign.
        pkl_path:
            Destination path for saving the fitted Pipeline as a .pkl file.

    Returns:
        The fitted sklearn Pipeline object.
    """
    X = df[numerical_columns + categorical_columns]
    y = df[target_column]

    algo = algo.lower()

    if algo == "random_forest":
        regressor = RandomForestRegressor(**best_params)
    elif algo == "svm":
        regressor = SVR(**best_params)
    elif algo == "mlp":
        regressor = MLPRegressor(**best_params)
    elif algo == "xgboost" or algo == "xgb":
        regressor = XGBRegressor(**best_params)
    elif algo == "lightgbm":
        regressor = LGBMRegressor(**best_params)
    elif algo == "catboost":
        regressor = CatBoostRegressor(**best_params)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor),
        ]
    )

    pipeline.fit(X, y)

    pkl_path = Path(pkl_path)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, pkl_path)

    return pipeline
