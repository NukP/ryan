"""
This module contains function to analyze the model performance.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from . import features, ml_util


def cross_val_and_plot(
    X_numerical: List[str],
    X_category: List[str],
    df_raw: pd.DataFrame,
    model: RegressorMixin,
    y_columns: List[str] = features.PRODUCTS,
    n_splits: int = 10,
    random_state: int = 27,
) -> None:
    """
    Perform K-fold cross-validation using a given model and plot results.

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
        y_columns:
            List of target column names.
        n_splits:
            Number of folds for KFold cross-validation.
        random_state:
            Random seed for the KFold splitter.

    Returns:
        None
    """
    # Build feature matrix X once (do not modify df_raw)
    feature_columns = X_numerical + X_category
    X = df_raw[feature_columns].copy()

    base_estimator = clone(model)

    for product in y_columns:
        # Run CV using the helper
        r2_mean, r2_std, rmse_mean, rmse_std = ml_util.run_kfold_cv(
            X_numerical=X_numerical,
            X_category=X_category,
            df_raw=df_raw,
            model=base_estimator,
            target=product,
            n_splits=n_splits,
            random_state=random_state,
        )

        print(f"R^2 for {product}: Mean = {r2_mean:.5f}, Std = {r2_std:.5f}")
        print(f"RMSE for {product}: Mean = {rmse_mean:.5f}, Std = {rmse_std:.5f}")

        # Train final model on all data for this target
        y = df_raw[product]
        final_model = clone(base_estimator)
        final_model.fit(X, y)
        y_pred = final_model.predict(X)

        # ===== True vs Predicted plot =====
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=y,
                y=y_pred,
                mode="markers",
                marker=dict(color="dodgerblue", opacity=0.35),
                name=f"True vs Predicted for {product}",
            )
        )

        min_value = min(y.min(), y_pred.min())
        max_value = max(y.max(), y_pred.max())

        fig.add_trace(
            go.Scatter(
                x=[min_value, max_value],
                y=[min_value, max_value],
                mode="lines",
                line=dict(color="black", dash="dash", width=4),
                name="Perfect Predictions",
            )
        )

        fig.update_layout(
            title=f"True vs Predicted Values for {product} (K={n_splits} CV)",
            xaxis_title="True (experimental) Faradaic Efficiency",
            yaxis_title="Predicted Faradaic Efficiency",
            legend=dict(x=0.02, y=0.98),
            width=800,
            height=600,
            margin=dict(l=40, r=40, b=40, t=40),
            plot_bgcolor="white",
            hovermode="closest",
            xaxis=dict(
                showline=True,
                linewidth=2,
                linecolor="black",
                mirror=True,
                title_font=dict(size=16, color="black"),
                tickfont=dict(size=14, color="black"),
            ),
            yaxis=dict(
                showline=True,
                linewidth=2,
                linecolor="black",
                mirror=True,
                title_font=dict(size=18, color="black"),
                tickfont=dict(size=16, color="black"),
            ),
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

        fig.show()

        # ===== Residual plot =====
        residuals = y_pred - y
        residual_fig = go.Figure()

        residual_fig.add_trace(
            go.Scatter(
                x=y,
                y=residuals,
                mode="markers",
                marker=dict(color="deeppink", opacity=0.35),
                name=f"Residuals for {product}",
            )
        )

        residual_fig.add_trace(
            go.Scatter(
                x=[y.min(), y.max()],
                y=[0, 0],
                mode="lines",
                line=dict(color="black", dash="dash", width=4),
                name="Zero Residual Line",
            )
        )

        residual_fig.update_layout(
            title=f"Residual Plot for {product} (K={n_splits} CV)",
            xaxis_title="True (experimental) Faradaic Efficiency",
            yaxis_title="Residuals (Predicted - True)",
            legend=dict(x=0.02, y=0.98),
            width=800,
            height=600,
            margin=dict(l=40, r=40, b=40, t=40),
            plot_bgcolor="white",
            hovermode="closest",
            xaxis=dict(
                showline=True,
                linewidth=2,
                linecolor="black",
                mirror=True,
                title_font=dict(size=16, color="black"),
                tickfont=dict(size=14, color="black"),
            ),
            yaxis=dict(
                showline=True,
                linewidth=2,
                linecolor="black",
                mirror=True,
                title_font=dict(size=18, color="black"),
                tickfont=dict(size=16, color="black"),
            ),
        )

        residual_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        residual_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

        residual_fig.show()


def plot_correlation_matrix(
    df_dataset: pd.DataFrame,
    ls_columns: list = [
        f
        for f in (features.PRODUCTS + features.X_RAW_DATA + features.X_METADATA_NUMERICAL + features.X_ENGINEERED)
        if f != "cathode_catalyst_loading"
    ],
    save_fig: bool = False,
) -> None:
    """
    Plot Pearson and Spearman correlation matrices for the given dataset.

    Args:
        df_dataset (pd.DataFrame): The dataset containing features and target variables.
        ls_columns (list): List of columns to include in the correlation matrix.
                            In this default, cathode_catalyst_loading is excluded as we found the value are all 0.5.
        savefig (bool): Whether to save the figures as PNG files. Default is False.

    Returns:
        None
    """
    df_to_calculate = pd.concat([df_dataset[ls_columns]], axis=1).dropna()
    df_to_calculate = df_to_calculate.fillna(df_to_calculate.mean())  # Fill NaNs with column mean
    pearson_corr = df_to_calculate.corr(method="pearson")
    spearmann_corr = df_to_calculate.corr(method="spearman")

    ls_to_plot = [pearson_corr, spearmann_corr]
    ls_titles = ["Pearson Correlation Matrix", "Spearman Correlation Matrix"]
    for idx, corr_matrix in enumerate(ls_to_plot):
        plt.figure(figsize=(40, 40))

        # Get minimum and maximum correlation for tick range
        min_val = corr_matrix.min().min()
        max_val = corr_matrix.max().max()

        # Create heatmap with cbar_kws for controlling the colorbar
        ax = sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".3f",
            cbar_kws={
                "shrink": 0.8,  # Adjust colorbar size
                "ticks": np.linspace(min_val, max_val, 21),  # More frequent ticks
            },
        )

        # Make title larger and bold
        plt.title(ls_titles[idx], fontsize=20, fontweight="bold")

        # Optionally adjust colorbar label sizes
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=12)
        if save_fig:
            filename_no_ext = ls_titles[idx].replace(" ", "_")
            filename_ext = filename_no_ext + ".png"
            plt.savefig(filename_ext, bbox_inches="tight")
            plt.show()


def plot_commulative_histogram(
    column_name: str,
    df: pd.DataFrame,
    bins: int,
    xlim: Optional[tuple[float, float]] = None,
    dir_save_fig: Optional[str] = None,
) -> None:
    """ "Plot histogram of a column with cumulative percentage

    Args:
    column_name: str: Name of the column to plot
    df: pd.DataFrame: Dataframe containing the data
    bins: int: Number of bins for the histogram
    xlim: tuple(int,int): Tuple with the minimum and maximum values for the x-axis
    dir_save_fig: str: Directory to save the figure. If None, the figure is not saved.

    Returns:
    None
    """
    _, ax = plt.subplots()
    ax.minorticks_on()

    # Get histogram data
    counts, bins, _ = plt.hist(df[column_name].dropna(), bins=bins, edgecolor="k", alpha=1, color="#FFFFB5")

    # Compute cumulative percentage
    cumulative_counts = np.cumsum(counts)
    cumulative_percentage = cumulative_counts / cumulative_counts[-1] * 100  # Convert to percentage

    # Plot cumulative percentage
    ax2 = ax.twinx()
    ax2.plot(bins[:-1], cumulative_percentage, color="red", marker=".", linestyle="-", alpha=0.5)
    ax2.set_ylabel("Cumulative Percentage (%)", fontsize=12, color="red")
    ax2.tick_params(axis="y", colors="black")
    ax2.set_yticks(np.arange(0, 101, 10))
    ax2.minorticks_on()

    plt.title(f"Cumulative Distribution of {column_name}")
    plt.xlabel(column_name, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax2.grid(True, which="major", linestyle="-", linewidth=0.5)
    ax2.set_ylim(0, 105)

    if xlim:
        xmin, xmax = xlim
        ax.set_xlim(xmin, xmax)

    if dir_save_fig is not None:
        filename = f"{dir_save_fig}/{column_name}_cumulative_histogram.png"
        plt.savefig(filename, bbox_inches="tight")
    plt.show()


def compute_shap_feature_importance(
    df_dataset: pd.DataFrame,
    model_object: Union[Path, str],
    ls_numerical_columns: List[str] = features.X_ALL_NUMERICAL,
    ls_categorical_columns: List[str] = features.X_METADATA_CATEGORICAL,
) -> pd.DataFrame:
    """
    Compute SHAP based feature importance for a trained regression Pipeline.

    The function loads a fitted sklearn Pipeline from disk, applies its
    preprocessing to the given dataset, and computes mean absolute SHAP
    values as a measure of feature importance.

    For tree based final estimators (e.g. XGBRegressor, LGBMRegressor,
    CatBoostRegressor, RandomForestRegressor), TreeExplainer is used.
    For other regressors (e.g. SVM, MLP), a model agnostic explainer
    based on ``model.predict`` is used.

    Args:
        df_dataset:
            DataFrame containing the dataset with all feature columns.
        model_object:
            Path to a serialized sklearn Pipeline object (.pkl) that contains
            both preprocessing (e.g. ColumnTransformer) and the regression model.
        ls_numerical_columns:
            List of numerical feature column names.
        ls_categorical_columns:
            List of categorical feature column names.

    Returns:
        DataFrame containing SHAP based feature importance with the columns:
            - 'feature'
            - 'relative_importance'
            - 'normalized_relative_importance (%)'
            - 'cumulative_importance (%)'

        The index starts at 1 and represents the rank (1 is most important).
    """
    model_path = Path(model_object)

    feature_columns = ls_numerical_columns + ls_categorical_columns
    df = df_dataset.copy()
    X = df[feature_columns]

    pipeline: Pipeline = joblib.load(model_path)

    # Expect pipeline structure: [("preprocessor", ...), ("model", ...)]
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
    except (AttributeError, KeyError):
        # Fallback: treat the loaded object as a generic estimator
        explainer = shap.Explainer(pipeline.predict, X)
        shap_result = explainer(X)
        shap_values = shap_result.values
        feature_names = list(X.columns)
    else:
        # Transform X to the representation the model actually sees
        X_processed = preprocessor.transform(X)

        # Feature names after preprocessing (e.g. one hot encoded categories)
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = list(preprocessor.get_feature_names_out())
        else:
            feature_names = [str(i) for i in range(X_processed.shape[1])]

        # Use fast tree-specific explainer when possible
        if isinstance(
            model,
            (
                XGBRegressor,
                LGBMRegressor,
                CatBoostRegressor,
                RandomForestRegressor,
            ),
        ):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_processed)
        else:
            # Generic fallback for non tree models (e.g. SVM, MLP)
            explainer = shap.Explainer(model.predict, X_processed)
            shap_result = explainer(X_processed)
            shap_values = shap_result.values

    feature_importance = np.abs(shap_values).mean(axis=0)

    feature_importance_df = (
        pd.DataFrame({
            "feature": feature_names,
            "relative_importance": feature_importance,
        })
        .sort_values(by="relative_importance", ascending=False)
        .reset_index(drop=True)
    )

    total = feature_importance_df["relative_importance"].sum()
    feature_importance_df["normalized_relative_importance (%)"] = (
        feature_importance_df["relative_importance"] / total * 100.0
    )
    feature_importance_df["cumulative_importance (%)"] = feature_importance_df[
        "normalized_relative_importance (%)"
    ].cumsum()

    feature_importance_df.index = feature_importance_df.index + 1

    return feature_importance_df


# !! This function recompute the model, and do not take into account the hyper-parameter tuning.
def plot_shap_scatter_checkbox(
    column_name: str,
    dir_dataset: Union[str, Path],
    dir_model_h2: Union[str, Path, XGBRegressor],
    dir_model_c2h4: Union[str, Path, XGBRegressor],
    dir_model_ch4: Union[str, Path, XGBRegressor],
    ls_numerical_columns: Optional[List[str]] = None,
    ls_categorical_columns: Optional[List[str]] = None,
    color_H2: str = "orange",
    color_C2H4: str = "dodgerblue",
    color_CH4: str = "deeppink",
    dropna: bool = True,
) -> None:
    # --- helpers in-function ---
    def _ensure_model(model_or_path):
        if isinstance(model_or_path, XGBRegressor):
            return model_or_path
        p = Path(model_or_path)
        ext = p.suffix.lower()
        if ext in {".json", ".ubj"}:
            m = XGBRegressor()
            m.load_model(str(p))
            return m
        if ext in {".pkl", ".pickle"}:
            return joblib.load(p)
        raise ValueError(f"Unsupported model type/path: {model_or_path}")

    def _shap_for_model(model, X_in: pd.DataFrame) -> np.ndarray:
        exp = shap.TreeExplainer(model)
        sv = exp.shap_values(X_in)
        if isinstance(sv, list):
            return np.mean([np.abs(a) for a in sv], axis=0)
        return sv

    # --- defaults from your features module ---
    if ls_numerical_columns is None:
        ls_numerical_columns = list(features.X_ALL_NUMERICAL)
    else:
        ls_numerical_columns = list(ls_numerical_columns)
    if ls_categorical_columns is None:
        ls_categorical_columns = list(features.X_METADATA_CATEGORICAL)
    else:
        ls_categorical_columns = list(ls_categorical_columns)

    # --- load & prep data ---
    dir_dataset = Path(dir_dataset)
    df = pd.read_excel(dir_dataset)
    if dropna:
        df = df.dropna()
    df[ls_categorical_columns] = df[ls_categorical_columns].apply(lambda c: c.astype("category"))
    X = pd.concat([df[ls_numerical_columns], df[ls_categorical_columns]], axis=1)
    if column_name not in X.columns:
        raise ValueError(f"'{column_name}' not found in dataset columns.")
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[ls_numerical_columns] = scaler.fit_transform(X[ls_numerical_columns])

    # --- models & SHAP ---
    model_H2 = _ensure_model(dir_model_h2)
    model_C2H4 = _ensure_model(dir_model_c2h4)
    model_CH4 = _ensure_model(dir_model_ch4)

    shap_H2 = _shap_for_model(model_H2, X_scaled)
    shap_C2H4 = _shap_for_model(model_C2H4, X_scaled)
    shap_CH4 = _shap_for_model(model_CH4, X_scaled)

    idx = X_scaled.columns.get_loc(column_name)
    x_vals = X[column_name]

    # --- figure (no tabs; boxed axes; legend on the left; larger fonts) ---
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x_vals, y=shap_H2[:, idx], mode="markers", marker=dict(color=color_H2, opacity=0.35), name="fe-H₂")
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=shap_C2H4[:, idx], mode="markers", marker=dict(color=color_C2H4, opacity=0.35), name="fe-C₂H₄"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=shap_CH4[:, idx], mode="markers", marker=dict(color=color_CH4, opacity=0.35), name="fe-CH₄"
        )
    )

    fig.update_layout(
        title=f"SHAP values for {column_name}",
        xaxis_title=column_name,
        yaxis_title="SHAP value",
        width=1400 * 0.9,
        height=600 * 0.9,
        showlegend=True,
        font=dict(size=16, color="black"),
        legend=dict(
            title="Products",
            orientation="v",
            y=0.5,
            yanchor="middle",
            x=-0.18,
            xanchor="right",
            bgcolor="rgba(255,255,255,0.0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=16, color="black"),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, b=60, t=60),
    )

    # Boxed axes (top/right lines via mirror)
    fig.update_xaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        tickfont=dict(size=16, color="black"),
        title_font=dict(size=18, color="black"),
    )
    fig.update_yaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        tickfont=dict(size=16, color="black"),
        title_font=dict(size=18, color="black"),
    )

    # zero line
    fig.add_shape(
        type="line",
        xref="paper",
        x0=0,
        x1=1,  # always full width of plot, zoom-proof
        yref="y",
        y0=0,
        y1=0,  # y=0 in data coordinates
        line=dict(color="black", width=2, dash="dash"),
        layer="above",  # ensure visibility above scatter points
    )

    fig.show()


def eval_feature_ablation(
    df_source: Union[pd.DataFrame, str, Path],
    drop_features: List[str],
    x_columns: Optional[List[str]] = None,
    x_categorical: Optional[List[str]] = None,
    target_columns: Optional[List[str]] = None,
    n_splits: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compare baseline vs. retrain-without-selected-features for each product.

    Args:
        df_source: DataFrame (left intact) or path to an Excel file.
        drop_features: Columns to exclude in the ablation model (compared to baseline with all X).
        x_columns: Feature columns to use for baseline. Defaults to features.X_ALL.
        x_categorical: Subset of x_columns that are categorical. Defaults to features.X_METADATA_CATEGORICAL.
        target_columns: Products/targets to evaluate. Defaults to features.PRODUCTS.
        n_splits: KFold splits.
        random_state: Seed for KFold shuffling.

    Returns:
        DataFrame with one row per product (target) and these columns:
          baseline_r2_mean, baseline_r2_std, baseline_rmse_mean, baseline_rmse_std,
          drop_r2_mean, drop_r2_std, drop_rmse_mean, drop_rmse_std,
          delta_drop_r2_mean, delta_drop_r2_std, delta_drop_rmse_mean, delta_drop_rmse_std
    """
    # ---- resolve inputs / defaults
    if isinstance(df_source, (str, Path)):
        df_raw = pd.read_excel(Path(df_source))
    else:
        df_raw = df_source.copy()

    df_raw = df_raw.dropna().copy()

    if x_columns is None:
        x_columns = list(features.X_ALL)
    if x_categorical is None:
        x_categorical = list(features.X_METADATA_CATEGORICAL)
    if target_columns is None:
        target_columns = list(features.PRODUCTS)

    present_x = [c for c in x_columns if c in df_raw.columns]
    if not present_x:
        raise ValueError("None of the requested x_columns are present in the dataframe.")
    X_full = df_raw[present_x].copy()

    cat_features_present = [c for c in x_categorical if c in X_full.columns]

    targets_to_use = [t for t in target_columns if t in df_raw.columns]
    if not targets_to_use:
        raise ValueError("None of the requested target_columns are present in the dataframe.")

    print(f"n_samples={len(X_full)}, n_features={X_full.shape[1]}")
    print("Targets to evaluate:", targets_to_use)

    # sanitize drop_features to existing columns
    drop_features_present = [c for c in drop_features if c in X_full.columns]
    missing = [c for c in drop_features if c not in X_full.columns]
    if missing:
        print(f"[WARN] The following drop_features are not present in X and will be ignored: {missing}")

    # ---- helpers (all internal)
    def _make_model() -> XGBRegressor:
        return XGBRegressor(enable_categorical=True, random_state=27)

    def _build_pipeline(cols: List[str], cat_present: List[str]) -> Pipeline:
        cat_in = [c for c in cols if c in cat_present]
        num_in = [c for c in cols if c not in cat_in]
        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_in),
                ("num", "passthrough", num_in),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )
        return Pipeline([("pre", pre), ("model", _make_model())])

    def _get_cv_splits(n_samples: int, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        idx = np.arange(n_samples)
        return list(kf.split(idx))

    def _cv_scores(
        X: pd.DataFrame,
        y: np.ndarray,
        cols: List[str],
        cat_present: List[str],
        splits: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        r2_list, rmse_list = [], []
        for train_idx, val_idx in splits:
            X_tr = X.iloc[train_idx][cols]
            X_va = X.iloc[val_idx][cols]
            y_tr = y[train_idx]
            y_va = y[val_idx]

            pipe = _build_pipeline(cols, cat_present)
            pipe.fit(X_tr, y_tr)
            yhat = pipe.predict(X_va)

            r2_list.append(r2_score(y_va, yhat))
            rmse_list.append(root_mean_squared_error(y_va, yhat))
        return np.asarray(r2_list, float), np.asarray(rmse_list, float)

    def _mean_std(arr_like: Union[List[float], np.ndarray]) -> Tuple[float, float]:
        arr = np.asarray(arr_like, dtype=float)
        mu = float(arr.mean())
        sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        return mu, sd

    # ---- main loop over products; one row per product
    rows: List[Dict[str, float]] = []
    for target in targets_to_use:
        print("\n" + "=" * 80)
        print(f"TARGET: {target}")
        print("=" * 80)

        y_full = df_raw[target].to_numpy()
        splits = _get_cv_splits(n_samples=len(X_full), n_folds=n_splits, seed=random_state)

        # Baseline: train with all x_columns present
        base_cols = X_full.columns.tolist()
        base_r2, base_rmse = _cv_scores(
            X=X_full, y=y_full, cols=base_cols, cat_present=cat_features_present, splits=splits
        )
        base_r2_m, base_r2_s = _mean_std(base_r2)
        base_rmse_m, base_rmse_s = _mean_std(base_rmse)

        print(f"\nBaseline (10-fold CV) for {target}:")
        print(f"  R²   = {base_r2_m:.4f} ± {base_r2_s:.4f}")
        print(f"  RMSE = {base_rmse_m:.6f} ± {base_rmse_s:.6f}")

        # Print the exact features requested to drop (after baseline)
        print("\nFeatures to drop (evaluated in the retrain-without-feature model):")
        print(drop_features_present)

        # Retrain without selected features
        cols_keep = [c for c in base_cols if c not in drop_features_present]
        if len(cols_keep) == 0:
            print("[WARN] Dropping all features leaves no predictors; results will be NaN.")
        drop_r2, drop_rmse = _cv_scores(
            X=X_full, y=y_full, cols=cols_keep, cat_present=cat_features_present, splits=splits
        )

        # Deltas per fold
        delta_r2 = drop_r2 - base_r2
        delta_rmse = drop_rmse - base_rmse

        row = {
            "product": target,
            "baseline_r2_mean": float(base_r2_m),
            "baseline_r2_std": float(base_r2_s),
            "baseline_rmse_mean": float(base_rmse_m),
            "baseline_rmse_std": float(base_rmse_s),
            "drop_r2_mean": float(drop_r2.mean()),
            "drop_r2_std": float(drop_r2.std(ddof=1)) if len(drop_r2) > 1 else 0.0,
            "drop_rmse_mean": float(drop_rmse.mean()),
            "drop_rmse_std": float(drop_rmse.std(ddof=1)) if len(drop_rmse) > 1 else 0.0,
            "delta_drop_r2_mean": float(delta_r2.mean()),
            "delta_drop_r2_std": float(delta_r2.std(ddof=1)) if len(delta_r2) > 1 else 0.0,
            "delta_drop_rmse_mean": float(delta_rmse.mean()),
            "delta_drop_rmse_std": float(delta_rmse.std(ddof=1)) if len(delta_rmse) > 1 else 0.0,
            "delta_r2_significant?": "✅" if float((-1) * delta_r2.mean()) > float(base_r2_s) else "❌",
            "delta_rmse_significant?": "✅" if float((-1) * delta_rmse.mean()) > float(base_rmse_s) else "❌",
        }
        rows.append(row)

    df_results = pd.DataFrame(rows).set_index("product")
    # Order columns for readability
    preferred = [
        "delta_drop_r2_mean",
        "delta_r2_significant?",
        "delta_rmse_significant?",
        "delta_drop_r2_std",
        "delta_drop_rmse_mean",
        "delta_drop_rmse_std",
        "drop_r2_mean",
        "drop_r2_std",
        "drop_rmse_mean",
        "drop_rmse_std",
        "baseline_r2_mean",
        "baseline_r2_std",
        "baseline_rmse_mean",
        "baseline_rmse_std",
    ]
    df_results = df_results.reindex(columns=preferred)
    try:
        from IPython.display import display  # type: ignore

        print("\n--- Retrain-without-selected-features (rows = product) ---")
        display(df_results)
    except Exception:
        pass

    return df_results


def plot_shap_multi_models(
    column_name: str,
    df_dataset: pd.DataFrame,
    shap_xgb_path: Union[str, Path],
    shap_lightgbm_path: Union[str, Path],
    shap_catboost_path: Union[str, Path],
    shap_rf_path: Union[str, Path],
    X_numerical: List[str] = features.X_ALL_NUMERICAL,
    X_category: List[str] = features.X_METADATA_CATEGORICAL,
) -> None:
    """
    Plot SHAP dependence for a single *input feature* across four tree-based models,
    using precomputed SHAP tables saved as .pkl files.

    For a given input feature `column_name`, this function produces a scatter plot:
        x-axis: raw values of `column_name` from df_dataset
        y-axis: SHAP values for `column_name` (per model)

    SHAP tables are expected to be created by `compute_and_save_shap_table`, i.e.
    each .pkl contains a dict with keys:
        - "shap_values":   (n_samples, n_transformed_features)
        - "feature_names": (n_transformed_features,)

    Notes:
        - `column_name` MUST be in the input feature space, i.e. in
          `X_numerical` or `X_category`.
        - Target/FE columns (like 'fe-H2' if they are the regression target) are
          NOT valid here unless they are also part of the model input.

    Args:
        column_name:
            Name of the *input* feature to analyse.
        df_dataset:
            Cleaned DataFrame containing at least all input features.
            Must be the same row order as used for SHAP computation.
        shap_xgb_path, shap_lgbm_path, shap_catboost_path, shap_rf_path:
            Paths to .pkl SHAP tables generated for each model.
        X_numerical:
            Numerical input feature names (default from features.X_ALL_NUMERICAL).
        X_category:
            Categorical input feature names (default from features.X_METADATA_CATEGORICAL).
    """

    # -------------------------------------------------------------------------
    # Sanity: column_name must be an input feature
    # -------------------------------------------------------------------------
    if column_name not in X_numerical and column_name not in X_category:
        raise ValueError(
            f"column_name='{column_name}' must be in numerical or categorical input "
            f"features, but it is not.\n"
            f"  numerical: {column_name in X_numerical}\n"
            f"  categorical: {column_name in X_category}"
        )

    # -------------------------------------------------------------------------
    # Helper: map original feature -> indices in transformed feature_names
    # -------------------------------------------------------------------------
    def _get_feature_indices_for_column(
        feature_names: np.ndarray,
        original_column: str,
        numerical_columns: List[str],
        categorical_columns: List[str],
    ) -> List[int]:
        """
        Map original feature -> indices in transformed feature space.

        Assumes ColumnTransformer naming from build_model_pipeline:
            "num__<num_feature>"
            "cat__<cat_feature>_<category>"
        """
        indices: List[int] = []

        if original_column in numerical_columns:
            suffix = f"__{original_column}"
            indices = [i for i, name in enumerate(feature_names) if name.endswith(suffix)]
        elif original_column in categorical_columns:
            prefix = f"cat__{original_column}_"
            indices = [i for i, name in enumerate(feature_names) if name.startswith(prefix)]
        else:
            raise ValueError(f"Column '{original_column}' not found in numerical or categorical columns.")

        if not indices:
            raise ValueError(
                f"No transformed features found for original column '{original_column}'. "
                "Check feature_names and ColumnTransformer structure."
            )

        return indices

    def _load_shap(path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray]:
        payload = joblib.load(path)
        shap_values = payload["shap_values"]
        feature_names = payload["feature_names"]
        return shap_values, feature_names

    def _extract_feature_shap(
        shap_values: np.ndarray,
        feature_names: np.ndarray,
        feature_name: str,
        numerical_columns: List[str],
        categorical_columns: List[str],
    ) -> np.ndarray:
        """
        Given a full SHAP matrix for a model, extract SHAP values for a single
        original feature (aggregating over OHE dummies if needed).
        """
        idx = _get_feature_indices_for_column(feature_names, feature_name, numerical_columns, categorical_columns)
        shap_feat = shap_values[:, idx]  # (n_samples, k_dummies or 1)

        if shap_feat.ndim == 2 and shap_feat.shape[1] > 1:
            return shap_feat.sum(axis=1)  # aggregate categoricals
        return shap_feat.ravel()

    # -------------------------------------------------------------------------
    # x-axis values
    # -------------------------------------------------------------------------
    if column_name not in df_dataset.columns:
        raise ValueError(f"'{column_name}' not found in df_dataset columns.")

    x_vals = df_dataset[column_name]

    # -------------------------------------------------------------------------
    # Load SHAP tables
    # -------------------------------------------------------------------------
    shap_xgb, fn_xgb = _load_shap(shap_xgb_path)
    shap_lgbm, fn_lgbm = _load_shap(shap_lightgbm_path)
    shap_cat, fn_cat = _load_shap(shap_catboost_path)
    shap_rf, fn_rf = _load_shap(shap_rf_path)

    # Optional sanity: check sample counts match df_dataset
    n_samples = len(df_dataset)
    for name, arr in [
        ("XGBoost", shap_xgb),
        ("LightGBM", shap_lgbm),
        ("CatBoost", shap_cat),
        ("Random Forest", shap_rf),
    ]:
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"SHAP table for {name} has {arr.shape[0]} samples, but df_dataset has {n_samples}. They must match."
            )

    # -------------------------------------------------------------------------
    # Extract SHAP for this feature from each model
    # -------------------------------------------------------------------------
    y_xgb = _extract_feature_shap(shap_xgb, fn_xgb, column_name, X_numerical, X_category)
    y_lgbm = _extract_feature_shap(shap_lgbm, fn_lgbm, column_name, X_numerical, X_category)
    y_cat = _extract_feature_shap(shap_cat, fn_cat, column_name, X_numerical, X_category)
    y_rf = _extract_feature_shap(shap_rf, fn_rf, column_name, X_numerical, X_category)

    # -------------------------------------------------------------------------
    # Plot: x = feature value, y = SHAP(feature)
    # -------------------------------------------------------------------------
    algos = ["xgb", "catboost", "lightgbm", "rf"]
    palette_hex = sns.color_palette("tab10", n_colors=len(algos)).as_hex()
    algo_to_color = {algo: palette_hex[i] for i, algo in enumerate(algos)}

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_xgb,
            mode="markers",
            marker=dict(color=algo_to_color["xgb"], opacity=0.35),
            name="XGBoost",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_lgbm,
            mode="markers",
            marker=dict(color=algo_to_color["lightgbm"], opacity=0.35),
            name="LightGBM",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_cat,
            mode="markers",
            marker=dict(color=algo_to_color["catboost"], opacity=0.35),
            name="CatBoost",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_rf,
            mode="markers",
            marker=dict(color=algo_to_color["rf"], opacity=0.35),
            name="Random Forest",
        )
    )

    fig.update_layout(
        title=f"SHAP values for feature {column_name}",
        xaxis_title=column_name,
        yaxis_title="SHAP value",
        width=1400 * 0.9,
        height=600 * 0.9,
        showlegend=True,
        font=dict(size=16, color="black"),
        legend=dict(
            title="Algorithm",
            orientation="v",
            y=0.5,
            yanchor="middle",
            x=-0.18,
            xanchor="right",
            bgcolor="rgba(255,255,255,0.0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=16, color="black"),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, b=60, t=60),
    )

    # Boxed axes
    fig.update_xaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        tickfont=dict(size=16, color="black"),
        title_font=dict(size=18, color="black"),
    )
    fig.update_yaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        tickfont=dict(size=16, color="black"),
        title_font=dict(size=18, color="black"),
    )

    # Zero line
    fig.add_shape(
        type="line",
        xref="paper",
        x0=0,
        x1=1,
        yref="y",
        y0=0,
        y1=0,
        line=dict(color="black", width=2, dash="dash"),
        layer="above",
    )

    fig.show()


def plot_shap_multi_products(
    column_name: str,
    df_dataset: pd.DataFrame,
    shap_H2_path: Union[str, Path],
    shap_C2H4_path: Union[str, Path],
    shap_CH4_path: Union[str, Path],
    shap_1_propanol_path: Union[str, Path],
    shap_Acetaldehyde_path: Union[str, Path],
    shap_EtOH_path: Union[str, Path],
    X_numerical: List[str] = features.X_ALL_NUMERICAL,
    X_category: List[str] = features.X_METADATA_CATEGORICAL,
) -> None:
    """
    Plot SHAP dependence for a single *input feature* across multiple product models,
    using precomputed SHAP tables saved as .pkl files.

    For a given input feature `column_name`, this function produces a scatter plot:
        x-axis: raw values of `column_name` from df_dataset
        y-axis: SHAP values for `column_name` (per product-specific model)

    SHAP tables are expected to be created by `ml_util.compute_shap_table`, i.e.
    each .pkl contains a dict with keys:
        - "shap_values":   (n_samples, n_transformed_features)
        - "feature_names": (n_transformed_features,)

    Args:
        column_name:
            Name of the *input* feature to analyse.
        df_dataset:
            Cleaned DataFrame containing at least all input features.
            Must be the same row order as used for SHAP computation.
        shap_H2_path, shap_C2H4_path, shap_CH4_path, shap_1_propanol_path,
        shap_Acetaldehyde_path, shap_EtOH_path:
            Paths to .pkl SHAP tables generated for each product model.
            Argument names mirror `features.PRODUCTS` without the 'fe-' prefix.
        X_numerical:
            Numerical input feature names (default from features.X_ALL_NUMERICAL).
        X_category:
            Categorical input feature names (default from features.X_METADATA_CATEGORICAL).
    """

    if column_name not in X_numerical and column_name not in X_category:
        raise ValueError(
            f"column_name='{column_name}' must be in numerical or categorical input features.\n"
            f"  numerical: {column_name in X_numerical}\n"
            f"  categorical: {column_name in X_category}"
        )
    if column_name not in df_dataset.columns:
        raise ValueError(f"'{column_name}' not found in df_dataset columns.")

    def _get_feature_indices_for_column(
        feature_names: np.ndarray,
        original_column: str,
        numerical_columns: List[str],
        categorical_columns: List[str],
    ) -> List[int]:
        indices: List[int] = []

        if original_column in numerical_columns:
            suffix = f"__{original_column}"
            indices = [i for i, name in enumerate(feature_names) if name.endswith(suffix)]
        elif original_column in categorical_columns:
            prefix = f"cat__{original_column}_"
            indices = [i for i, name in enumerate(feature_names) if name.startswith(prefix)]
        else:
            raise ValueError(f"Column '{original_column}' not found in numerical or categorical columns.")

        if not indices:
            raise ValueError(
                f"No transformed features found for original column '{original_column}'. "
                "Check feature_names and ColumnTransformer structure."
            )

        return indices

    def _load_shap(path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray]:
        payload = joblib.load(path)
        shap_values = payload["shap_values"]
        feature_names = payload["feature_names"]
        return shap_values, feature_names

    def _extract_feature_shap(
        shap_values: np.ndarray,
        feature_names: np.ndarray,
        feature_name: str,
        numerical_columns: List[str],
        categorical_columns: List[str],
    ) -> np.ndarray:
        idx = _get_feature_indices_for_column(feature_names, feature_name, numerical_columns, categorical_columns)
        shap_feat = shap_values[:, idx]
        if shap_feat.ndim == 2 and shap_feat.shape[1] > 1:
            return shap_feat.sum(axis=1)
        return shap_feat.ravel()

    x_vals = df_dataset[column_name]

    product_paths = [
        ("fe-H2", "fe-H₂", shap_H2_path),
        ("fe-C2H4", "fe-C₂H₄", shap_C2H4_path),
        ("fe-CH4", "fe-CH₄", shap_CH4_path),
        ("fe-1-propanol", "fe-1-propanol", shap_1_propanol_path),
        ("fe-Acetaldehyde", "fe-Acetaldehyde", shap_Acetaldehyde_path),
        ("fe-EtOH", "fe-EtOH", shap_EtOH_path),
    ]

    loaded_products: List[tuple[str, str, np.ndarray, np.ndarray]] = []
    n_samples = len(df_dataset)
    for product_id, product_label, path in product_paths:
        shap_vals, feature_names = _load_shap(path)
        if shap_vals.shape[0] != n_samples:
            raise ValueError(
                f"SHAP table for {product_id} has {shap_vals.shape[0]} samples, but df_dataset has {n_samples}. "
                "They must match."
            )
        loaded_products.append((product_id, product_label, shap_vals, feature_names))

    product_to_color = {
        "fe-H2": "orange",
        "fe-C2H4": "dodgerblue",
        "fe-CH4": "deeppink",
        "fe-1-propanol": "mediumseagreen",
        "fe-Acetaldehyde": "mediumpurple",
        "fe-EtOH": "firebrick",
    }

    fig = go.Figure()

    for product_id, product_label, shap_vals, feature_names in loaded_products:
        y_vals = _extract_feature_shap(shap_vals, feature_names, column_name, X_numerical, X_category)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=dict(color=product_to_color[product_id], opacity=0.35),
                name=product_label,
            )
        )

    fig.update_layout(
        title=f"SHAP values for feature {column_name} across products",
        xaxis_title=column_name,
        yaxis_title="SHAP value",
        width=1400 * 0.9,
        height=600 * 0.9,
        showlegend=True,
        font=dict(size=16, color="black"),
        legend=dict(
            title="Product",
            orientation="v",
            y=0.5,
            yanchor="middle",
            x=-0.18,
            xanchor="right",
            bgcolor="rgba(255,255,255,0.0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=16, color="black"),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, b=60, t=60),
    )

    fig.update_xaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        tickfont=dict(size=16, color="black"),
        title_font=dict(size=18, color="black"),
    )
    fig.update_yaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
        tickfont=dict(size=16, color="black"),
        title_font=dict(size=18, color="black"),
    )

    fig.add_shape(
        type="line",
        xref="paper",
        x0=0,
        x1=1,
        yref="y",
        y0=0,
        y1=0,
        line=dict(color="black", width=2, dash="dash"),
        layer="above",
    )

    fig.show()
