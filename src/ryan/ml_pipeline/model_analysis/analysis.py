"""
This module contains function to analyze the model performance.
"""

from pathlib import Path
from typing import List, Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import shap
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import features


def cross_val_and_plot(
    X_numerical: list, X_category: list, df_raw: pd.DataFrame, y_columns: list = features.PRODUCTS
) -> None:
    """
    Perform 10-fold cross-validation and plot results for 'fe-H2', 'fe-C2H4', and 'fe-CH4'.

    Parameters:
    X_numerical (list): List of numerical column names
    X_category (list): List of categorical column names
    df_raw (pd.DataFrame): The dataset containing both features and target columns
    y_columns (list): List of target column names (default: ['fe-H2', 'fe-C2H4', 'fe-CH4'])

    Returns:
    None
    """
    # The three gas products
    y_columns = ["fe-H2", "fe-C2H4", "fe-CH4"]

    # Combine numerical and categorical columns into X DataFrame
    X = pd.concat([df_raw[X_numerical], df_raw[X_category].apply(lambda col: col.astype("category"))], axis=1)

    # Initialize the scaler for numerical columns
    scaler = StandardScaler()

    # Loop through the products
    for product in y_columns:
        y = df_raw[product]

        # Set up 10-fold cross-validation
        kf = KFold(n_splits=10, shuffle=True, random_state=27)

        # Define scoring metrics
        r2_scores = []
        rmse_scores = []

        # Perform cross-validation
        for train_index, test_index in kf.split(X):
            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

            # Scale only the numeric columns
            X_train_fold_scaled = X_train_fold.copy()
            X_test_fold_scaled = X_test_fold.copy()
            X_train_fold_scaled[X_numerical] = scaler.fit_transform(X_train_fold[X_numerical])
            X_test_fold_scaled[X_numerical] = scaler.transform(X_test_fold[X_numerical])

            # Initialize the XGBoost model
            model = XGBRegressor(enable_categorical=True, random_state=27)

            # Train the model on the current fold
            model.fit(X_train_fold_scaled, y_train_fold)

            # Predict on the test fold
            y_pred_fold = model.predict(X_test_fold_scaled)

            # Calculate R² and RMSE
            r2_fold = r2_score(y_test_fold, y_pred_fold)
            rmse_fold = root_mean_squared_error(y_test_fold, y_pred_fold)

            # Store the scores
            r2_scores.append(r2_fold)
            rmse_scores.append(rmse_fold)

        # Print the mean and standard deviation of R² and RMSE for each product
        print(f"R^2 for {product}: Mean = {np.mean(r2_scores):.5f}, Std = {np.std(r2_scores):.5f}")
        print(f"RMSE for {product}: Mean = {np.mean(rmse_scores):.5f}, Std = {np.std(rmse_scores):.5f}")

        # Train the final model on the entire dataset
        X_scaled = X.copy()
        X_scaled[X_numerical] = scaler.fit_transform(X[X_numerical])
        model.fit(X_scaled, y)

        # Predict on the entire dataset
        y_pred = model.predict(X_scaled)

        # Plot True vs Predicted Values
        fig = go.Figure()

        # Add scatter plot for true vs predicted values
        fig.add_trace(
            go.Scatter(
                x=y,
                y=y_pred,
                mode="markers",
                marker=dict(color="dodgerblue", opacity=0.35),
                name=f"True vs Predicted for {product}",
            )
        )

        # Add line for perfect predictions
        max_value = max(y.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_value],
                y=[0, max_value],
                mode="lines",
                line=dict(color="black", dash="dash", width=4),
                name="Perfect Predictions",
            )
        )

        # Customize layout
        fig.update_layout(
            title=f"True vs Predicted Values for {product} (After 10-Fold Cross-Validation)",
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

        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

        # Show the True vs Predicted plot
        fig.show()

        # Residuals Calculation and Plot
        residuals = y_pred - y
        residual_fig = go.Figure()

        # Add scatter plot for residuals
        residual_fig.add_trace(
            go.Scatter(
                x=y,
                y=residuals,
                mode="markers",
                marker=dict(color="deeppink", opacity=0.35),
                name=f"Residuals for {product}",
            )
        )

        # Add horizontal line at zero
        residual_fig.add_trace(
            go.Scatter(
                x=[y.min(), y.max()],
                y=[0, 0],
                mode="lines",
                line=dict(color="black", dash="dash", width=4),
                name="Zero Residual Line",
            )
        )

        # Customize layout for residual plot
        residual_fig.update_layout(
            title=f"Residual Plot for {product} (After 10-Fold Cross-Validation)",
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

        # Add grid
        residual_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        residual_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

        # Show the Residual plot
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
    dir_dataset: Union[Path, str],
    model_object: Union[Path, str],
    ls_numerical_columns: List[str] = features.X_ALL_NUMERICAL,
    ls_categorical_columns: List[str] = features.X_METADATA_CATEGORICAL,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Compute SHAP-based feature importance for a trained tree-based model.

    Args:
        dir_dataset: Path to the Excel dataset file.
        ls_numerical_column: List of numerical feature column names.
        categorical_columns: List of categorical feature column names.
        model_object: Path to a serialized model object (.pkl).
        target_column: Name to show in the print header (e.g., a product/target label).
        dropna: If True, drop rows with any NA before processing.

    Returns:
        pd.DataFrame: Feature importance table with columns:
            ['feature', 'relative_importance',
             'normalized_relative_importance (%)',
             'cumulative_importance (%)']
             Indexed starting from 1 (rank order).
    """
    # Ensure paths
    dir_dataset = Path(dir_dataset)
    model_object = Path(model_object)

    # Load data (work on a copy to keep original safe)
    df_raw = pd.read_excel(dir_dataset)
    df = df_raw.copy()
    if dropna:
        df = df.dropna()

    # Build X: cast categoricals, keep numericals
    X_num = df[ls_numerical_columns]
    X_cat = df[ls_categorical_columns].apply(lambda col: col.astype("category"))
    X = pd.concat([X_num, X_cat], axis=1)

    # Scale numeric columns only
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[ls_numerical_columns] = scaler.fit_transform(X[ls_numerical_columns])

    # Load model
    model = joblib.load(model_object)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # Mean absolute SHAP -> relative importance
    feature_importance = np.abs(shap_values).mean(axis=0)

    feature_importance_df = (
        pd.DataFrame({
            "feature": X_scaled.columns,
            "relative_importance": feature_importance,
        })
        .sort_values(by="relative_importance", ascending=False)
        .reset_index(drop=True)
    )

    # Normalize to percentage and cumulative
    total = feature_importance_df["relative_importance"].sum()
    feature_importance_df["normalized_relative_importance (%)"] = (
        feature_importance_df["relative_importance"] / total * 100.0
    )
    feature_importance_df["cumulative_importance (%)"] = feature_importance_df[
        "normalized_relative_importance (%)"
    ].cumsum()

    # Rank index starts from 1
    feature_importance_df.index = feature_importance_df.index + 1

    return feature_importance_df


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
