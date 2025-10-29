"""
This module contains function to analyze the model performance.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor
import plotly.graph_objects as go
import numpy as np
from . import features

def cross_val_and_plot(X_numerical:list,
                    X_category:list,
                    df_raw:pd.DataFrame,
                    y_columns:list=features.PRODUCTS
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
    y_columns = ['fe-H2', 'fe-C2H4', 'fe-CH4']
    
    # Combine numerical and categorical columns into X DataFrame
    X = pd.concat([df_raw[X_numerical], df_raw[X_category].apply(lambda col: col.astype('category'))], axis=1)

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
        fig.add_trace(go.Scatter(
            x=y,
            y=y_pred,
            mode='markers',
            marker=dict(color='dodgerblue', opacity=0.35),
            name=f'True vs Predicted for {product}'
        ))

        # Add line for perfect predictions
        max_value = max(y.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[0, max_value],
            y=[0, max_value],
            mode='lines',
            line=dict(color='black', dash='dash', width=4),
            name='Perfect Predictions'
        ))

        # Customize layout
        fig.update_layout(
            title=f'True vs Predicted Values for {product} (After 10-Fold Cross-Validation)',
            xaxis_title='True (experimental) Faradaic Efficiency',
            yaxis_title='Predicted Faradaic Efficiency',
            legend=dict(x=0.02, y=0.98),
            width=800,
            height=600,
            margin=dict(l=40, r=40, b=40, t=40),
            plot_bgcolor='white',
            hovermode='closest',
            xaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror=True,
                title_font=dict(size=16, color='black'),
                tickfont=dict(size=14, color='black')
            ),
            yaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror=True,
                title_font=dict(size=18, color='black'),
                tickfont=dict(size=16, color='black')
            )
        )

        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        # Show the True vs Predicted plot
        fig.show()

        # Residuals Calculation and Plot
        residuals = y_pred - y
        residual_fig = go.Figure()

        # Add scatter plot for residuals
        residual_fig.add_trace(go.Scatter(
            x=y,
            y=residuals,
            mode='markers',
            marker=dict(color='deeppink', opacity=0.35),
            name=f'Residuals for {product}'
        ))

        # Add horizontal line at zero
        residual_fig.add_trace(go.Scatter(
            x=[y.min(), y.max()],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', dash='dash', width=4),
            name='Zero Residual Line'
        ))

        # Customize layout for residual plot
        residual_fig.update_layout(
            title=f'Residual Plot for {product} (After 10-Fold Cross-Validation)',
            xaxis_title='True (experimental) Faradaic Efficiency',
            yaxis_title='Residuals (Predicted - True)',
            legend=dict(x=0.02, y=0.98),
            width=800,
            height=600,
            margin=dict(l=40, r=40, b=40, t=40),
            plot_bgcolor='white',
            hovermode='closest',
            xaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror=True,
                title_font=dict(size=16, color='black'),
                tickfont=dict(size=14, color='black')
            ),
            yaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror=True,
                title_font=dict(size=18, color='black'),
                tickfont=dict(size=16, color='black')
            )
        )

        # Add grid
        residual_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        residual_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        # Show the Residual plot
        residual_fig.show()