# Housing Price Forecasting with Tree-Based Models

This project presents an end-to-end machine learning pipeline to forecast housing prices using the Zillow Home Value Index (ZHVI). The entire analysis is built in a Jupyter notebook, leveraging tree-based models (LightGBM, XGBoost) and SARIMAX for temporal forecasting. The notebook includes model interpretation, validation, and forecast generation with actionable insights for economic planning and housing market evaluation.

What’s Inside:

A comprehensive walkthrough using LightGBM, XGBoost, and SARIMAX
Exploratory Data Analysis highlighting macroeconomic trends (e.g., 2008 crash, COVID-19 boom)
Feature engineering: lag variables, calendar features (year, quarter, month)
Cross-validation (K-Fold) with metrics like R², MAE, RMSE, MAPE, SMAPE
Residual diagnostics, Q-Q plots, and studentized error analysis
Price-tier segmentation (Low, Mid, High) to combat heteroskedasticity
6-month ahead stepwise forecasts for each U.S. region

Model Highlights:
Trains models to predict regional home prices using historical time-series data
Applies LightGBM and XGBoost with strong out-of-sample performance (R² > 0.93)
Implements tiered modeling strategy to improve forecast reliability across market segments
Includes SARIMAX for comparison, showcasing limits of linear autoregressive models
Automates recursive forecasting for March–August 2025
Saves final predictions to a structured CSV file by region and price tier

Why You Should Check It Out:
This notebook provides a structured, well-explained, and highly interpretable example. You’ll see practical decisions around:
Model selection and validation
Lag-based autoregressive design
Addressing overfitting and residual imbalance
Forecasting in high-volatility regions

Files Included:
housing_price_forecasting.ipynb — Full notebook: data cleaning, EDA, modeling, forecasting
requirements.txt — All dependencies
ZHVI.csv — Cleaned Zillow housing index dataset
6_month_forecast.csv — Final forecast output

Get Started:
Clone the repository:
https://github.com/Stefanjafry/housing-price-forecasting

Install requirements:
pip install -r requirements.txt

Launch notebook:
jupyter notebook housing_price_forecasting.ipynb

License:
This project is licensed under the MIT License. You are free to use, modify, and distribute this code.
