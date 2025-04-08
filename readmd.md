# St. Louis Crime Analysis & Prediction

![Crime Analysis Dashboard](screenshot.png)

## Project Overview

This project implements advanced spatiotemporal modeling to predict theft crimes in St. Louis. It combines:

- **Prophet** for temporal forecasting
- **Kernel Density Estimation (KDE)** for spatial hotspot analysis
- **Hawkes Process** modeling to capture the self-exciting nature of crime

The web application provides interactive visualizations of crime patterns, model performance, and forecasting.

## Features

- **Data Exploration**: Visualize crime trends and patterns
- **Temporal Analysis**: Prophet-based time series forecasting
- **Spatial Analysis**: Crime hotspot mapping using KDE
- **Hawkes Process Modeling**: Combined spatiotemporal prediction
- **Forecasting**: Daily theft predictions with confidence intervals

## Technologies Used

- Python (Pandas, NumPy, SciPy)
- Streamlit for web application
- Prophet for time series forecasting
- Scikit-learn for spatial modeling
- SciPy for optimization
- Plotly and Matplotlib for visualization
- GeoPandas and Contextily for geospatial analysis

## Installation & Running Locally

1. Clone this repository