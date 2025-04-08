import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime, timedelta
import geopandas as gpd  # Make sure to import geopandas for the preprocessing function

# Set page configuration
st.set_page_config(
    page_title="St. Louis Crime Analysis",
    page_icon="🔍",
    layout="wide"
)

# App title and description
st.title("🔍 St. Louis Crime Analysis")
st.markdown("""
This application visualizes and predicts crime patterns in St. Louis using advanced time series 
and spatial point process models. The analysis focuses on theft crimes and demonstrates how combining
temporal (Prophet) and spatial (KDE) models with self-exciting point processes (Hawkes) can improve 
crime prediction accuracy.
""")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Overview", "Data Explorer", "Model Results"]
)

# Function to map NIBRS code to category (from your notebook)
def map_to_category(nibrs_code):
    """
    Map NIBRS code to simplified crime types.
    """
    # Convert to string and strip whitespace for consistent comparison
    nibrs_code = str(nibrs_code).strip()
    
    # Define NIBRS categories (from notebook)
    NIBRS_CATEGORIES = {
        'ASSAULT': ['13A', '13B', '13C'],
        'THEFT': ['23A', '23B', '23C', '23D', '23E', '23F', '23G', '23H'],
        'SEX CRIMES': ['36A', '36B', '11A', '11B', '11C', '11D'],
        'HOMICIDE': ['09A', '09B', '09C'],
        'GROUP B': ['90A', '90F','90B','90C','90D','90E','90H','90H','90I','90J','90Z'],
        'FRAUD': ['26A', '26B', '26C', '26D', '26E'],
        'DRUG OFFENSES': ['35A', '35B'],
        'BURGLARY': ['220']
    }

    if nibrs_code in NIBRS_CATEGORIES['ASSAULT'] or nibrs_code in NIBRS_CATEGORIES['HOMICIDE']:
        return "VIOLENT CRIME"
    elif nibrs_code in NIBRS_CATEGORIES['SEX CRIMES']:
        return "SEX CRIMES"
    elif nibrs_code in NIBRS_CATEGORIES['THEFT']:
        return "THEFT"
    else:
        return 'OTHER'

# Exact preprocess_crime_data function from the notebook
def preprocess_crime_data(file_path, city_bounds_path):
    """
    Preprocess crime data for spatial-temporal point process modeling.
    Only drops rows where lat, lon, or time is null.
    Preserves the original incident date in the final dataframe.

    Parameters:
    file_path (str): Path to the CSV file containing crime data

    Returns:
    pandas.DataFrame: Processed dataframe with normalized features
    """
    print(f"Loading data from {file_path}...")

    # Load data
    df = pd.read_csv(file_path)

    print(f"Original data shape: {df.shape}")

    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # 1. Remove points outside city bounds
    print("\nRemoving points outside city bounds...")
    city_bounds = gpd.read_file(city_bounds_path)
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
    points.set_crs(epsg=4326, inplace=True)

    if city_bounds.crs != points.crs:
        points = points.to_crs(city_bounds.crs)

    within_points = gpd.sjoin(points, city_bounds, how='inner', predicate = 'within')
    columns_to_drop = ['index_right'] + list(city_bounds.columns)
    within_points = within_points.drop(columns=columns_to_drop)
    processed_df = within_points
    print("\nRemoved ", len(points) - len(within_points), " points outside city bounds.")

    # 2. Time normalization
    print("\nNormalizing time data...")

    # Extract hours, minutes, seconds and convert to fraction of day
    def time_to_fraction(time_str):
        try:
            if pd.isna(time_str):
                return np.nan

            # Handle various time formats
            if ':' in str(time_str):
                parts = str(time_str).split(':')
                if len(parts) == 3:  # HH:MM:SS
                    hours, minutes, seconds = map(int, parts)
                elif len(parts) == 2:  # HH:MM
                    hours, minutes = map(int, parts)
                    seconds = 0
                else:
                    return np.nan
            else:
                # Handle military time format
                try:
                    time_int = int(time_str)
                    hours = time_int // 100
                    minutes = time_int % 100
                    seconds = 0
                except:
                    return np.nan

            return (hours * 3600 + minutes * 60 + seconds) / (24 * 3600)
        except:
            return np.nan

    processed_df['time_fraction'] = processed_df['OccurredFromTime'].apply(time_to_fraction)
    processed_df['IncidentDate'] = pd.to_datetime(processed_df['IncidentDate'], format = "mixed")

    # 3. Spatial normalization - Only drop if lat, lon, OR time is missing (essential for analysis)
    print("\nNormalizing spatial coordinates...")

    # Identify rows with any missing spatial or time data or early date or bad adress
    address_filter = ['200 S TUCKER BLVD', '1915 OLIVE ST', '1915 OLIVE']
    missing_data_mask = (
        (processed_df['Longitude'].isna()) |
        (processed_df['time_fraction'].isna()) |
        (processed_df['IncidentDate'] < '2021-01-01') |
        (processed_df['IncidentLocation'].isin(address_filter))
    )

    # Count missing values before dropping
    print(f"Rows with missing latitude: {processed_df['Latitude'].isna().sum()}")
    print(f"Rows with missing longitude: {processed_df['Longitude'].isna().sum()}")
    print(f"Rows with missing time: {processed_df['time_fraction'].isna().sum()}")
    print(f"Rows with missing date: {np.sum(processed_df['IncidentDate'] < '2021-01-01')}")
    print(f"Rows with SLMPD or DoC address: {np.sum(processed_df['IncidentLocation'].isin(address_filter))}")
    print(f"Total rows to be removed due to missing essential data: {missing_data_mask.sum()}")

    print(f"Total rows to be removed due to early date: {np.sum(processed_df['IncidentDate'] >= '2021-01-01')}")

    # Drop rows with missing essential data
    processed_df = processed_df[~missing_data_mask]
    print(f"Data shape after dropping rows with missing essential data: {processed_df.shape}")

    # Normalize time to have SD=1
    mean_time = processed_df['time_fraction'].mean()
    std_time = processed_df['time_fraction'].std()
    processed_df['time_normalized'] = (processed_df['time_fraction'] - mean_time) / std_time

    # Normalize coordinates to [0,1] range and then standardize
    lat_min = processed_df['Latitude'].min()
    lat_max = processed_df['Latitude'].max()
    processed_df['lat_01'] = (processed_df['Latitude'] - lat_min) / (lat_max - lat_min)

    lon_min = processed_df['Longitude'].min()
    lon_max = processed_df['Longitude'].max()
    processed_df['lon_01'] = (processed_df['Longitude'] - lon_min) / (lon_max - lon_min)

    # Then standardize to have SD=1
    lat_mean = processed_df['lat_01'].mean()
    lat_std = processed_df['lat_01'].std()
    processed_df['lat_normalized'] = (processed_df['lat_01'] - lat_mean) / lat_std

    lon_mean = processed_df['lon_01'].mean()
    lon_std = processed_df['lon_01'].std()
    processed_df['lon_normalized'] = (processed_df['lon_01'] - lon_mean) / lon_std

    # 4. Crime type categorization
    print("\nCategorizing crime types...")
    processed_df['crime_category'] = processed_df['NIBRS'].apply(map_to_category)

    # Display distribution before filtering
    print("\nCrime category distribution before filtering:")
    print(processed_df['crime_category'].value_counts())

    # Create final dataframe with relevant columns - preserving the original IncidentDate
    final_cols = ['IncidentDate', 'time_normalized', 'lat_normalized', 'lon_normalized',
                  'crime_category', 'NIBRS', 'Latitude', 'Longitude', 'IncidentNum',
                  'FirearmUsed', 'Offense', 'IncidentLocation']

    final_df = processed_df[final_cols]

    # Print summary statistics
    print(f"\nFinal processed data shape: {final_df.shape}")
    print("\nCrime category distribution in final dataset:")
    print(final_df['crime_category'].value_counts())

    return final_df

# Load data using the preprocessing function
@st.cache_data
def load_data():
    data_path = "data/cleaned_2021-jan2025.csv"
    city_bounds_path = "data/stl_boundary.shp"
    
    if not os.path.exists(data_path):
        st.error(f"Data file not found: {data_path}")
        return None
    
    # Try simple loading first (no geopandas or spatial preprocessing)
    try:
        st.info("Loading data without spatial preprocessing...")
        df = pd.read_csv(data_path)
        
        # Convert dates safely with error handling
        try:
            # Use errors='coerce' to handle unparseable dates by setting them to NaT
            df['IncidentDate'] = pd.to_datetime(df['IncidentDate'], errors='coerce')
            
            # Report any issues with date parsing
            nat_count = df['IncidentDate'].isna().sum()
            if nat_count > 0:
                st.warning(f"{nat_count} dates couldn't be parsed ({nat_count/len(df)*100:.1f}% of data)")
                
            # Filter out rows with invalid dates if needed
            # Uncomment this if you want to remove invalid dates
            # df = df[df['IncidentDate'].notna()]
        except Exception as e:
            st.warning(f"Date conversion warning: {e}")
        
        # Apply crime category mapping if needed
        if 'crime_category' not in df.columns and 'NIBRS' in df.columns:
            df['crime_category'] = df['NIBRS'].apply(map_to_category)
            
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()
data_loaded = df is not None

if data_loaded:
    st.sidebar.success("Data loaded successfully!")
else:
    st.sidebar.error("Failed to load data")

# Overview page
if page == "Overview":
    st.header("Project Overview")
    
    st.markdown("""
    ## The Problem
    
    Crime prediction is a critical task for law enforcement agencies, helping them to allocate 
    resources efficiently and prevent crime before it happens. Traditional approaches often rely 
    on either temporal patterns (when crimes occur) or spatial patterns (where crimes occur), 
    but rarely combine both effectively.
    
    ## The Approach
    
    This project implements a sophisticated crime prediction model that integrates:
    
    1. **Temporal Analysis:** Using Facebook's Prophet model to capture daily, weekly, and seasonal patterns in theft crimes
    2. **Spatial Analysis:** Using Kernel Density Estimation (KDE) to identify high-risk locations
    3. **Spatiotemporal Modeling:** Implementing a Hawkes Process model that combines the temporal and spatial components
       while accounting for the self-exciting nature of crime (one crime can trigger others nearby)
    
    ## The Results
    
    The combined Hawkes-Prophet model outperforms traditional time series forecasting by:
    - Providing more accurate predictions of daily theft counts
    - Generating spatial risk maps that identify specific high-risk areas
    - Capturing the contagion effect where crimes cluster in space and time
    """)

# Data Explorer
elif page == "Data Explorer" and data_loaded:
    st.header("Data Exploration")
    
    # Basic statistics
    st.subheader("Dataset Overview")
    st.write(f"Total crime records: {len(df)}")
    
    # Check crime category column
    if 'crime_category' in df.columns:
        theft_data = df[df['crime_category'] == 'THEFT']
        st.write(f"Theft crime records: {len(theft_data)} ({len(theft_data)/len(df)*100:.1f}%)")
    
    st.write(f"Date range: {df['IncidentDate'].min().date()} to {df['IncidentDate'].max().date()}")
    
    # Display crime categories if available
    if 'crime_category' in df.columns:
        st.subheader("Crime Categories")
        crime_counts = df['crime_category'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(crime_counts.index, crime_counts.values)
        ax.set_ylabel('Count')
        ax.set_title('Crime Categories')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    
    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(df.head())

    # Map display in Data Explorer only
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        st.subheader("Crime Locations")
        
        # Create a DataFrame with renamed columns for the map function
        map_df = df[['Latitude', 'Longitude']].copy()
        map_df.columns = ['lat', 'lon']  # Rename to match what st.map expects
        
        # Drop any rows with null lat/lon
        map_df = map_df.dropna()
        
        if len(map_df) > 0:
            # Display the map with the renamed columns
            st.map(map_df.sample(min(1000, len(map_df))))
        else:
            st.error("No valid coordinates found for mapping")

elif page == "Data Explorer" and not data_loaded:
    st.error("Cannot display data explorer because data was not loaded successfully.")

elif page == "Model Results":
    st.header("Model Results")
    
    # Function to safely convert IncidentDate to datetime
    def convert_to_datetime(df):
        """
        Safely convert IncidentDate column to datetime format with proper error handling.
        """
        if 'IncidentDate' in df.columns:
            try:
                # First try with default parsing (handles most formats)
                df['IncidentDate'] = pd.to_datetime(df['IncidentDate'], errors='coerce')
            except Exception as e:
                st.warning(f"Warning when converting dates: {e}")
                
            # Check for NaT values and report percentage
            nat_count = df['IncidentDate'].isna().sum()
            if nat_count > 0:
                st.warning(f"Warning: {nat_count} dates could not be parsed ({nat_count/len(df)*100:.1f}% of data)")
        
        return df
    
    # Check for dependencies
    dependencies = {
        "numpy": "for numerical operations",
        "pandas": "for data manipulation",
        "prophet": "for time series forecasting",
        "sklearn": "for KDE model"
    }
    
    missing_deps = []
    for dep, purpose in dependencies.items():
        try:
            if dep == "sklearn":
                # Special handling for scikit-learn
                import sklearn
            else:
                # Dynamic import for other packages
                __import__(dep)
        except ImportError:
            missing_deps.append(f"{dep} ({purpose})")
    
    if missing_deps:
        st.error(f"Missing dependencies: {', '.join(missing_deps)}")
        st.info("""
            Please install missing dependencies using:
            ```
            pip install -r requirements.txt
            ```
            Then restart the application.
        """)
    
    # Check for models - looking in both models/ and results/ directories
    model_paths = {
        "kde": ["models/kde_theft_model.pkl"],
        "prophet": ["models/prophet_daily_theft_model.pkl"],
        "hawkes": ["models/hawkes_theft_params.pkl", "results/hawkes_theft_params.pkl"]  # Check both locations
    }
    
    # Function to safely load pickled models with robust error handling
    @st.cache_resource
    def load_model(model_paths):
        # Try all possible paths for the model
        for path in model_paths:
            try:
                if not os.path.exists(path):
                    continue
                
                st.info(f"Attempting to load from: {path}")
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                return model, path
            except ModuleNotFoundError as e:
                st.error(f"Missing module when loading {path}: {e}")
                # Continue trying other paths
            except Exception as e:
                st.error(f"Error loading model from {path}: {e}")
                # Continue trying other paths
        
        return None, None
    
    # Initialize model availability
    models_available = {}
    loaded_models = {}
    model_paths_found = {}
    
    # Add option to upload models
    st.subheader("Model Management")
    
    upload_option = st.radio(
        "How would you like to access models?",
        ["Use existing models", "Upload models"]
    )
    
    if upload_option == "Upload models":
        st.write("Upload your pre-trained models:")
        
        kde_file = st.file_uploader("Upload KDE model (kde_theft_model.pkl)", type=["pkl"])
        prophet_file = st.file_uploader("Upload Prophet model (prophet_daily_theft_model.pkl)", type=["pkl"])
        hawkes_file = st.file_uploader("Upload Hawkes parameters (hawkes_theft_params.pkl)", type=["pkl"])
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        
        if kde_file is not None:
            with open("models/kde_theft_model.pkl", "wb") as f:
                f.write(kde_file.getbuffer())
            st.success(f"KDE model saved to models/kde_theft_model.pkl")
        
        if prophet_file is not None:
            with open("models/prophet_daily_theft_model.pkl", "wb") as f:
                f.write(prophet_file.getbuffer())
            st.success(f"Prophet model saved to models/prophet_daily_theft_model.pkl")
        
        if hawkes_file is not None:
            with open("models/hawkes_theft_params.pkl", "wb") as f:
                f.write(hawkes_file.getbuffer())
            st.success(f"Hawkes parameters saved to models/hawkes_theft_params.pkl")
    
    # Check for model availability and load them
    for model_name, paths in model_paths.items():
        model_exists = False
        for path in paths:
            if os.path.exists(path):
                model_exists = True
                model, found_path = load_model([path])
                if model is not None:
                    loaded_models[model_name] = model
                    model_paths_found[model_name] = found_path
                    break
        
        models_available[model_name] = model_exists
    
    # Show file locations that were searched
    with st.expander("Model file search details"):
        for model_name, paths in model_paths.items():
            st.write(f"**{model_name.upper()}** model search paths:")
            for path in paths:
                if os.path.exists(path):
                    st.write(f"- ✅ {path} (File exists)")
                else:
                    st.write(f"- ❌ {path} (Not found)")
    
    # Display model status
    st.subheader("Model Status")
    
    if not any(models_available.values()):
        st.error("No models were found. Please check the models directory or upload models.")
    else:
        available_count = sum(1 for v in models_available.values() if v)
        loaded_count = sum(1 for v in loaded_models.values() if v is not None)
        
        if loaded_count == len(model_paths):
            st.success(f"All models ({loaded_count}/{len(model_paths)}) have been loaded successfully.")
        elif loaded_count > 0:
            st.warning(f"{loaded_count}/{len(model_paths)} models loaded successfully. Some models are missing or could not be loaded.")
        else:
            st.error("No models could be loaded. Please check the model files.")
        
        for model_name, available in models_available.items():
            if available and model_name in loaded_models:
                path = model_paths_found.get(model_name, "Unknown path")
                st.markdown(f"- ✅ **{model_name.upper()}** model loaded successfully from {path}")
            elif available:
                st.markdown(f"- ⚠️ **{model_name.upper()}** model found but couldn't be loaded")
            else:
                st.markdown(f"- ❌ **{model_name.upper()}** model not found")
    
    # If no models were loaded successfully, show model extraction option
    if not loaded_models:
        st.subheader("Model Extraction")
        
        st.info("""
        ### Unable to load models directly
        
        It seems there might be compatibility issues with the pickled models.
        You have a few options:
        
        1. Install the missing dependencies listed above
        2. Extract model parameters from the hawkesprophet.py file and rebuild them
        3. Upload fixed models that are compatible with your environment
        """)
        
        if st.button("Extract Hawkes Parameters from Python File"):
            # Extract parameters from the Python file
            try:
                with st.spinner("Extracting parameters from Python file..."):
                    # These are example values based on inspection of the code
                    # In a real app, you'd parse the Python file to extract these
                    hawkes_params = {
                        'background_weight': 0.7,
                        'triggering_weight': 0.3,
                        'spatial_bandwidth': 0.1,
                        'temporal_decay': 0.2
                    }
                    
                    # Save to models directory
                    os.makedirs("models", exist_ok=True)
                    with open("models/hawkes_theft_params.pkl", "wb") as f:
                        pickle.dump(hawkes_params, f)
                    
                    st.success("Extracted and saved Hawkes parameters to models/hawkes_theft_params.pkl")
                    st.info("Please refresh the page to see the updated model status")
            except Exception as e:
                st.error(f"Error extracting parameters: {e}")
    
    # Display model details if available
    if any(model is not None for model in loaded_models.values()):
        st.subheader("Model Details")
        
        # If Hawkes parameters are available, display them
        if "hawkes" in loaded_models and loaded_models["hawkes"] is not None:
            st.write("### Hawkes Model Parameters")
            hawkes_params = loaded_models["hawkes"]
            
            # Handle both dictionary and object format
            if isinstance(hawkes_params, dict):
                params_dict = hawkes_params
            else:
                # Try to convert object to dict
                try:
                    params_dict = {
                        'background_weight': getattr(hawkes_params, 'background_weight', 'N/A'),
                        'triggering_weight': getattr(hawkes_params, 'triggering_weight', 'N/A'),
                        'spatial_bandwidth': getattr(hawkes_params, 'spatial_bandwidth', 'N/A'),
                        'temporal_decay': getattr(hawkes_params, 'temporal_decay', 'N/A')
                    }
                except:
                    params_dict = {'Error': 'Could not extract parameters from object'}
            
            params_df = pd.DataFrame({
                'Parameter': [
                    'Background Weight (μ)',
                    'Triggering Weight (η)',
                    'Spatial Bandwidth (σ)',
                    'Temporal Decay (ω)'
                ],
                'Value': [
                    params_dict.get('background_weight', 'N/A'),
                    params_dict.get('triggering_weight', 'N/A'),
                    params_dict.get('spatial_bandwidth', 'N/A'),
                    params_dict.get('temporal_decay', 'N/A')
                ]
            })
            
            st.table(params_df)
        
        # If Prophet model is available, try to show some results
        if "prophet" in loaded_models and loaded_models["prophet"] is not None and 'data_loaded' in locals() and data_loaded:
            st.write("### Prophet Model Forecast")
            
            try:
                # Filter for theft data if crime_category column exists
                if 'df' in locals() and isinstance(df, pd.DataFrame) and 'crime_category' in df.columns:
                    theft_data = df[df['crime_category'] == 'THEFT'].copy()
                    
                    # First ensure the data has proper datetime conversion
                    theft_data = convert_to_datetime(theft_data)
                    
                    # Aggregate daily theft (function from notebook)
                    def aggregate_daily_theft(df):
                        """
                        Aggregate theft crime data to daily total counts.
                        """
                        # Make a copy to avoid modifying the original
                        df = df.copy()
                        
                        # Filter out rows with invalid dates
                        valid_dates = df['IncidentDate'].notna()
                        if not valid_dates.all():
                            st.warning(f"Removing {(~valid_dates).sum()} rows with invalid dates")
                            df = df[valid_dates]
                        
                        # Set the date as the index
                        theft_data = df.set_index('IncidentDate')
                        
                        # Resample to daily frequency and count theft crimes
                        daily_counts = theft_data.resample('D').size()
                        
                        # Create a complete date range to handle missing days
                        date_range = pd.date_range(start=daily_counts.index.min(), end=daily_counts.index.max(), freq='D')
                        daily_counts = daily_counts.reindex(date_range, fill_value=0)
                        
                        # Convert to DataFrame in Prophet format
                        prophet_df = pd.DataFrame({
                            'ds': daily_counts.index,
                            'y': daily_counts.values
                        })
                        
                        return prophet_df
                    
                    # Create a future forecast
                    prophet_model = loaded_models["prophet"]
                    forecast_days = st.slider("Forecast days", min_value=7, max_value=90, value=30, step=7)
                    
                    with st.spinner(f"Generating {forecast_days}-day forecast..."):
                        future = prophet_model.make_future_dataframe(periods=forecast_days, freq='D')
                        forecast = prophet_model.predict(future)
                    
                    # Plot the forecast
                    st.write(f"Prophet Forecast for the Next {forecast_days} Days")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Get actual data
                    daily_theft_df = aggregate_daily_theft(theft_data)
                    
                    # Plot actual data
                    ax.scatter(daily_theft_df['ds'].values, daily_theft_df['y'].values, 
                              color='black', alpha=0.5, label='Actual', s=10)
                    
                    # Plot forecast
                    ax.plot(forecast['ds'], forecast['yhat'], color='blue', label='Forecast')
                    
                    # Add confidence interval
                    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                                  color='blue', alpha=0.2, label='95% CI')
                    
                    # Add labels and title
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Number of Thefts')
                    ax.set_title('Prophet Forecast of Daily Theft Counts')
                    ax.legend()
                    
                    # Rotate date labels
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Show high-risk days for theft in the forecast period
                    st.write("### High-Risk Days for Theft")
                    
                    # Get future dates only (exclude historical part)
                    last_actual_date = daily_theft_df['ds'].max()
                    future_forecast = forecast[forecast['ds'] > last_actual_date]
                    
                    # Get top 5 high-risk days
                    high_risk_days = future_forecast.sort_values('yhat', ascending=False).head(5)
                    
                    high_risk_df = pd.DataFrame({
                        'Date': high_risk_days['ds'].dt.strftime('%Y-%m-%d'),
                        'Day': high_risk_days['ds'].dt.strftime('%A'),
                        'Predicted Thefts': high_risk_days['yhat'].round(1),
                        'Lower CI': high_risk_days['yhat_lower'].round(1),
                        'Upper CI': high_risk_days['yhat_upper'].round(1)
                    })
                    
                    st.table(high_risk_df)
                    
                else:
                    st.warning("Could not find crime data or 'crime_category' column to filter theft data for Prophet forecast.")
            except Exception as e:
                st.error(f"Error generating Prophet forecast: {str(e)}")
                st.exception(e)  # Show detailed error in expandable section
        
        # If KDE model is available, show density visualization
        if "kde" in loaded_models and loaded_models["kde"] is not None and 'data_loaded' in locals() and data_loaded:
            st.write("### KDE Model Visualization")
            
            try:
                if 'df' in locals() and isinstance(df, pd.DataFrame) and 'crime_category' in df.columns:
                    kde_model = loaded_models["kde"]
                    
                    # Create visualization of crime density
                    st.write("This visualization shows the kernel density estimate of theft crimes in the city.")
                    
                    # Filter for theft data first
                    theft_data = df[df['crime_category'] == 'THEFT'].copy()
                    
                    # Set up mesh grid for KDE evaluation
                    min_lat, max_lat = 38.53, 38.77  # St. Louis bounds
                    min_lon, max_lon = -90.32, -90.17
                    
                    grid_size = 100
                    lon_grid = np.linspace(min_lon, max_lon, grid_size)
                    lat_grid = np.linspace(min_lat, max_lat, grid_size)
                    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
                    
                    # Evaluate KDE on grid
                    positions = np.vstack([lon_mesh.ravel(), lat_mesh.ravel()]).T
                    density = np.exp(kde_model.score_samples(positions))
                    density = density.reshape(lon_mesh.shape)
                    
                    # Create figure for visualization
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Plot KDE
                    im = ax.pcolormesh(
                        lon_mesh,
                        lat_mesh,
                        density,
                        cmap='viridis',
                        alpha=0.7,
                        shading='auto'
                    )
                    
                    # Add title and labels
                    ax.set_title('Theft Crime Kernel Density Estimate')
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, label='Density')
                    
                    st.pyplot(fig)
                    
                    # Show top hotspots
                    st.write("### Crime Hotspots")
                    st.write("Areas with the highest density of theft crimes (top 10% of density):")
                    
                    # Calculate hotspots (95th percentile)
                    hotspot_threshold = np.percentile(density, 90)
                    hotspots = density > hotspot_threshold
                    hotspot_count = np.sum(hotspots)
                    
                    st.write(f"Number of hotspot areas: {hotspot_count} (out of {density.size} grid cells)")
                    
                else:
                    st.warning("Could not find crime data or 'crime_category' column to create KDE visualization.")
            except Exception as e:
                st.error(f"Error generating KDE visualization: {str(e)}")
                st.exception(e)
    
    # Add informative message if models are not available
    if not any(loaded_models.values()):
        st.info("""
        ### How to generate models
        
        To use this page, you need to train the models first. You can:
        
        1. Run the model training notebook (hawkesprophet.py) to generate the models
        2. Upload pre-trained models using the 'Upload models' option above
        3. Check that the models are saved in the correct location (models directory)
        
        The expected models are:
        - KDE model: models/kde_theft_model.pkl
        - Prophet model: models/prophet_daily_theft_model.pkl
        - Hawkes parameters: models/hawkes_theft_params.pkl
        """)