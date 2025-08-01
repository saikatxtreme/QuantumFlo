import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm
import math # For math.ceil
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore') # Suppress warnings from statsmodels, etc.

# Try to import Prophet and ARIMA, but handle potential errors gracefully
try:
    from prophet import Prophet
    from statsmodels.tsa.arima.model import ARIMA
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophet and ARIMA are not available. This is likely due to complex dependencies like pystan failing to install. The app will proceed using XGBoost and RandomForestRegressor.")


# --- Configuration for Dummy Data Generation ---
DEFAULT_NUM_SKUS = 15
DEFAULT_NUM_COMPONENTS_PER_SKU = 5
DEFAULT_START_DATE = datetime(2022, 1, 1)
DEFAULT_END_DATE = datetime(2025, 12, 31)
DEFAULT_PROMOTION_FREQUENCY_DAYS = 60
DEFAULT_MAX_LEAD_TIME_DAYS = 30
DEFAULT_MAX_SKU_SHELF_LIFE_DAYS = 365
DEFAULT_SALES_CHANNELS = ["Distributor Network", "Amazon", "Own Website"]

# --- Default Cost Parameters (used if global_config.csv is not uploaded) ---
DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY = 0.10
DEFAULT_ORDERING_COST_PER_ORDER = 50.00
DEFAULT_SHORTAGE_COST_PER_UNIT = 10.00

# --- State Management ---
if 'sales_df' not in st.session_state:
    st.session_state.sales_df = pd.DataFrame()
if 'sku_config_df' not in st.session_state:
    st.session_state.sku_config_df = pd.DataFrame()
if 'bom_df' not in st.session_state:
    st.session_state.bom_df = pd.DataFrame()
if 'inventory_df' not in st.session_state:
    st.session_state.inventory_df = pd.DataFrame()
if 'promotions_df' not in st.session_state:
    st.session_state.promotions_df = pd.DataFrame()
if 'global_config_df' not in st.session_state:
    st.session_state.global_config_df = pd.DataFrame()
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = pd.DataFrame()
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = pd.DataFrame()
if 'all_inventory_df' not in st.session_state:
    st.session_state.all_inventory_df = pd.DataFrame()
if 'all_stockout_rates_df' not in st.session_state:
    st.session_state.all_stockout_rates_df = pd.DataFrame()
if 'all_costs_df' not in st.session_state:
    st.session_state.all_costs_df = pd.DataFrame()


# --- Helper Functions ---
def generate_dummy_data():
    """Generates a complete set of dummy data for the application."""
    st.session_state.sales_df = generate_sales_data()
    st.session_state.sku_config_df = generate_sku_config()
    st.session_state.bom_df = generate_bom()
    st.session_state.inventory_df = generate_inventory_data()
    st.session_state.promotions_df = generate_promotions_data()
    st.session_state.global_config_df = generate_global_config()

def generate_sales_data():
    """Generates dummy sales data."""
    date_range = pd.date_range(DEFAULT_START_DATE, DEFAULT_END_DATE)
    sales_data = []
    skus = [f'SKU_{i+1}' for i in range(DEFAULT_NUM_SKUS)]
    for date in date_range:
        for sku in skus:
            # Introduce seasonality and trend
            base_demand = 50 + 10 * math.sin(date.month * 2 * math.pi / 12) + (date.year - DEFAULT_START_DATE.year) * 5
            promotion_effect = 0
            if random.random() < 0.1: # 10% chance of a promotion day
                promotion_effect = random.randint(50, 150)
            demand = max(0, int(base_demand + random.normalvariate(0, 10) + promotion_effect))
            sales_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'SKU_ID': sku,
                'Sales_Quantity': demand,
                'Sales_Channel': random.choice(DEFAULT_SALES_CHANNELS)
            })
    return pd.DataFrame(sales_data)

def generate_sku_config():
    """Generates dummy SKU configuration data."""
    sku_data = []
    skus = [f'SKU_{i+1}' for i in range(DEFAULT_NUM_SKUS)]
    for sku in skus:
        sku_data.append({
            'SKU_ID': sku,
            'Product_Name': f'Product {sku}',
            'Lead_Time_Days': random.randint(5, DEFAULT_MAX_LEAD_TIME_DAYS),
            'Shelf_Life_Days': random.randint(90, DEFAULT_MAX_SKU_SHELF_LIFE_DAYS)
        })
    return pd.DataFrame(sku_data)

def generate_bom():
    """Generates dummy Bill of Materials data."""
    bom_data = []
    skus = [f'SKU_{i+1}' for i in range(DEFAULT_NUM_SKUS)]
    for sku in skus:
        for component_id in range(random.randint(1, DEFAULT_NUM_COMPONENTS_PER_SKU)):
            bom_data.append({
                'SKU_ID': sku,
                'Component_ID': f'C{component_id+1}-{sku}',
                'Quantity_Required': random.randint(1, 4)
            })
    return pd.DataFrame(bom_data)

def generate_inventory_data():
    """Generates dummy inventory data."""
    date_range = pd.date_range(DEFAULT_START_DATE, DEFAULT_END_DATE)
    inventory_data = []
    skus = [f'SKU_{i+1}' for i in range(DEFAULT_NUM_SKUS)]
    for date in date_range:
        for sku in skus:
            inventory_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'SKU_ID': sku,
                'Quantity_in_Stock': random.randint(100, 500)
            })
    return pd.DataFrame(inventory_data)

def generate_promotions_data():
    """Generates dummy promotions data."""
    promo_data = []
    skus = [f'SKU_{i+1}' for i in range(DEFAULT_NUM_SKUS)]
    promo_dates = pd.date_range(DEFAULT_START_DATE, DEFAULT_END_DATE, freq=f'{DEFAULT_PROMOTION_FREQUENCY_DAYS}D')
    for date in promo_dates:
        for sku in random.sample(skus, 2):
            promo_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'SKU_ID': sku,
                'Promotion_Type': 'Seasonal Sale',
                'Discount_Percentage': random.uniform(5, 25)
            })
    return pd.DataFrame(promo_data)

def generate_global_config():
    """Generates dummy global configuration data."""
    return pd.DataFrame([
        {'Parameter': 'Holding_Cost_Per_Unit_Per_Day', 'Value': DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY},
        {'Parameter': 'Ordering_Cost_Per_Order', 'Value': DEFAULT_ORDERING_COST_PER_ORDER},
        {'Parameter': 'Shortage_Cost_Per_Unit', 'Value': DEFAULT_SHORTAGE_COST_PER_UNIT}
    ])


def upload_data():
    """Handles file uploads and updates the session state."""
    st.subheader("Upload Your Data")
    st.info("Upload your CSV files to run the analysis. You can also run the analysis on Sample Data.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("---")
        st.write("**Mandatory Files:**")
        sales_file = st.file_uploader("Upload sales.csv", type=['csv'], key="sales_uploader")
        sku_config_file = st.file_uploader("Upload sku_config.csv", type=['csv'], key="sku_config_uploader")
        
        st.write("---")
        st.write("**Optional Files:**")
        bom_file = st.file_uploader("Upload bom.csv", type=['csv'], key="bom_uploader")
        inventory_file = st.file_uploader("Upload inventory.csv", type=['csv'], key="inventory_uploader")
        promotions_file = st.file_uploader("Upload promotions.csv", type=['csv'], key="promotions_uploader")
        global_config_file = st.file_uploader("Upload global_config.csv", type=['csv'], key="global_config_uploader")

    with col2:
        if sales_file:
            st.session_state.sales_df = pd.read_csv(sales_file, parse_dates=['Date'])
            st.success("Sales data uploaded successfully!")
        if sku_config_file:
            st.session_state.sku_config_df = pd.read_csv(sku_config_file)
            st.success("SKU config data uploaded successfully!")
        if bom_file:
            st.session_state.bom_df = pd.read_csv(bom_file)
            st.success("BOM data uploaded successfully!")
        if inventory_file:
            st.session_state.inventory_df = pd.read_csv(inventory_file, parse_dates=['Date'])
            st.success("Inventory data uploaded successfully!")
        if promotions_file:
            st.session_state.promotions_df = pd.read_csv(promotions_file, parse_dates=['Date'])
            st.success("Promotions data uploaded successfully!")
        if global_config_file:
            st.session_state.global_config_df = pd.read_csv(global_config_file)
            st.success("Global config data uploaded successfully!")


def download_sample_data():
    """Generates and provides download links for sample data."""
    st.subheader("Download Sample Data")
    
    st.info("You can use the following sample data templates to get started.")

    st.write("**Mandatory Templates:**")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download sales.csv",
            data=st.session_state.sales_df.to_csv(index=False).encode('utf-8'),
            file_name="sales.csv",
            mime="text/csv"
        )
    with col2:
        st.download_button(
            label="Download sku_config.csv",
            data=st.session_state.sku_config_df.to_csv(index=False).encode('utf-8'),
            file_name="sku_config.csv",
            mime="text/csv"
        )

    st.write("**Optional Templates:**")
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        st.download_button(
            label="Download bom.csv",
            data=st.session_state.bom_df.to_csv(index=False).encode('utf-8'),
            file_name="bom.csv",
            mime="text/csv"
        )
    with col4:
        st.download_button(
            label="Download inventory.csv",
            data=st.session_state.inventory_df.to_csv(index=False).encode('utf-8'),
            file_name="inventory.csv",
            mime="text/csv"
        )
    with col5:
        st.download_button(
            label="Download promotions.csv",
            data=st.session_state.promotions_df.to_csv(index=False).encode('utf-8'),
            file_name="promotions.csv",
            mime="text/csv"
        )
    with col6:
        st.download_button(
            label="Download global_config.csv",
            data=st.session_state.global_config_df.to_csv(index=False).encode('utf-8'),
            file_name="global_config.csv",
            mime="text/csv"
        )


def preprocess_data(sales_df, promotions_df):
    """Performs data preprocessing and feature engineering."""
    df = sales_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Resample to daily frequency and fill missing dates
    df = df.groupby('SKU_ID')['Sales_Quantity'].resample('D').sum().fillna(0).reset_index()
    
    # Feature Engineering
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfMonth'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Quarter'] = df['Date'].dt.quarter
    
    # Lag features
    df['Lag_7'] = df.groupby('SKU_ID')['Sales_Quantity'].shift(7)
    df['Lag_30'] = df.groupby('SKU_ID')['Sales_Quantity'].shift(30)
    
    # Rolling average features
    df['Rolling_Mean_7'] = df.groupby('SKU_ID')['Sales_Quantity'].transform(lambda x: x.rolling(window=7).mean())
    
    # Add promotions data
    if not promotions_df.empty:
        promotions_df['Date'] = pd.to_datetime(promotions_df['Date'])
        promotions_df['Is_Promo'] = 1
        df = df.merge(promotions_df[['Date', 'SKU_ID', 'Is_Promo']], on=['Date', 'SKU_ID'], how='left').fillna({'Is_Promo': 0})
    else:
        df['Is_Promo'] = 0

    df = df.fillna(0)
    return df


def train_and_forecast(df, model_choice, forecast_horizon_days):
    """Trains a model and generates forecasts."""
    all_forecasts = []
    model_performance_list = []
    
    unique_skus = df['SKU_ID'].unique()
    
    for sku in unique_skus:
        sku_df = df[df['SKU_ID'] == sku].copy()
        
        # Split data for training and validation
        train_df = sku_df[sku_df['Date'] < sku_df['Date'].max() - timedelta(days=forecast_horizon_days)].copy()
        test_df = sku_df[sku_df['Date'] >= sku_df['Date'].max() - timedelta(days=forecast_horizon_days)].copy()

        if model_choice in ['XGBoost', 'RandomForest']:
            features = ['DayOfWeek', 'DayOfMonth', 'Month', 'Year', 'Quarter', 'Lag_7', 'Lag_30', 'Rolling_Mean_7', 'Is_Promo']
            X_train = train_df[features]
            y_train = train_df['Sales_Quantity']
            X_test = test_df[features]
            y_test = test_df['Sales_Quantity']

            if model_choice == 'XGBoost':
                model = XGBRegressor(objective='reg:squarederror')
            else:
                model = RandomForestRegressor(n_estimators=100)

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)

            # Generate future dates for forecasting
            last_date = sku_df['Date'].max()
            future_dates = pd.date_range(last_date + timedelta(days=1), last_date + timedelta(days=forecast_horizon_days))
            
            future_df = pd.DataFrame({'Date': future_dates})
            future_df['SKU_ID'] = sku
            future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
            future_df['DayOfMonth'] = future_df['Date'].dt.day
            future_df['Month'] = future_df['Date'].dt.month
            future_df['Year'] = future_df['Date'].dt.year
            future_df['Quarter'] = future_df['Date'].dt.quarter
            future_df['Is_Promo'] = 0 # Assume no promotions in the forecast period
            
            # Simple lag/rolling mean for future, assuming no future sales data
            future_df['Lag_7'] = 0
            future_df['Lag_30'] = 0
            future_df['Rolling_Mean_7'] = 0
            
            forecasts = model.predict(future_df[features])
            
        elif model_choice == 'Prophet' and PROPHET_AVAILABLE:
            # Prophet requires specific column names
            prophet_df = sku_df[['Date', 'Sales_Quantity']].rename(columns={'Date': 'ds', 'Sales_Quantity': 'y'})
            m = Prophet()
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=forecast_horizon_days, freq='D')
            forecast_result = m.predict(future)
            
            # Extract and evaluate predictions from the test set
            predictions = forecast_result.loc[forecast_result['ds'].isin(test_df['Date']), 'yhat'].values
            mae = mean_absolute_error(test_df['Sales_Quantity'], predictions)
            mse = mean_squared_error(test_df['Sales_Quantity'], predictions)
            
            forecasts = forecast_result['yhat'].values[-forecast_horizon_days:]

        elif model_choice == 'ARIMA' and PROPHET_AVAILABLE:
            # ARIMA model
            arima_model = ARIMA(train_df['Sales_Quantity'], order=(1,1,1))
            arima_result = arima_model.fit()
            
            predictions = arima_result.predict(start=len(train_df), end=len(train_df)+len(test_df)-1)
            mae = mean_absolute_error(test_df['Sales_Quantity'], predictions)
            mse = mean_squared_error(test_df['Sales_Quantity'], predictions)
            
            forecasts = arima_result.forecast(steps=forecast_horizon_days)

        else:
            st.error(f"Model choice '{model_choice}' is not supported or its dependencies are not met.")
            continue

        model_performance_list.append({'SKU_ID': sku, 'Model': model_choice, 'MAE': mae, 'MSE': mse})
        
        forecast_df = pd.DataFrame({
            'Date': pd.date_range(df['Date'].max() + timedelta(days=1), periods=forecast_horizon_days),
            'SKU_ID': sku,
            'Forecasted_Sales': np.maximum(0, forecasts).round(0)
        })
        all_forecasts.append(forecast_df)
        
    if not all_forecasts:
        st.error("No forecasts were generated. Please check your data and parameters.")
        return pd.DataFrame(), pd.DataFrame()

    all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    model_performance_df = pd.DataFrame(model_performance_list)

    return all_forecasts_df, model_performance_df


def calculate_safety_stock(sku_id, forecast_df, sku_config_df, method, service_level, reorder_point_buffer):
    """Calculates safety stock based on the chosen method."""
    sku_lead_time = sku_config_df[sku_config_df['SKU_ID'] == sku_id]['Lead_Time_Days'].iloc[0]

    if method == "Manual Input":
        # Let's use the reorder_point_buffer as a manual safety stock for simplicity
        return reorder_point_buffer
    
    elif method == "Heuristic":
        # Heuristic: A simple percentage of average lead time demand
        avg_daily_demand = forecast_df['Forecasted_Sales'].mean()
        safety_stock = avg_daily_demand * sku_lead_time * 0.5  # 50% of average lead time demand
        return safety_stock
        
    elif method == "Probabilistic":
        # Probabilistic: Z-score * Std Dev of Demand during Lead Time
        avg_daily_demand = forecast_df['Forecasted_Sales'].mean()
        std_daily_demand = forecast_df['Forecasted_Sales'].std()
        
        # Z-score for the desired service level (e.g., 95% is 1.645)
        z_score = norm.ppf(service_level)
        
        # Standard deviation of demand during lead time
        std_lead_time_demand = std_daily_demand * math.sqrt(sku_lead_time)
        
        safety_stock = z_score * std_lead_time_demand
        return safety_stock
        
    return 0


def calculate_reorder_point(sku_id, forecast_df, sku_config_df, safety_stock):
    """Calculates the reorder point."""
    sku_lead_time = sku_config_df[sku_config_df['SKU_ID'] == sku_id]['Lead_Time_Days'].iloc[0]
    avg_daily_demand = forecast_df['Forecasted_Sales'].mean()
    
    # Reorder Point = (Avg Daily Demand * Lead Time) + Safety Stock
    reorder_point = (avg_daily_demand * sku_lead_time) + safety_stock
    return reorder_point


def calculate_inventory_levels(
    historical_df, forecast_df, sku_config_df,
    holding_cost, ordering_cost, shortage_cost,
    safety_stock_method, service_level, reorder_point_buffer
):
    """
    Simulates inventory levels and calculates costs based on forecasts and
    inventory parameters.
    """
    all_inventory = []
    all_stockout_rates = []
    all_costs = []
    
    unique_skus = forecast_df['SKU_ID'].unique()
    
    for sku in unique_skus:
        sku_forecast_df = forecast_df[forecast_df['SKU_ID'] == sku].copy()
        sku_historical_df = historical_df[historical_df['SKU_ID'] == sku].copy()
        
        # Combine historical and forecasted data for a full view
        combined_df = pd.concat([sku_historical_df, sku_forecast_df.rename(columns={'Forecasted_Sales': 'Sales_Quantity'})], ignore_index=True)
        combined_df = combined_df.sort_values('Date').reset_index(drop=True)
        combined_df['Sales_Quantity'] = combined_df['Sales_Quantity'].fillna(0) # In case of gaps

        # Get initial inventory from the last historical record
        last_historical_date = sku_historical_df['Date'].max()
        if not st.session_state.inventory_df.empty:
            initial_inventory = st.session_state.inventory_df[
                (st.session_state.inventory_df['SKU_ID'] == sku) &
                (st.session_state.inventory_df['Date'] == last_historical_date)
            ]['Quantity_in_Stock'].values
            if len(initial_inventory) > 0:
                current_inventory = initial_inventory[0]
            else:
                current_inventory = 0
        else:
            current_inventory = 0
        
        # Calculate safety stock and reorder point
        safety_stock = calculate_safety_stock(sku, sku_forecast_df, sku_config_df, safety_stock_method, service_level, reorder_point_buffer)
        reorder_point = calculate_reorder_point(sku, sku_forecast_df, sku_config_df, safety_stock)
        
        inventory_records = []
        holding_cost_total = 0
        ordering_cost_total = 0
        shortage_cost_total = 0
        stockout_days = 0
        
        for index, row in combined_df.iterrows():
            date = row['Date']
            sales = row['Sales_Quantity']
            
            # Update inventory level at the beginning of the day
            inventory_on_hand = current_inventory
            
            # Check for stockout
            if inventory_on_hand < sales:
                stockout_days += 1
                shortage_cost_total += (sales - inventory_on_hand) * shortage_cost
                inventory_on_hand = 0
            else:
                inventory_on_hand -= sales
            
            holding_cost_total += inventory_on_hand * holding_cost
            
            # Place a new order if inventory falls below reorder point
            order_quantity = 0
            if inventory_on_hand <= reorder_point:
                # Simple order up to logic
                order_quantity = max(0, (reorder_point + safety_stock) * 2 - inventory_on_hand)
                ordering_cost_total += ordering_cost
                
            # Assume orders arrive instantly for this simple simulation
            current_inventory = inventory_on_hand + order_quantity
            
            inventory_records.append({
                'Date': date,
                'SKU_ID': sku,
                'Inventory_Level': inventory_on_hand,
                'Sales_Quantity': sales,
                'Order_Quantity': order_quantity,
                'Safety_Stock': safety_stock,
                'Reorder_Point': reorder_point
            })

        inventory_df = pd.DataFrame(inventory_records)
        all_inventory.append(inventory_df)
        
        # Calculate stockout rate
        total_days = len(inventory_df)
        stockout_rate = (stockout_days / total_days) * 100 if total_days > 0 else 0
        all_stockout_rates.append({'SKU_ID': sku, 'Stockout_Rate': stockout_rate})
        
        # Calculate total cost
        total_cost = holding_cost_total + ordering_cost_total + shortage_cost_total
        all_costs.append({
            'SKU_ID': sku,
            'Total_Cost': total_cost,
            'Holding_Cost': holding_cost_total,
            'Ordering_Cost': ordering_cost_total,
            'Shortage_Cost': shortage_cost_total
        })

    return pd.concat(all_inventory), pd.DataFrame(all_stockout_rates), pd.DataFrame(all_costs)


def aggregate_kpi_for_plot(df, sku, value_col, date_format, aggregation_method='sum'):
    """Aggregates a KPI for plotting based on the selected date format."""
    df_sku = df[df['SKU_ID'] == sku].copy()
    df_sku['Date'] = pd.to_datetime(df_sku['Date'])
    
    if date_format == "Daily":
        return df_sku.set_index('Date').resample('D')[value_col].agg(aggregation_method).reset_index().rename(columns={value_col: 'Value'})
    elif date_format == "Weekly":
        return df_sku.set_index('Date').resample('W')[value_col].agg(aggregation_method).reset_index().rename(columns={value_col: 'Value'})
    elif date_format == "Monthly":
        return df_sku.set_index('Date').resample('M')[value_col].agg(aggregation_method).reset_index().rename(columns={value_col: 'Value'})
    
    return pd.DataFrame()


# --- Streamlit UI ---
st.set_page_config(
    page_title="Demand and Inventory Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Demand and Inventory Intelligence")

# Page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data & Parameters", "Analysis & Results"])


# --- Page 1: Data & Parameters ---
if page == "Data & Parameters":
    st.header("Data and Input Parameters")
    st.markdown("""
    This page allows you to upload your data and configure the parameters for the demand forecasting and inventory optimization.
    
    **About the Data Tables:**
    - `sales.csv` (Mandatory): Your historical sales data. It must contain `Date`, `SKU_ID`, and `Sales_Quantity`.
    - `sku_config.csv` (Mandatory): Configuration for each SKU. Must contain `SKU_ID`, `Lead_Time_Days`, and `Shelf_Life_Days`.
    - `bom.csv` (Optional): Bill of Materials, linking SKUs to their components. Columns: `SKU_ID`, `Component_ID`, `Quantity_Required`.
    - `inventory.csv` (Optional): Historical inventory levels. Columns: `Date`, `SKU_ID`, `Quantity_in_Stock`.
    - `promotions.csv` (Optional): Data on past promotions. Columns: `Date`, `SKU_ID`, `Promotion_Type`, `Discount_Percentage`.
    - `global_config.csv` (Optional): Cost parameters. Columns: `Parameter`, `Value`. If not provided, default values will be used.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Management")
        upload_data()
        if st.button("Run on Sample Data"):
            generate_dummy_data()
            st.success("Sample data generated and loaded. You can now configure parameters and run the analysis on the next page.")

    with col2:
        st.subheader("Sample Data Templates")
        download_sample_data()

    st.markdown("---")
    st.header("Analysis Parameters")
    
    st.subheader("Forecasting Parameters")
    model_choices = ['XGBoost', 'RandomForestRegressor']
    if PROPHET_AVAILABLE:
        model_choices.append('Prophet')
        model_choices.append('ARIMA')

    st.session_state.forecast_model = st.selectbox(
        "Select Forecasting Model",
        options=model_choices,
        help="Choose a model for demand forecasting. Note: Prophet and ARIMA may fail due to complex dependencies."
    )
    st.session_state.forecast_horizon = st.number_input(
        "Forecast Horizon (in days)",
        min_value=1,
        value=90,
        help="The number of days into the future to forecast demand."
    )
    st.session_state.forecast_roll_up_choice = st.selectbox(
        "Forecast Roll-up",
        options=["Daily", "Weekly", "Monthly"],
        help="Aggregate the forecast and results by day, week, or month."
    )

    st.subheader("Inventory Optimization Parameters")
    st.session_state.safety_stock_method = st.selectbox(
        "Safety Stock Calculation Method",
        options=["Probabilistic", "Heuristic", "Manual Input"],
        help="Choose a method to calculate safety stock."
    )
    st.session_state.service_level = st.slider(
        "Service Level (for Probabilistic method)",
        min_value=0.50,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="The desired probability of not having a stockout. (e.g., 95% = 0.95)"
    )
    st.session_state.reorder_point_buffer = st.number_input(
        "Manual Reorder Point (or Buffer for Safety Stock)",
        min_value=0,
        value=50,
        help="A manual buffer to be used for safety stock calculation if 'Manual Input' is selected."
    )


# --- Page 2: Analysis & Results ---
elif page == "Analysis & Results":
    st.header("Analysis and Results")
    st.markdown("""
    This page presents the results of the analysis, including model performance, demand forecasts, and inventory optimization KPIs.

    **Process Steps & Calculations:**
    1. **Data Preprocessing:** Historical sales data is aggregated and new features (like day of week, month, and lagged sales) are created to help the models learn patterns.
    2. **Model Training:** The selected forecasting model is trained on the historical data. The last portion of the data is held out to validate the model's accuracy.
    3. **Forecasting:** The trained model is used to predict future demand for the specified forecast horizon.
    4. **Inventory Simulation:** The forecasted demand is used to simulate inventory levels over time. Costs are calculated based on user-defined parameters or defaults.
    5. **KPI Calculation:** Key Performance Indicators (KPIs) like total cost, holding cost, ordering cost, and stockout rate are calculated from the simulation.
    """)

    if st.button("Run Analysis"):
        if st.session_state.sales_df.empty or st.session_state.sku_config_df.empty:
            st.error("Please upload the mandatory sales and SKU config files or run on sample data.")
        else:
            with st.spinner("Running analysis... This may take a moment."):
                # Get cost parameters from global config or use defaults
                if not st.session_state.global_config_df.empty:
                    global_config = st.session_state.global_config_df.set_index('Parameter').to_dict()['Value']
                    holding_cost = float(global_config.get('Holding_Cost_Per_Unit_Per_Day', DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY))
                    ordering_cost = float(global_config.get('Ordering_Cost_Per_Order', DEFAULT_ORDERING_COST_PER_ORDER))
                    shortage_cost = float(global_config.get('Shortage_Cost_Per_Unit', DEFAULT_SHORTAGE_COST_PER_UNIT))
                else:
                    holding_cost = DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY
                    ordering_cost = DEFAULT_ORDERING_COST_PER_ORDER
                    shortage_cost = DEFAULT_SHORTAGE_COST_PER_UNIT

                # 1. Preprocess data
                preprocessed_df = preprocess_data(st.session_state.sales_df, st.session_state.promotions_df)
                
                # 2. Train and Forecast
                st.session_state.forecast_df, st.session_state.model_performance = train_and_forecast(
                    preprocessed_df,
                    st.session_state.forecast_model,
                    st.session_state.forecast_horizon
                )

                # 3. Simulate inventory and calculate costs
                st.session_state.all_inventory_df, st.session_state.all_stockout_rates_df, st.session_state.all_costs_df = calculate_inventory_levels(
                    st.session_state.sales_df,
                    st.session_state.forecast_df,
                    st.session_state.sku_config_df,
                    holding_cost, ordering_cost, shortage_cost,
                    st.session_state.safety_stock_method,
                    st.session_state.service_level,
                    st.session_state.reorder_point_buffer
                )
            
            st.success("Analysis complete!")

    if not st.session_state.forecast_df.empty:
        st.subheader("Model Performance")
        st.dataframe(st.session_state.model_performance, use_container_width=True)

        st.markdown("---")
        st.subheader("Forecasted Demand & Inventory Simulation")
        
        # Select SKU for detailed plots
        unique_skus_for_plot = st.session_state.sales_df['SKU_ID'].unique()
        selected_sku_for_plot = st.selectbox("Select a SKU to view plots:", options=unique_skus_for_plot)
        
        # Forecast plot
        historical_sku_df = aggregate_kpi_for_plot(st.session_state.sales_df, selected_sku_for_plot, 'Sales_Quantity', st.session_state.forecast_roll_up_choice)
        forecast_sku_df = aggregate_kpi_for_plot(st.session_state.forecast_df, selected_sku_for_plot, 'Forecasted_Sales', st.session_state.forecast_roll_up_choice)
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=historical_sku_df['Date'], y=historical_sku_df['Value'], mode='lines', name='Historical Sales', line=dict(color='blue')))
        fig_forecast.add_trace(go.Scatter(x=forecast_sku_df['Date'], y=forecast_sku_df['Value'], mode='lines', name='Forecasted Sales', line=dict(color='orange')))
        
        fig_forecast.update_layout(
            title=f"Demand Forecast for {selected_sku_for_plot} (Roll-up: {st.session_state.forecast_roll_up_choice})",
            xaxis_title="Date",
            yaxis_title="Sales Quantity",
            hovermode="x unified"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Inventory plot
        if not st.session_state.all_inventory_df.empty:
            inventory_plot_df = st.session_state.all_inventory_df[st.session_state.all_inventory_df['SKU_ID'] == selected_sku_for_plot]
            
            fig_inventory = go.Figure()
            fig_inventory.add_trace(go.Scatter(x=inventory_plot_df['Date'], y=inventory_plot_df['Inventory_Level'], mode='lines', name='Inventory Level', line=dict(color='green')))
            fig_inventory.add_trace(go.Scatter(x=inventory_plot_df['Date'], y=inventory_plot_df['Safety_Stock'], mode='lines', name='Safety Stock', line=dict(dash='dot', color='red')))
            fig_inventory.add_trace(go.Scatter(x=inventory_plot_df['Date'], y=inventory_plot_df['Reorder_Point'], mode='lines', name='Reorder Point', line=dict(dash='dash', color='orange')))

            fig_inventory.update_layout(
                title=f"Inventory Simulation for {selected_sku_for_plot}",
                xaxis_title="Date",
                yaxis_title="Quantity",
                hovermode="x unified"
            )
            st.plotly_chart(fig_inventory, use_container_width=True)

        st.markdown("---")
        st.subheader("Inventory Optimization KPIs")
        col_kpi1, col_kpi2 = st.columns(2)
        with col_kpi1:
            if not st.session_state.all_costs_df.empty:
                st.metric("Total Holding Cost", f"${st.session_state.all_costs_df['Holding_Cost'].sum():,.2f}")
                st.metric("Total Ordering Cost", f"${st.session_state.all_costs_df['Ordering_Cost'].sum():,.2f}")
            else:
                st.info("No cost data available.")
        
        with col_kpi2:
            if not st.session_state.all_stockout_rates_df.empty:
                avg_stockout_rate = st.session_state.all_stockout_rates_df['Stockout_Rate'].mean()
                st.metric("Average Stockout Rate", f"{avg_stockout_rate:.2f}%")
            else:
                st.info("No stockout rate data available.")


        with st.expander("View detailed KPI tables"):
            st.subheader("Detailed Costs per SKU")
            st.dataframe(st.session_state.all_costs_df.set_index('SKU_ID'), use_container_width=True)
            
            st.subheader("Detailed Stockout Rates per SKU")
            st.dataframe(st.session_state.all_stockout_rates_df.set_index('SKU_ID'), use_container_width=True)

    else:
        st.info("Please go to the 'Data & Parameters' page to configure and run the analysis.")
