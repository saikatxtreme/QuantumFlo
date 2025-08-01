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
import graphviz # Import graphviz

# Prophet and ARIMA models have been removed as per the user's request due to dependency issues.
# import from statsmodels.tsa.arima.model.ARIMA has been removed
import warnings
warnings.filterwarnings('ignore') # Suppress warnings from libraries, etc.

# For Plotting
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration for Dummy Data Generation (used for templates and sample run) ---
DEFAULT_NUM_SKUS = 5
DEFAULT_NUM_COMPONENTS_PER_SKU = 3
DEFAULT_START_DATE = datetime(2023, 1, 1)
DEFAULT_END_DATE = datetime(2024, 12, 31)
DEFAULT_PROMOTION_FREQUENCY_DAYS = 60
DEFAULT_MAX_LEAD_TIME_DAYS = 30
DEFAULT_MAX_SKU_SHELF_LIFE_DAYS = 365
DEFAULT_SALES_CHANNELS = ["Distributor Network", "Amazon", "Own Website"]


# --- Default Cost Parameters (used if global_config.csv is not uploaded) ---
DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY = 0.10
DEFAULT_ORDERING_COST_PER_ORDER = 50.00

# --- Dummy Data Generation Functions (Unchanged) ---
def generate_inventory_data(sku_ids, start_date, end_date):
    """
    Generates dummy inventory data with a realistic trend.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    all_data = []

    for sku_id in sku_ids:
        for date in date_range:
            # Simulate inventory levels with some fluctuation
            inventory_level = max(0, 1000 + random.randint(-200, 200))
            all_data.append({
                'Date': date,
                'SKU_ID': sku_id,
                'Inventory_Level': inventory_level
            })
    return pd.DataFrame(all_data)


def generate_sales_data(sku_ids, start_date, end_date, sales_channels, promotion_frequency):
    """
    Generates dummy sales data with seasonal, trend, and promotional effects.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    all_data = []

    for sku_id in sku_ids:
        # Base sales trend
        base_sales = np.linspace(10, 50, len(date_range))
        noise = np.random.normal(0, 5, len(date_range))

        # Add seasonality (weekly and yearly)
        weekly_seasonality = 5 * np.sin(2 * np.pi * date_range.dayofweek / 7)
        yearly_seasonality = 10 * np.sin(2 * np.pi * date_range.dayofyear / 365.25)

        # Add promotion effects
        promotion_dates = pd.date_range(start=start_date, end=end_date, freq=f'{promotion_frequency}D')
        promotion_effect = pd.Series(0, index=date_range)
        for p_date in promotion_dates:
            promotion_effect.loc[p_date:p_date + timedelta(days=2)] = 20

        sales = base_sales + noise + weekly_seasonality + yearly_seasonality + promotion_effect

        for i, date in enumerate(date_range):
            channel = random.choice(sales_channels)
            units_sold = max(1, int(sales[i] + np.random.randint(-5, 5)))
            all_data.append({
                'Date': date,
                'SKU_ID': sku_id,
                'Sales_Channel': channel,
                'Units_Sold': units_sold
            })

    return pd.DataFrame(all_data)


def generate_sku_metadata(sku_ids, max_lead_time_days, max_sku_shelf_life_days):
    """
    Generates dummy SKU metadata like lead time and shelf life.
    """
    all_data = []
    for sku_id in sku_ids:
        lead_time = random.randint(1, max_lead_time_days)
        shelf_life = random.randint(30, max_sku_shelf_life_days)
        all_data.append({
            'SKU_ID': sku_id,
            'Lead_Time_Days': lead_time,
            'Shelf_Life_Days': shelf_life
        })
    return pd.DataFrame(all_data)


def generate_component_bom(sku_ids, num_components):
    """
    Generates a dummy Bill of Materials (BOM) for SKUs.
    """
    all_data = []
    component_ids = [f'COMP_{i+1}' for i in range(num_components * len(sku_ids))]
    for sku_id in sku_ids:
        for i in range(random.randint(1, num_components)):
            component_id = random.choice(component_ids)
            all_data.append({
                'SKU_ID': sku_id,
                'Component_ID': component_id,
                'Quantity': random.randint(1, 5)
            })
    return pd.DataFrame(all_data)


def generate_global_config(default_holding_cost, default_ordering_cost):
    """
    Generates dummy global configuration data.
    """
    return pd.DataFrame([{
        'Holding_Cost_Per_Unit_Per_Day': default_holding_cost,
        'Ordering_Cost_Per_Order': default_ordering_cost
    }])


def generate_dummy_data():
    """
    Generates a set of dummy dataframes for a full sample run.
    """
    st.info("Generating sample data...")
    # Define SKUs
    sku_ids = [f'SKU_{i+1}' for i in range(DEFAULT_NUM_SKUS)]

    # Generate dataframes
    sales_df = generate_sales_data(sku_ids, DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_SALES_CHANNELS, DEFAULT_PROMOTION_FREQUENCY_DAYS)
    sku_metadata_df = generate_sku_metadata(sku_ids, DEFAULT_MAX_LEAD_TIME_DAYS, DEFAULT_MAX_SKU_SHELF_LIFE_DAYS)
    inventory_df = generate_inventory_data(sku_ids, DEFAULT_START_DATE, DEFAULT_END_DATE)
    component_bom_df = generate_component_bom(sku_ids, DEFAULT_NUM_COMPONENTS_PER_SKU)
    global_config_df = generate_global_config(DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY, DEFAULT_ORDERING_COST_PER_ORDER)

    st.success("Sample data generated successfully!")
    return sales_df, sku_metadata_df, inventory_df, component_bom_df, global_config_df


# --- Core Forecasting and Inventory Logic ---
def create_features(df):
    """
    Creates time-series features for forecasting models.
    """
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Week'] = df.index.isocalendar().week.astype(int)
    df['Day'] = df.index.day
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfYear'] = df.index.dayofyear
    df['IsMonthEnd'] = df.index.is_month_end.astype(int)
    df['IsQuarterEnd'] = df.index.is_quarter_end.astype(int)
    df['IsYearEnd'] = df.index.is_year_end.astype(int)
    return df


def forecast_demand_ml(data, forecast_horizon=90, model_type='XGBoost'):
    """
    Forecasts demand using a machine learning model (XGBoost or Random Forest).
    """
    # Assuming 'data' is a DataFrame with a datetime index and 'Units_Sold' column
    df = data.copy()
    df = create_features(df)
    
    # Check for sufficient data
    if len(df) < 2:
        st.warning(f"Not enough data to train the model for SKU: {df.name}. Skipping forecast.")
        return pd.DataFrame()

    train_data = df[:-forecast_horizon]
    test_data = df[-forecast_horizon:]

    features = ['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'IsMonthEnd', 'IsQuarterEnd', 'IsYearEnd']
    target = 'Units_Sold'

    if train_data.empty:
        st.warning(f"Training data is empty for SKU: {df.name}. Skipping forecast.")
        return pd.DataFrame()

    X_train = train_data[features]
    y_train = train_data[target]
    
    if model_type == 'XGBoost':
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    else:  # 'Random Forest'
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)

    # Generate future dates for forecasting
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_horizon, freq='D')
    future_df = pd.DataFrame(index=future_dates)
    future_df = create_features(future_df)
    X_future = future_df[features]

    forecast_values = model.predict(X_future)
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Demand': np.maximum(0, forecast_values)})
    
    return forecast_df


def forecast_demand_moving_average(data, forecast_horizon=90, window=30):
    """
    Forecasts demand using a simple moving average model.
    """
    df = data.copy()
    if len(df) < window:
        st.warning(f"Not enough data for a {window}-day moving average for SKU: {df.name}. Skipping forecast.")
        return pd.DataFrame()

    # Calculate the moving average of the last `window` days
    last_known_avg = df['Units_Sold'].rolling(window=window).mean().iloc[-1]

    # Generate future dates and use the last moving average as the forecast
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_horizon, freq='D')
    forecast_values = np.full(forecast_horizon, last_known_avg)
    
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Demand': np.maximum(0, forecast_values)})
    
    return forecast_df


def forecast_demand(sales_df, selected_sku, model_type, forecast_horizon):
    """
    Selects and runs the appropriate forecasting model.
    """
    st.info(f"Forecasting demand for SKU: {selected_sku} using {model_type}...")
    
    sku_sales_df = sales_df[sales_df['SKU_ID'] == selected_sku]
    
    if sku_sales_df.empty:
        st.warning(f"No sales data found for SKU: {selected_sku}. Cannot forecast.")
        return pd.DataFrame()
    
    # Resample to get daily total sales
    sku_sales_daily = sku_sales_df.groupby('Date').agg({'Units_Sold': 'sum'})
    sku_sales_daily.name = selected_sku # Attach SKU name for better logging
    
    if model_type == 'XGBoost':
        forecast_df = forecast_demand_ml(sku_sales_daily, forecast_horizon, 'XGBoost')
    elif model_type == 'Random Forest':
        forecast_df = forecast_demand_ml(sku_sales_daily, forecast_horizon, 'Random Forest')
    elif model_type == 'Moving Average':
        # You can adjust the moving average window here
        forecast_df = forecast_demand_moving_average(sku_sales_daily, forecast_horizon, window=30)
    else:
        st.warning(f"Unknown model type: {model_type}. Falling back to XGBoost.")
        forecast_df = forecast_demand_ml(sku_sales_daily, forecast_horizon, 'XGBoost')

    if not forecast_df.empty:
        st.success(f"Demand forecast for {selected_sku} completed.")
        return forecast_df
    else:
        st.error(f"Demand forecast for {selected_sku} failed.")
        return pd.DataFrame()


def calculate_safety_stock(forecast_df, lead_time_days, service_level=0.95):
    """
    Calculates safety stock based on forecast variability and lead time.
    """
    if forecast_df.empty:
        return 0

    # Calculate the standard deviation of forecast errors (using a simple proxy here)
    # In a real-world scenario, you would use actual forecast errors.
    # Here, we'll use the standard deviation of the last 30 days of the forecast as a proxy for variability.
    if len(forecast_df['Forecasted_Demand']) > 30:
        demand_std_dev = forecast_df['Forecasted_Demand'].tail(30).std()
    else:
        demand_std_dev = forecast_df['Forecasted_Demand'].std()
    
    # Handle case where demand_std_dev is NaN (e.g., if forecast is a single value)
    if pd.isna(demand_std_dev):
        demand_std_dev = 0
    
    # Calculate Z-score for the desired service level
    z_score = norm.ppf(service_level)
    
    # Safety stock formula
    safety_stock = z_score * demand_std_dev * math.sqrt(lead_time_days)
    
    return max(0, math.ceil(safety_stock))


def calculate_reorder_point(avg_daily_demand, lead_time_days, safety_stock):
    """
    Calculates the reorder point.
    """
    return math.ceil((avg_daily_demand * lead_time_days) + safety_stock)


def calculate_optimal_order_quantity_eoq(holding_cost, ordering_cost, annual_demand):
    """
    Calculates optimal order quantity using the Economic Order Quantity (EOQ) model.
    """
    if annual_demand <= 0:
        return 0
    # The formula uses annual holding cost, so we multiply the daily cost by 365
    annual_holding_cost = holding_cost * 365
    if annual_holding_cost <= 0:
        return 0
    eoq = math.sqrt((2 * annual_demand * ordering_cost) / annual_holding_cost)
    return math.ceil(eoq)


def calculate_kpis(sales_df, inventory_df, forecast_df, sku_metadata_df, global_config_df, start_date, end_date):
    """
    Calculates key performance indicators for all SKUs.
    """
    st.info("Calculating KPIs for all SKUs...")
    all_kpis = []
    
    unique_skus = sales_df['SKU_ID'].unique()
    
    # Extract global costs
    holding_cost = global_config_df['Holding_Cost_Per_Unit_Per_Day'].iloc[0]
    ordering_cost = global_config_df['Ordering_Cost_Per_Order'].iloc[0]
    
    for sku_id in unique_skus:
        # Filter data for the current SKU
        sku_sales = sales_df[sales_df['SKU_ID'] == sku_id]
        sku_inventory = inventory_df[inventory_df['SKU_ID'] == sku_id]
        sku_metadata = sku_metadata_df[sku_metadata_df['SKU_ID'] == sku_id]
        
        # Calculate demand and inventory related metrics
        total_demand = sku_sales['Units_Sold'].sum()
        total_days = (end_date - start_date).days + 1
        avg_daily_demand = total_demand / total_days if total_days > 0 else 0
        
        # Calculate stockout rates (simplified)
        # This is a simplification. A real stockout rate calculation would require
        # a comparison of demand vs. available inventory on a daily basis.
        # We assume a stockout if inventory is zero.
        stockout_days = len(sku_inventory[sku_inventory['Inventory_Level'] == 0])
        stockout_rate = (stockout_days / total_days) * 100 if total_days > 0 else 0
        
        # Calculate cost metrics
        total_holding_cost = (sku_inventory['Inventory_Level'] * holding_cost).sum()
        # This is a simplification, assumes a constant ordering frequency.
        # A more complex model would track actual orders.
        num_orders = total_demand / 100 # Assume an average order size of 100
        total_ordering_cost = num_orders * ordering_cost
        total_inventory_cost = total_holding_cost + total_ordering_cost
        
        # Get lead time for safety stock calculation
        lead_time_days = sku_metadata['Lead_Time_Days'].iloc[0] if not sku_metadata.empty else DEFAULT_MAX_LEAD_TIME_DAYS
        
        # Safety stock and reorder point calculation
        if not forecast_df.empty:
            safety_stock = calculate_safety_stock(forecast_df[forecast_df['SKU_ID'] == sku_id], lead_time_days)
            reorder_point = calculate_reorder_point(avg_daily_demand, lead_time_days, safety_stock)
            eoq = calculate_optimal_order_quantity_eoq(holding_cost, ordering_cost, total_demand)
        else:
            safety_stock = 0
            reorder_point = 0
            eoq = 0
            
        all_kpis.append({
            'SKU_ID': sku_id,
            'Avg_Daily_Demand': avg_daily_demand,
            'Total_Demand': total_demand,
            'Stockout_Rate': stockout_rate,
            'Total_Holding_Cost': total_holding_cost,
            'Total_Ordering_Cost': total_ordering_cost,
            'Total_Inventory_Cost': total_inventory_cost,
            'Safety_Stock': safety_stock,
            'Reorder_Point': reorder_point,
            'Optimal_Order_Quantity_EOQ': eoq
        })
        
    st.success("KPI calculation completed.")
    return pd.DataFrame(all_kpis)


def aggregate_kpi_for_plot(df, selected_sku, kpi_name, roll_up_choice, aggregation_method='sum'):
    """
    Aggregates KPI data for plotting based on user's roll-up choice.
    """
    if df.empty or 'Date' not in df.columns:
        st.info("Input DataFrame is empty or does not have a 'Date' column.")
        return pd.DataFrame()
    
    plot_df = df[df['SKU_ID'] == selected_sku].copy()
    
    if plot_df.empty:
        return pd.DataFrame()
    
    plot_df['Date'] = pd.to_datetime(plot_df['Date'])
    plot_df.set_index('Date', inplace=True)
    
    # Assuming the KPI is already a column in the DataFrame
    plot_df['Value'] = plot_df[kpi_name]
    
    if roll_up_choice == 'Daily':
        # The data is already daily, just return it
        return plot_df.reset_index()
    elif roll_up_choice == 'Weekly':
        if aggregation_method == 'mean':
            return plot_df.resample('W').mean().reset_index()
        else:
            return plot_df.resample('W').sum().reset_index()
    elif roll_up_choice == 'Monthly':
        if aggregation_method == 'mean':
            return plot_df.resample('M').mean().reset_index()
        else:
            return plot_df.resample('M').sum().reset_index()
    else:
        return plot_df.reset_index()
    

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="QuantumFlo Demand and Inventory Intelligence", page_icon=":bar_chart:")
st.title(":bar_chart: QuantumFlo Demand and Inventory Intelligence")

# Initialize session state variables if they don't exist
if 'sales_df' not in st.session_state:
    st.session_state.sales_df = pd.DataFrame()
if 'sku_metadata_df' not in st.session_state:
    st.session_state.sku_metadata_df = pd.DataFrame()
if 'inventory_df' not in st.session_state:
    st.session_state.inventory_df = pd.DataFrame()
if 'component_bom_df' not in st.session_state:
    st.session_state.component_bom_df = pd.DataFrame()
if 'global_config_df' not in st.session_state:
    st.session_state.global_config_df = pd.DataFrame()
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = pd.DataFrame()
if 'all_kpis_df' not in st.session_state:
    st.session_state.all_kpis_df = pd.DataFrame()
if 'all_stockout_rates_df' not in st.session_state:
    st.session_state.all_stockout_rates_df = pd.DataFrame()

# --- Sidebar for Data Upload and Configuration ---
with st.sidebar:
    st.header("Data Configuration")
    data_source_choice = st.radio("Choose Data Source", ("Upload Your Own", "Run on Sample Data"))

    if data_source_choice == "Upload Your Own":
        st.info("Upload CSV files for your data.")
        uploaded_sales_file = st.file_uploader("Upload Sales Data (CSV)", type="csv")
        uploaded_sku_metadata_file = st.file_uploader("Upload SKU Metadata (CSV)", type="csv")
        uploaded_inventory_file = st.file_uploader("Upload Inventory Data (CSV)", type="csv")
        uploaded_bom_file = st.file_uploader("Upload Component BOM (CSV)", type="csv")
        uploaded_global_config_file = st.file_uploader("Upload Global Config (CSV)", type="csv")

        if st.button("Load Uploaded Data"):
            if uploaded_sales_file:
                st.session_state.sales_df = pd.read_csv(uploaded_sales_file)
            if uploaded_sku_metadata_file:
                st.session_state.sku_metadata_df = pd.read_csv(uploaded_sku_metadata_file)
            if uploaded_inventory_file:
                st.session_state.inventory_df = pd.read_csv(uploaded_inventory_file)
            if uploaded_bom_file:
                st.session_state.component_bom_df = pd.read_csv(uploaded_bom_file)
            if uploaded_global_config_file:
                st.session_state.global_config_df = pd.read_csv(uploaded_global_config_file)
            st.success("Data loaded successfully!")

    if data_source_choice == "Run on Sample Data":
        if st.button("Generate and Load Sample Data"):
            (st.session_state.sales_df,
             st.session_state.sku_metadata_df,
             st.session_state.inventory_df,
             st.session_state.component_bom_df,
             st.session_state.global_config_df) = generate_dummy_data()

# --- Main App Content ---
if not st.session_state.sales_df.empty:
    st.success("Data is available. You can now proceed with analysis and forecasting.")

    # Get unique SKUs for selection
    unique_skus = st.session_state.sales_df['SKU_ID'].unique()
    selected_sku = st.selectbox("Select SKU for Analysis", unique_skus)

    # --- Forecasting Section ---
    st.header("Demand Forecasting")
    
    # Model selection without Prophet
    model_choice = st.radio(
        "Choose Forecasting Model",
        ("XGBoost", "Random Forest", "Moving Average"),
        index=0 # Default to XGBoost
    )
    forecast_horizon = st.slider("Forecast Horizon (in days)", 30, 365, 90)

    if st.button("Run Forecast"):
        if selected_sku:
            forecast_df = forecast_demand(st.session_state.sales_df, selected_sku, model_choice, forecast_horizon)
            
            if not forecast_df.empty:
                st.session_state.forecast_results = forecast_df
                
                # Plotting the forecast
                combined_df = st.session_state.sales_df[st.session_state.sales_df['SKU_ID'] == selected_sku]
                combined_df = combined_df.groupby('Date')['Units_Sold'].sum().reset_index()
                
                fig = px.line(combined_df, x='Date', y='Units_Sold', title=f"Historical and Forecasted Demand for {selected_sku}")
                
                # Add the forecast line
                fig.add_trace(go.Scatter(
                    x=st.session_state.forecast_results['Date'],
                    y=st.session_state.forecast_results['Forecasted_Demand'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', dash='dot')
                ))
                
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Forecast could not be generated. Please check the data.")
        else:
            st.warning("Please select an SKU to run the forecast.")

    # --- KPI Analysis Section ---
    st.header("Inventory KPIs and Optimization")
    if st.button("Calculate KPIs"):
        if (not st.session_state.sales_df.empty and 
            not st.session_state.inventory_df.empty and 
            not st.session_state.sku_metadata_df.empty and 
            not st.session_state.global_config_df.empty):
            
            # Combine forecast with sales to get a full time series for stockout analysis
            # This is a placeholder for a more robust stockout calculation
            all_stockout_df = st.session_state.inventory_df.copy()
            
            # The calculation of stockout rate here is simplified.
            # It should be a more complex simulation comparing daily demand vs. daily inventory
            all_stockout_df['Date'] = pd.to_datetime(all_stockout_df['Date'])
            all_stockout_df['Stockout_Rate'] = np.where(all_stockout_df['Inventory_Level'] <= 0, 100, 0)
            
            st.session_state.all_stockout_rates_df = all_stockout_df
            st.session_state.all_kpis_df = calculate_kpis(
                st.session_state.sales_df,
                st.session_state.inventory_df,
                st.session_state.forecast_results,
                st.session_state.sku_metadata_df,
                st.session_state.global_config_df,
                DEFAULT_START_DATE,
                DEFAULT_END_DATE
            )
            st.success("KPIs calculated and ready for review.")
        else:
            st.warning("Please load all data files before calculating KPIs.")

    if not st.session_state.all_kpis_df.empty:
        st.subheader("Inventory KPIs Summary")
        st.dataframe(st.session_state.all_kpis_df, use_container_width=True)
        
        # KPI Plotting
        st.subheader(f"KPIs for {selected_sku}")
        kpi_to_plot = st.selectbox(
            "Select KPI to Plot",
            ('Total_Demand', 'Stockout_Rate', 'Total_Inventory_Cost')
        )
        forecast_roll_up_choice = st.selectbox(
            "Roll-up Period for Plot",
            ('Daily', 'Weekly', 'Monthly'),
            key='kpi_roll_up'
        )

        # Plotting Demand
        st.subheader("Historical Demand over time")
        if not st.session_state.sales_df.empty:
            demand_plot_df = aggregate_kpi_for_plot(
                st.session_state.sales_df, selected_sku, 'Units_Sold', forecast_roll_up_choice, 'sum'
            )
            if not demand_plot_df.empty:
                fig_demand = px.line(
                    demand_plot_df,
                    x="Date",
                    y="Value",
                    title=f"Historical Demand for {selected_sku} (Roll-up: {forecast_roll_up_choice})",
                    labels={"Value": "Units Sold", "Date": forecast_roll_up_choice},
                    color_discrete_sequence=['blue']
                )
                fig_demand.update_layout(hovermode="x unified")
                st.plotly_chart(fig_demand, use_container_width=True)
            else:
                st.info(f"No Demand data to plot for {selected_sku}.")
        else:
            st.info("No sales data available. Please upload your data or run on Sample Data.")

        # Plotting Stockout Rate
        st.subheader("Stockout Rate over time")
        if not st.session_state.inventory_df.empty:
            unique_skus_for_stockout_plot = st.session_state.inventory_df['SKU_ID'].unique()
            if len(unique_skus_for_stockout_plot) > 0:
                stockout_plot_df = aggregate_kpi_for_plot(
                    st.session_state.all_stockout_rates_df, selected_sku, 'Stockout_Rate', forecast_roll_up_choice, 'mean'
                )
                
                if not stockout_plot_df.empty:
                    fig_stockout = px.line(
                        stockout_plot_df,
                        x="Date",
                        y="Value",
                        title=f"Stockout Rate for {selected_sku} (Roll-up: {forecast_roll_up_choice})",
                        labels={"Value": "Stockout Rate (%)", "Date": forecast_roll_up_choice},
                        color_discrete_sequence=['orange']
                    )
                    fig_stockout.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_stockout, use_container_width=True)
                else:
                    st.info(f"No Stockout Rate data to plot for {selected_sku}.")
            else:
                st.info("No SKUs available for Stockout Rate plotting.")
        else:
            st.info("No Inventory data available. Please upload your data or run on Sample Data.")

else:
    st.info("Please upload your data or generate sample data to begin.")
