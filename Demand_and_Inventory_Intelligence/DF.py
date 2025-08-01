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
import math
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# --- Configuration for Dummy Data Generation ---
DEFAULT_NUM_SKUS = 5
DEFAULT_NUM_COMPONENTS_PER_SKU = 3
DEFAULT_START_DATE = datetime(2023, 1, 1)
DEFAULT_END_DATE = datetime(2024, 12, 31)
DEFAULT_PROMOTION_FREQUENCY_DAYS = 60
DEFAULT_MAX_LEAD_TIME_DAYS = 30
DEFAULT_MAX_SKU_SHELF_LIFE_DAYS = 365
DEFAULT_SALES_CHANNELS = ["Distributor Network", "Amazon", "Own Website"]
DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY = 0.10
DEFAULT_ORDERING_COST_PER_ORDER = 50.00

# --- Data Generation Functions ---
@st.cache_data
def generate_all_sample_data(num_skus, num_components_per_sku, start_date, end_date):
    """
    Generates all necessary sample data for the application.
    Returns a dictionary of DataFrames.
    """
    st.info("Generating all sample data. This may take a moment...")
    
    # 1. Generate BOM
    bom_data = []
    component_types = ["Raw_Material", "Packaging", "Sub_Assembly"]
    component_count = 1
    for i in range(1, num_skus + 1):
        sku_id = f"SKU_{i:03d}"
        num_components = random.randint(1, num_components_per_sku)
        for _ in range(num_components):
            comp_id = f"COMP_{component_count:03d}"
            bom_data.append({
                "Parent_SKU_ID": sku_id,
                "Component_ID": comp_id,
                "Component_Type": random.choice(component_types),
                "Quantity_Required": random.randint(1, 5)
            })
            component_count += 1
    bom_df = pd.DataFrame(bom_data)
    
    unique_skus = bom_df['Parent_SKU_ID'].unique()
    unique_components = bom_df['Component_ID'].unique()
    
    # 2. Generate Sales Data
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    sales_data = []
    for sku_id in unique_skus:
        for channel in DEFAULT_SALES_CHANNELS:
            base_demand = random.randint(50, 200)
            if channel == "Amazon": base_demand = int(base_demand * 1.2)
            elif channel == "Own Website": base_demand = int(base_demand * 0.8)
            seasonality_amplitude = base_demand * 0.3
            trend_slope = random.uniform(0.01, 0.05)
            for j, date in enumerate(dates):
                seasonality = seasonality_amplitude * np.sin(2 * np.pi * (date.dayofyear / 365))
                trend = trend_slope * j
                quantity = max(0, int(base_demand + seasonality + trend + random.randint(-20, 20)))
                sales_data.append({
                    "Date": date, "SKU_ID": sku_id, "Sales_Quantity": quantity,
                    "Sales_Channel": channel
                })
    sales_df = pd.DataFrame(sales_data)
    
    # 3. Generate Inventory Data (simplified)
    inventory_data = []
    current_sku_stock = {sku: random.randint(500, 1500) for sku in unique_skus}
    for date in dates:
        daily_sales = sales_df[sales_df['Date'] == date].groupby('SKU_ID')['Sales_Quantity'].sum()
        for sku_id, stock in current_sku_stock.items():
            sold = daily_sales.get(sku_id, 0)
            current_sku_stock[sku_id] = max(0, stock - sold)
            inventory_data.append({"Date": date, "Item_ID": sku_id, "Current_Stock": current_sku_stock[sku_id], "Item_Type": "Finished_Good"})
    inventory_df = pd.DataFrame(inventory_data)
    
    # 4. Generate Promotions
    promo_data = []
    unique_dates = sorted(sales_df['Date'].unique())
    for i in range(0, len(unique_dates), DEFAULT_PROMOTION_FREQUENCY_DAYS):
        promo_date = unique_dates[i]
        skus_on_promo = random.sample(list(unique_skus), min(random.randint(1, 3), len(unique_skus)))
        for sku_id in skus_on_promo:
            promo_channel = random.choice(DEFAULT_SALES_CHANNELS + ["All"])
            channels_for_promo = DEFAULT_SALES_CHANNELS if promo_channel == "All" else [promo_channel]
            for channel in channels_for_promo:
                promo_data.append({"Date": promo_date, "SKU_ID": sku_id, "Promotion_Type": "Discount", "Sales_Channel": channel})
    promo_df = pd.DataFrame(promo_data)

    # 5. Generate External Factors
    external_factors = []
    for date in dates:
        external_factors.append({
            "Date": date, "Economic_Index": random.uniform(90, 110),
            "Holiday_Flag": 1 if date.dayofyear in [1, 359] else 0
        })
    external_df = pd.DataFrame(external_factors)
    
    # 6. Generate Lead Times & Costs
    lead_times_data = []
    cost_data = []
    for sku_id in unique_skus:
        lead_times_data.append({"Item_ID": sku_id, "Item_Type": "Finished_Good", "Lead_Time_Days": random.randint(5, DEFAULT_MAX_LEAD_TIME_DAYS)})
        cost_data.append({"Item_ID": sku_id, "Holding_Cost_Per_Unit_Per_Day": round(DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY * random.uniform(0.5, 1.5), 2), "Unit_Cost": round(random.uniform(50, 200), 2)})
    cost_data.append({"Item_ID": "GLOBAL_CONFIG", "Ordering_Cost_Per_Order": DEFAULT_ORDERING_COST_PER_ORDER})
    lead_times_df = pd.DataFrame(lead_times_data)
    cost_config_df = pd.DataFrame(cost_data)
    
    st.success("Sample data generated successfully!")
    return {
        "sales_df": sales_df, "inventory_df": inventory_df, "bom_df": bom_df,
        "promo_df": promo_df, "external_factors_df": external_df,
        "lead_times_df": lead_times_df, "cost_config_df": cost_config_df
    }


# --- Data Preprocessing ---
def preprocess_data(sales_df, promo_df, external_df, roll_up_choice):
    """
    Combines and aggregates sales data with external factors and promotions.
    """
    
    # Ensure 'Date' columns are datetime objects
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    promo_df['Date'] = pd.to_datetime(promo_df['Date'])
    external_df['Date'] = pd.to_datetime(external_df['Date'])

    # Aggregate sales to the chosen frequency
    if roll_up_choice == 'Weekly':
        sales_df['Date'] = sales_df['Date'].dt.to_period('W').dt.start_time
    elif roll_up_choice == 'Monthly':
        sales_df['Date'] = sales_df['Date'].dt.to_period('M').dt.start_time
    
    agg_sales_df = sales_df.groupby(['Date', 'SKU_ID', 'Sales_Channel'])['Sales_Quantity'].sum().reset_index()
    
    # Merge with promotions
    agg_sales_df = pd.merge(agg_sales_df, promo_df.drop_duplicates(subset=['Date', 'SKU_ID', 'Sales_Channel']),
                            on=['Date', 'SKU_ID', 'Sales_Channel'], how='left')
    agg_sales_df['Promotion_Flag'] = agg_sales_df['Promotion_Type'].notna().astype(int)
    
    # Merge with external factors
    if roll_up_choice == 'Weekly':
        external_df['Date'] = external_df['Date'].dt.to_period('W').dt.start_time
        agg_external_df = external_df.groupby('Date').mean().reset_index()
    elif roll_up_choice == 'Monthly':
        external_df['Date'] = external_df['Date'].dt.to_period('M').dt.start_time
        agg_external_df = external_df.groupby('Date').mean().reset_index()
    else: # Daily
        agg_external_df = external_df.copy()
        
    agg_sales_df = pd.merge(agg_sales_df, agg_external_df, on='Date', how='left')
    
    agg_sales_df['Promotion_Flag'] = agg_sales_df['Promotion_Flag'].fillna(0)
    
    # Create new features from date
    agg_sales_df['Day_of_Week'] = agg_sales_df['Date'].dt.dayofweek
    agg_sales_df['Month'] = agg_sales_df['Date'].dt.month
    agg_sales_df['Year'] = agg_sales_df['Date'].dt.year

    return agg_sales_df.dropna()


# --- Forecasting Models ---
def train_and_forecast(model_name, data_df, sku_id, sales_channel, forecast_horizon):
    """
    Trains a selected model and generates a forecast.
    """
    sku_channel_df = data_df[(data_df['SKU_ID'] == sku_id) & (data_df['Sales_Channel'] == sales_channel)].copy()
    if sku_channel_df.empty: return pd.DataFrame()
    sku_channel_df.sort_values('Date', inplace=True)
    
    features = [col for col in sku_channel_df.columns if col not in ['Date', 'SKU_ID', 'Sales_Channel', 'Sales_Quantity', 'Promotion_Type']]
    target = 'Sales_Quantity'
    
    if len(sku_channel_df) == 0:
        return pd.DataFrame()
    
    forecast_df = pd.DataFrame()
    last_date = sku_channel_df['Date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon)
    
    if model_name in ['Moving Average', 'Moving Median']:
        window = 30
        if len(sku_channel_df) < window:
            st.warning(f"Not enough data for {model_name} on SKU {sku_id}. Skipping.")
            return pd.DataFrame()
        
        last_n_sales = sku_channel_df['Sales_Quantity'].tail(window)
        if model_name == 'Moving Average':
            forecast_value = last_n_sales.mean()
        else:
            forecast_value = last_n_sales.median()
        
        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Quantity': forecast_value})
    
    elif model_name in ['XGBoost', 'Random Forest']:
        X = sku_channel_df[features]
        y = sku_channel_df[target]
        
        if model_name == 'XGBoost':
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X, y)
        
        future_data = {f: [sku_channel_df[f].iloc[-1]] * forecast_horizon for f in features}
        future_data['Date'] = future_dates
        future_df = pd.DataFrame(future_data)
        
        for date_col in ['Day_of_Week', 'Month', 'Year']:
            if date_col in future_df.columns:
                future_df[date_col] = [getattr(d, date_col.lower()) for d in future_dates]

        forecast_values = model.predict(future_df[features])
        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Quantity': np.maximum(0, forecast_values)})

    return forecast_df


def run_auto_model_selection(data_df, sku_id, sales_channel, forecast_horizon):
    """
    Trains and evaluates all available models and selects the best one based on MAE.
    """
    available_models = ['XGBoost', 'Random Forest', 'Moving Average', 'Moving Median']
    results = {}
    best_model_name = None
    best_mae = float('inf')

    st.subheader(f"Auto Model Selection for {sku_id} ({sales_channel})")
    
    sku_channel_df = data_df[(data_df['SKU_ID'] == sku_id) & (data_df['Sales_Channel'] == sales_channel)].copy()
    if sku_channel_df.empty or len(sku_channel_df) < 60:
        st.warning(f"Not enough data for auto selection on {sku_id} ({sales_channel}).")
        return None, None
    
    # Use the last 30 periods as the test set to mimic real-world performance
    test_size = 30
    train_df = sku_channel_df.iloc[:-test_size]
    test_df = sku_channel_df.iloc[-test_size:]

    for model_name in available_models:
        st.write(f"  - Evaluating {model_name}...")
        try:
            if model_name in ['Moving Average', 'Moving Median']:
                window = 30
                if len(train_df) < window:
                    mae = float('inf')
                else:
                    last_n_sales = train_df['Sales_Quantity'].tail(window)
                    if model_name == 'Moving Average':
                        forecast_value = last_n_sales.mean()
                    else:
                        forecast_value = last_n_sales.median()
                    forecast_values = [forecast_value] * test_size
                    mae = mean_absolute_error(test_df['Sales_Quantity'], forecast_values)
            else: # ML Models
                features = [col for col in train_df.columns if col not in ['Date', 'SKU_ID', 'Sales_Channel', 'Sales_Quantity', 'Promotion_Type']]
                X_train, y_train = train_df[features], train_df['Sales_Quantity']
                X_test, y_test = test_df[features], test_df['Sales_Quantity']
                
                if model_name == 'XGBoost':
                    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
            
            results[model_name] = mae
            if mae < best_mae:
                best_mae = mae
                best_model_name = model_name
        except Exception as e:
            st.error(f"  - Error evaluating {model_name}: {e}")
            results[model_name] = float('inf')

    if best_model_name:
        st.success(f"Best model for {sku_id} ({sales_channel}): **{best_model_name}** with MAE of {best_mae:.2f}")
    else:
        st.error(f"Could not determine a best model for {sku_id} ({sales_channel}).")
        
    return best_model_name


# --- Inventory Optimization ---
def calculate_safety_stock(sales_df, lead_times_df, service_level, roll_up_choice):
    """Calculates safety stock based on historical demand variability and lead time."""
    safety_stock_results = []
    
    unique_skus = sales_df['SKU_ID'].unique()
    z_score = norm.ppf(service_level)
    
    for sku_id in unique_skus:
        sku_sales = sales_df[sales_df['SKU_ID'] == sku_id].copy()
        if sku_sales.empty: continue
        
        # Aggregate historical sales by the roll-up choice
        if roll_up_choice == 'Weekly':
            sku_sales['Period_Date'] = sku_sales['Date'].dt.to_period('W').dt.start_time
        elif roll_up_choice == 'Monthly':
            sku_sales['Period_Date'] = sku_sales['Date'].dt.to_period('M').dt.start_time
        else:
            sku_sales['Period_Date'] = sku_sales['Date']
        
        aggregated_demand = sku_sales.groupby('Period_Date')['Sales_Quantity'].sum()
        demand_std = aggregated_demand.std()
        
        lead_time_days = lead_times_df[lead_times_df['Item_ID'] == sku_id]['Lead_Time_Days'].iloc[0] if not lead_times_df[lead_times_df['Item_ID'] == sku_id].empty else 1
        
        if roll_up_choice == 'Daily':
            periods_in_lead_time = lead_time_days
        elif roll_up_choice == 'Weekly':
            periods_in_lead_time = max(1, math.ceil(lead_time_days / 7))
        else: # Monthly
            periods_in_lead_time = max(1, math.ceil(lead_time_days / 30))
        
        safety_stock = z_score * demand_std * np.sqrt(periods_in_lead_time)
        
        safety_stock_results.append({
            'Item_ID': sku_id,
            'Lead_Time_Days': lead_time_days,
            'Service_Level': service_level,
            'Safety_Stock_Units': round(safety_stock, 2)
        })
        
    return pd.DataFrame(safety_stock_results)


def calculate_inventory_metrics(forecast_df, safety_stock_df, lead_times_df, cost_config_df):
    """
    Calculates reorder point and other inventory costs.
    """
    inventory_results = []
    ordering_cost_per_order = cost_config_df[cost_config_df['Item_ID'] == 'GLOBAL_CONFIG']['Ordering_Cost_Per_Order'].iloc[0]

    for _, row in safety_stock_df.iterrows():
        sku_id = row['Item_ID']
        safety_stock = row['Safety_Stock_Units']
        lead_time_days = row['Lead_Time_Days']
        
        sku_forecast = forecast_df[forecast_df['SKU_ID'] == sku_id]
        if sku_forecast.empty: continue

        sku_cost = cost_config_df[cost_config_df['Item_ID'] == sku_id]
        if sku_cost.empty: continue
        
        holding_cost = sku_cost['Holding_Cost_Per_Unit_Per_Day'].iloc[0]
        unit_cost = sku_cost['Unit_Cost'].iloc[0]
        
        lead_time_demand = sku_forecast['Forecasted_Quantity'].head(lead_time_days).sum()
        reorder_point = lead_time_demand + safety_stock
        
        avg_daily_demand = sku_forecast['Forecasted_Quantity'].mean()
        annual_demand = avg_daily_demand * 365
        
        eoq = 0
        if annual_demand > 0 and holding_cost > 0 and ordering_cost_per_order > 0:
            eoq = np.sqrt((2 * annual_demand * ordering_cost_per_order) / (unit_cost * holding_cost * 365))
            
        avg_inventory_level = (eoq / 2) + safety_stock
        total_holding_cost = avg_inventory_level * holding_cost * 365
        
        num_orders_per_year = annual_demand / eoq if eoq > 0 else 0
        total_ordering_cost = num_orders_per_year * ordering_cost_per_order
        
        total_inventory_cost = total_holding_cost + total_ordering_cost

        inventory_results.append({
            'Item_ID': sku_id,
            'Reorder_Point': round(reorder_point, 2),
            'Safety_Stock_Units': round(safety_stock, 2),
            'EOQ': round(eoq, 2),
            'Average_Inventory_Level': round(avg_inventory_level, 2),
            'Total_Inventory_Cost_USD': round(total_inventory_cost, 2)
        })
    return pd.DataFrame(inventory_results)


# --- KPI & Analysis Functions ---
def calculate_kpis(sales_df, forecast_df, inventory_df):
    """Calculates key performance indicators."""
    kpi_results = {}
    
    # 1. Forecast Accuracy (MAPE)
    merged_df = pd.merge(sales_df, forecast_df, on=['Date', 'SKU_ID', 'Sales_Channel'], how='inner')
    if not merged_df.empty:
        # Filter for non-zero sales to avoid division by zero in MAPE
        merged_df = merged_df[merged_df['Sales_Quantity'] > 0].copy()
        if not merged_df.empty:
            mape = np.mean(np.abs((merged_df['Sales_Quantity'] - merged_df['Forecasted_Quantity']) / merged_df['Sales_Quantity'])) * 100
            kpi_results['MAPE'] = f"{mape:.2f}%"
        else:
            kpi_results['MAPE'] = "N/A (No sales in forecast period)"
    else:
        kpi_results['MAPE'] = "N/A (No overlapping data)"
        
    # 2. Stockout Rate
    stockout_rates = []
    for sku_id in sales_df['SKU_ID'].unique():
        sku_sales = sales_df[sales_df['SKU_ID'] == sku_id]
        sku_inventory = inventory_df[inventory_df['Item_ID'] == sku_id]
        
        if sku_sales.empty or sku_inventory.empty: continue
            
        merged_stockout_df = pd.merge(sku_sales, sku_inventory, left_on=['Date', 'SKU_ID'], right_on=['Date', 'Item_ID'], how='left')
        merged_stockout_df['Is_Stockout'] = (merged_stockout_df['Sales_Quantity'] > 0) & (merged_stockout_df['Current_Stock'] <= 0)
        
        total_sales_days = merged_stockout_df['Date'].nunique()
        stockout_days = merged_stockout_df[merged_stockout_df['Is_Stockout']]['Date'].nunique()
        
        stockout_rate = (stockout_days / total_sales_days) * 100 if total_sales_days > 0 else 0
        stockout_rates.append({'SKU_ID': sku_id, 'Stockout_Rate': stockout_rate})
        
    stockout_rate_df = pd.DataFrame(stockout_rates)
    kpi_results['Average Stockout Rate'] = f"{stockout_rate_df['Stockout_Rate'].mean():.2f}%" if not stockout_rate_df.empty else "N/A"

    # 3. Inventory Turnover
    total_sales_qty = sales_df['Sales_Quantity'].sum()
    avg_inventory = inventory_df['Current_Stock'].mean()
    
    inventory_turnover = total_sales_qty / avg_inventory if avg_inventory > 0 else 0
    kpi_results['Inventory Turnover'] = f"{inventory_turnover:.2f}"
    
    return kpi_results


# --- Streamlit UI ---
def main():
    st.set_page_config(layout="wide", page_title="Demand & Inventory Intelligence")
    st.title("Demand and Inventory Intelligence")
    st.subheader("A Comprehensive Forecasting and Optimization Solution")
    
    # --- Session State Initialization ---
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.forecast_df = pd.DataFrame()
        st.session_state.inventory_cost_df = pd.DataFrame()
        st.session_state.kpi_results = {}
        st.session_state.best_model = None

    # --- Sidebar ---
    with st.sidebar:
        st.header("Data & Parameters")
        data_source = st.radio("Choose Data Source", ("Run on Sample Data", "Upload Your Own"))
        
        if data_source == "Run on Sample Data":
            st.subheader("Sample Data Configuration")
            num_skus = st.slider("Number of SKUs", 1, 20, DEFAULT_NUM_SKUS)
            num_components = st.slider("Components per SKU", 1, 10, DEFAULT_NUM_COMPONENTS_PER_SKU)
            if st.button("Generate and Load Sample Data"):
                data = generate_all_sample_data(num_skus, num_components, DEFAULT_START_DATE, DEFAULT_END_DATE)
                for key, df in data.items():
                    st.session_state[key] = df
                st.session_state.data_loaded = True
                
        else: # Upload Your Own
            st.warning("Please upload all necessary files for a complete analysis.")
            uploaded_files = {
                "sales_df": st.file_uploader("Upload Sales Data (sales.csv)", type=["csv"]),
                "inventory_df": st.file_uploader("Upload Inventory Data (inventory.csv)", type=["csv"]),
                "bom_df": st.file_uploader("Upload Bill of Materials (bom.csv)", type=["csv"]),
                "promo_df": st.file_uploader("Upload Promotion Data (promotions.csv)", type=["csv"]),
                "external_factors_df": st.file_uploader("Upload External Factors (external_factors.csv)", type=["csv"]),
                "lead_times_df": st.file_uploader("Upload Lead Times (lead_times.csv)", type=["csv"]),
                "cost_config_df": st.file_uploader("Upload Cost Configuration (cost_config.csv)", type=["csv"]),
            }
            if st.button("Load Uploaded Data"):
                if all(uploaded_files.values()):
                    try:
                        for key, file in uploaded_files.items():
                            st.session_state[key] = pd.read_csv(file)
                        st.session_state.data_loaded = True
                        st.success("All data files loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading files: {e}")
                else:
                    st.error("Please upload all required files.")
    
    if st.session_state.data_loaded:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Forecasting & Inventory Settings")
        forecast_roll_up_choice = st.sidebar.selectbox("Forecast Roll-up", ["Daily", "Weekly", "Monthly"])
        forecast_horizon = st.sidebar.slider("Forecast Horizon (in days)", 1, 365, 30)
        model_options = ["Auto", 'XGBoost', 'Random Forest', 'Moving Average', 'Moving Median']
        selected_model = st.sidebar.selectbox("Choose Forecasting Model", model_options)
        service_level = st.sidebar.slider("Desired Service Level (%)", 50, 99, 95) / 100
        
        if st.sidebar.button("Run Analysis"):
            with st.spinner("Processing data and running models..."):
                preprocessed_data = preprocess_data(
                    st.session_state.sales_df, st.session_state.promo_df,
                    st.session_state.external_factors_df, forecast_roll_up_choice
                )
                
                unique_skus = preprocessed_data['SKU_ID'].unique()
                unique_channels = preprocessed_data['Sales_Channel'].unique()
                
                all_forecasts = []
                best_model_for_all = {}
                for sku in unique_skus:
                    for channel in unique_channels:
                        if selected_model == "Auto":
                            best_model_name = run_auto_model_selection(preprocessed_data, sku, channel, forecast_horizon)
                            if best_model_name:
                                best_model_for_all[(sku, channel)] = best_model_name
                                forecast_df_part = train_and_forecast(best_model_name, preprocessed_data, sku, channel, forecast_horizon)
                                if not forecast_df_part.empty:
                                    forecast_df_part['SKU_ID'] = sku
                                    forecast_df_part['Sales_Channel'] = channel
                                    all_forecasts.append(forecast_df_part)
                        else:
                            best_model_for_all[(sku, channel)] = selected_model
                            forecast_df_part = train_and_forecast(selected_model, preprocessed_data, sku, channel, forecast_horizon)
                            if not forecast_df_part.empty:
                                forecast_df_part['SKU_ID'] = sku
                                forecast_df_part['Sales_Channel'] = channel
                                all_forecasts.append(forecast_df_part)

                if all_forecasts:
                    st.session_state.forecast_df = pd.concat(all_forecasts, ignore_index=True)
                    st.session_state.best_model = best_model_for_all
                    st.success("Forecasting complete!")
                    
                    st.subheader("Running Inventory Optimization...")
                    st.session_state.safety_stock_df = calculate_safety_stock(st.session_state.sales_df, st.session_state.lead_times_df, service_level, forecast_roll_up_choice)
                    st.session_state.inventory_cost_df = calculate_inventory_metrics(st.session_state.forecast_df, st.session_state.safety_stock_df, st.session_state.lead_times_df, st.session_state.cost_config_df)
                    st.success("Inventory optimization complete!")
                    
                    st.subheader("Calculating KPIs...")
                    st.session_state.kpi_results = calculate_kpis(st.session_state.sales_df, st.session_state.forecast_df, st.session_state.inventory_df)
                    st.success("KPIs calculated!")
                else:
                    st.error("No forecasts were generated. Please check your data and parameters.")

    # --- Main Content Area ---
    if st.session_state.data_loaded:
        tab1, tab2, tab3 = st.tabs(["Demand Planning", "Inventory Optimization", "KPIs & Analysis"])
        
        with tab1:
            st.header("Demand Planning")
            if not st.session_state.forecast_df.empty:
                unique_skus_for_plot = st.session_state.sales_df['SKU_ID'].unique()
                unique_channels_for_plot = st.session_state.sales_df['Sales_Channel'].unique()
                
                col1, col2 = st.columns(2)
                with col1:
                    selected_sku_for_plot = st.selectbox("Select SKU for Plotting", unique_skus_for_plot)
                with col2:
                    selected_channel_for_plot = st.selectbox("Select Sales Channel", unique_channels_for_plot)
                
                historical_sales = st.session_state.sales_df[(st.session_state.sales_df['SKU_ID'] == selected_sku_for_plot) & (st.session_state.sales_df['Sales_Channel'] == selected_channel_for_plot)]
                forecast_sales = st.session_state.forecast_df[(st.session_state.forecast_df['SKU_ID'] == selected_sku_for_plot) & (st.session_state.forecast_df['Sales_Channel'] == selected_channel_for_plot)]
                
                if not historical_sales.empty and not forecast_sales.empty:
                    historical_sales.set_index('Date', inplace=True)
                    historical_sales = historical_sales.resample('D')['Sales_Quantity'].sum().reset_index()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=historical_sales['Date'], y=historical_sales['Sales_Quantity'], mode='lines', name='Historical Sales'))
                    fig.add_trace(go.Scatter(x=forecast_sales['Date'], y=forecast_sales['Forecasted_Quantity'], mode='lines', name='Forecasted Demand', line=dict(color='orange')))
                    
                    fig.update_layout(title=f"Sales and Forecast for {selected_sku_for_plot} ({selected_channel_for_plot})", xaxis_title="Date", yaxis_title="Sales Quantity")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data available to plot for the selected SKU and channel.")
            else:
                st.info("Please run the analysis to generate forecasts.")
                
        with tab2:
            st.header("Inventory Optimization")
            if not st.session_state.inventory_cost_df.empty:
                st.subheader("Inventory Metrics and Costs")
                st.dataframe(st.session_state.inventory_cost_df, use_container_width=True)
                
                unique_skus_for_inv = st.session_state.inventory_cost_df['Item_ID'].unique()
                selected_sku_for_inv = st.selectbox("Select SKU to View Details", unique_skus_for_inv, key='inv_plot_sku')
                
                if selected_sku_for_inv:
                    inv_df_sku = st.session_state.inventory_cost_df[st.session_state.inventory_cost_df['Item_ID'] == selected_sku_for_inv].iloc[0]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Reorder Point", f"{inv_df_sku['Reorder_Point']:.2f}")
                    col2.metric("Safety Stock", f"{inv_df_sku['Safety_Stock_Units']:.2f}")
                    col3.metric("EOQ", f"{inv_df_sku['EOQ']:.2f}")

            else:
                st.info("No inventory optimization data available. Please run the analysis first.")
                
        with tab3:
            st.header("KPIs & Analysis")
            if st.session_state.kpi_results:
                st.subheader("Overall Performance Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Average MAPE", st.session_state.kpi_results.get('MAPE', 'N/A'))
                col2.metric("Avg. Stockout Rate", st.session_state.kpi_results.get('Average Stockout Rate', 'N/A'))
                col3.metric("Inventory Turnover", st.session_state.kpi_results.get('Inventory Turnover', 'N/A'))
                
                st.subheader("Inventory Level over Time")
                if not st.session_state.inventory_df.empty:
                    unique_skus_for_inv_plot = st.session_state.inventory_df['Item_ID'].unique()
                    selected_sku_for_inv_plot = st.selectbox("Select SKU to view Inventory", unique_skus_for_inv_plot, key='inv_level_plot_sku')
                    
                    inventory_plot_df = st.session_state.inventory_df[st.session_state.inventory_df['Item_ID'] == selected_sku_for_inv_plot]
                    if not inventory_plot_df.empty:
                        fig_inv_level = px.line(inventory_plot_df, x="Date", y="Current_Stock", title=f"Inventory Level for {selected_sku_for_inv_plot}")
                        st.plotly_chart(fig_inv_level, use_container_width=True)
                    else:
                        st.info("No inventory data to plot for this SKU.")
                else:
                    st.info("No inventory data available. Please load data first.")

    else:
        st.info("Please use the sidebar to load sample data or upload your own to begin the analysis.")
        
if __name__ == "__main__":
    main()

