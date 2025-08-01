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
from io import StringIO
import base64
import json

# --- Configuration for Data Generation and Inventory Policy ---
# Using Streamlit's session state for persistent parameters
if 'NUM_SKUS' not in st.session_state:
    st.session_state.NUM_SKUS = 5
if 'NUM_COMPONENTS_PER_SKU' not in st.session_state:
    st.session_state.NUM_COMPONENTS_PER_SKU = 3
if 'START_DATE' not in st.session_state:
    st.session_state.START_DATE = datetime(2023, 1, 1)
if 'END_DATE' not in st.session_state:
    st.session_state.END_DATE = datetime(2024, 12, 31)
if 'FORECAST_HORIZON_DAYS' not in st.session_state:
    st.session_state.FORECAST_HORIZON_DAYS = 30
if 'SERVICE_LEVEL' not in st.session_state:
    st.session_state.SERVICE_LEVEL = 0.95
if 'HOLDING_COST_PER_UNIT_PER_DAY' not in st.session_state:
    st.session_state.HOLDING_COST_PER_UNIT_PER_DAY = 0.05
if 'ORDERING_COST_PER_ORDER' not in st.session_state:
    st.session_state.ORDERING_COST_PER_ORDER = 50.0

# --- Helper Functions to Load Data from Session State or Uploaded Files ---
def load_data(uploaded_file, default_data_func, **kwargs):
    if uploaded_file is not None:
        try:
            dataframe = pd.read_csv(uploaded_file)
            st.success(f"Successfully uploaded {uploaded_file.name}!")
            return dataframe
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    else:
        return default_data_func(**kwargs)

# --- Dummy Data Generation Functions (Unchanged) ---
def generate_sales_data(num_skus, start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    sales_data = []
    for i in range(1, num_skus + 1):
        sku_id = f"SKU_{i:03d}"
        base_demand = random.randint(50, 200)
        seasonality_amplitude = base_demand * 0.3
        trend_slope = random.uniform(0.01, 0.05)
        for j, date in enumerate(dates):
            seasonality = seasonality_amplitude * np.sin(2 * np.pi * (date.dayofyear / 365))
            trend = trend_slope * j
            noise = random.randint(-20, 20)
            quantity = max(0, int(base_demand + seasonality + trend + noise))
            sales_data.append({
                "Date": date,
                "SKU_ID": sku_id,
                "Sales_Quantity": quantity,
                "Price": round(random.uniform(10, 100), 2),
                "Customer_Segment": random.choice(["Retail", "Wholesale", "Online"])
            })
    return pd.DataFrame(sales_data)

def generate_inventory_data(sales_df, start_date):
    inventory_data = []
    initial_stock = {sku: random.randint(500, 1500) for sku in sales_df['SKU_ID'].unique()}
    current_stock = initial_stock.copy()
    dates = pd.date_range(start=start_date, end=sales_df['Date'].max(), freq='D')
    for date in dates:
        daily_sales = sales_df[sales_df['Date'] == date].set_index('SKU_ID')['Sales_Quantity']
        for sku_id, stock in current_stock.items():
            sold = daily_sales.get(sku_id, 0)
            current_stock[sku_id] = max(0, stock - sold + random.randint(0, 50))
            inventory_data.append({
                "Date": date,
                "SKU_ID": sku_id,
                "Current_Stock": current_stock[sku_id]
            })
    return pd.DataFrame(inventory_data)

def generate_promotion_data(sales_df, promo_frequency_days=60):
    promotion_data = []
    unique_dates = sorted(sales_df['Date'].unique())
    unique_skus = sales_df['SKU_ID'].unique()
    for i in range(0, len(unique_dates), promo_frequency_days):
        promo_date = unique_dates[i]
        num_promos = random.randint(1, 3)
        skus_on_promo = random.sample(list(unique_skus), min(num_promos, len(unique_skus)))
        for sku_id in skus_on_promo:
            promotion_data.append({
                "Date": promo_date,
                "SKU_ID": sku_id,
                "Promotion_Type": random.choice(["Discount", "BOGO", "Bundle"]),
                "Discount_Percentage": round(random.uniform(0.05, 0.30), 2) if "Discount" in random.choice(["Discount", "BOGO", "Bundle"]) else None
            })
    return pd.DataFrame(promotion_data)

def generate_external_factors_data(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    external_factors = []
    for date in dates:
        external_factors.append({
            "Date": date,
            "Economic_Index": round(random.uniform(90, 110), 2),
            "Holiday_Flag": 1 if date.month in [11, 12] and date.day in range(20, 32) else 0,
            "Temperature_Celsius": round(random.uniform(5, 35), 1),
            "Competitor_Activity_Index": round(random.uniform(0.5, 1.5), 2)
        })
    return pd.DataFrame(external_factors)

def generate_lead_times_data(skus, max_lead_time_days=30):
    lead_times = []
    for sku_id in skus:
        lead_times.append({
            "Item_ID": sku_id,
            "Item_Type": "Finished_Good",
            "Supplier_ID": f"SUP_{random.randint(1, 3)}",
            "Lead_Time_Days": random.randint(5, max_lead_time_days),
            "Shelf_Life_Days": random.choice([None, random.randint(90, 365)]),
            "Min_Order_Quantity": random.choice([10, 50, 100]),
            "Order_Multiple": random.choice([10, 20, 50])
        })
    return pd.DataFrame(lead_times)

def generate_bom_data(skus, num_components_per_sku, lead_times_df_ref):
    bom_data = []
    component_id_counter = 1000
    for sku_id in skus:
        num_components = random.randint(1, num_components_per_sku)
        for _ in range(num_components):
            component_id = f"COMP_{component_id_counter:04d}"
            component_id_counter += 1
            bom_data.append({
                "Parent_SKU_ID": sku_id,
                "Component_ID": component_id,
                "Quantity_Required": random.randint(1, 5),
                "Component_Type": random.choice(["Raw_Material", "Sub_Assembly"]),
                "Shelf_Life_Days": random.choice([None, random.randint(30, 180), random.randint(7, 20)])
            })
            if component_id not in lead_times_df_ref['Item_ID'].values:
                lead_times_df_ref.loc[len(lead_times_df_ref)] = {
                    "Item_ID": component_id,
                    "Item_Type": "Component",
                    "Supplier_ID": f"SUP_{random.randint(1, 5)}",
                    "Lead_Time_Days": random.randint(2, 15),
                    "Shelf_Life_Days": random.choice([None, random.randint(30, 180)]),
                    "Min_Order_Quantity": random.choice([100, 200, 500]),
                    "Order_Multiple": random.choice([50, 100])
                }
    return pd.DataFrame(bom_data)

# --- Core Logic Functions (with Streamlit caching) ---
@st.cache_data
def prepare_data_for_forecasting(sales_df, inventory_df, promotions_df, external_factors_df):
    """
    Merges all relevant dataframes and creates features for the ML model.
    This version uses vectorized operations for efficiency.
    """
    try:
        merged_df = pd.merge(sales_df, inventory_df, on=['Date', 'SKU_ID'], how='left')
        merged_df = pd.merge(merged_df, promotions_df, on=['Date', 'SKU_ID'], how='left')
        merged_df['Promotion_Active'] = merged_df['Promotion_Type'].notna().astype(int)
        merged_df['Discount_Percentage'] = merged_df['Discount_Percentage'].fillna(0)
        promo_type_dummies = pd.get_dummies(merged_df['Promotion_Type'], prefix='Promo_Type', dummy_na=False)
        merged_df = pd.concat([merged_df.drop(columns=['Promotion_Type']), promo_type_dummies], axis=1)
        merged_df = pd.merge(merged_df, external_factors_df, on='Date', how='left')
        merged_df['Date'] = pd.to_datetime(merged_df['Date'])
        merged_df = merged_df.sort_values(by=['SKU_ID', 'Date']).reset_index(drop=True)
        merged_df['Year'] = merged_df['Date'].dt.year
        merged_df['Month'] = merged_df['Date'].dt.month
        merged_df['Day'] = merged_df['Date'].dt.day
        merged_df['DayOfWeek'] = merged_df['Date'].dt.dayofweek
        merged_df['DayOfYear'] = merged_df['Date'].dt.dayofyear
        merged_df['WeekOfYear'] = merged_df['Date'].dt.isocalendar().week.astype(int)
        merged_df['Sales_Quantity_Lag_1'] = merged_df.groupby('SKU_ID')['Sales_Quantity'].shift(1).fillna(0)
        merged_df['Sales_Quantity_Lag_7'] = merged_df.groupby('SKU_ID')['Sales_Quantity'].shift(7).fillna(0)
        merged_df = merged_df.drop(columns=['Price', 'Customer_Segment'], errors='ignore')
        merged_df = merged_df.fillna(0)
        return merged_df
    except Exception as e:
        st.error(f"Error during data preparation: {e}")
        return pd.DataFrame()

@st.cache_resource
def train_forecast_model(data_df, model_type='XGBoost'):
    trained_models = {}
    evaluation_metrics = {}
    unique_skus = data_df['SKU_ID'].unique()
    for sku_id in unique_skus:
        sku_data = data_df[data_df['SKU_ID'] == sku_id].copy()
        if len(sku_data) < 10:
            st.warning(f"Not enough data to train for {sku_id}. Skipping.")
            continue
        features = [col for col in sku_data.columns if col not in ['SKU_ID', 'Sales_Quantity', 'Date']]
        X = sku_data[features]
        y = sku_data['Sales_Quantity']
        split_point = int(len(sku_data) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        if len(X_train) == 0 or len(X_test) == 0:
            st.warning(f"Insufficient data for train/test split for {sku_id}. Skipping.")
            continue
        try:
            if model_type == 'RandomForest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            trained_models[sku_id] = model
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            evaluation_metrics[sku_id] = {"MAE": mae, "RMSE": rmse}
        except Exception as e:
            st.error(f"Error training model for {sku_id}: {e}")
    return trained_models, evaluation_metrics

@st.cache_data
def predict_demand(trained_models, processed_data, forecast_horizon_days, external_factors_df):
    forecasts = {}
    current_date = processed_data['Date'].max()
    future_dates = pd.date_range(start=current_date + timedelta(days=1), periods=forecast_horizon_days, freq='D')
    
    future_data_list = []
    
    for sku_id in trained_models.keys():
        last_sku_row = processed_data[processed_data['SKU_ID'] == sku_id].sort_values(by='Date', ascending=False).iloc[0]
        last_sales_quantity = last_sku_row['Sales_Quantity']
        last_sales_quantity_lag_7 = last_sku_row['Sales_Quantity_Lag_7']

        for i, forecast_date in enumerate(future_dates):
            future_row_dict = {
                'Date': forecast_date,
                'SKU_ID': sku_id,
                'Current_Stock': last_sku_row['Current_Stock'],
                'Promotion_Active': 0,
                'Discount_Percentage': 0,
                'Year': forecast_date.year,
                'Month': forecast_date.month,
                'Day': forecast_date.day,
                'DayOfWeek': forecast_date.dayofweek,
                'DayOfYear': forecast_date.dayofyear,
                'WeekOfYear': forecast_date.isocalendar().week,
                'Sales_Quantity_Lag_1': last_sales_quantity,
                'Sales_Quantity_Lag_7': last_sales_quantity_lag_7
            }
            last_sales_quantity_lag_7 = last_sku_row['Sales_Quantity_Lag_1']
            last_sales_quantity = 0 
            future_data_list.append(future_row_dict)

    if not future_data_list:
        st.warning("No future data to forecast.")
        return {}

    future_df = pd.DataFrame(future_data_list)
    future_df = pd.merge(future_df, external_factors_df, on='Date', how='left')
    future_df = future_df.fillna(0)

    all_forecasts_df = pd.DataFrame()

    for sku_id, model in trained_models.items():
        sku_future_df = future_df[future_df['SKU_ID'] == sku_id].copy()
        features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [col for col in sku_future_df.columns if col not in ['SKU_ID', 'Date']]
        
        missing_features = [f for f in features if f not in sku_future_df.columns]
        if missing_features:
            st.warning(f"Warning: Missing features for {sku_id}: {missing_features}. Skipping.")
            continue
            
        X_future = sku_future_df[features]
        predictions = model.predict(X_future)
        sku_future_df['Forecasted_Quantity'] = [max(0, int(q)) for q in predictions]
        
        for i in range(1, len(sku_future_df)):
            sku_future_df.loc[sku_future_df.index[i], 'Sales_Quantity_Lag_1'] = sku_future_df.loc[sku_future_df.index[i-1], 'Forecasted_Quantity']
            sku_future_df.loc[sku_future_df.index[i], 'Sales_Quantity_Lag_7'] = sku_future_df.loc[sku_future_df.index[i-1], 'Sales_Quantity_Lag_7']
        
        forecasts[sku_id] = sku_future_df[['Date', 'SKU_ID', 'Forecasted_Quantity']]

    return forecasts

@st.cache_data
def calculate_auto_indent(forecasts, inventory_df, lead_times_df, evaluation_metrics, service_level, holding_cost_per_unit_per_day):
    indent_recommendations = []
    current_date = datetime.now()
    Z_SCORE = norm.ppf(service_level)
    
    for sku_id, forecast_df in forecasts.items():
        latest_inventory_row = inventory_df[inventory_df['SKU_ID'] == sku_id].sort_values(by='Date', ascending=False).iloc[0]
        current_stock = latest_inventory_row['Current_Stock']

        sku_lead_time_row = lead_times_df[(lead_times_df['Item_ID'] == sku_id) & (lead_times_df['Item_Type'] == 'Finished_Good')]
        if sku_lead_time_row.empty:
            st.warning(f"Warning: Lead time or details not found for SKU {sku_id}. Skipping auto-indent.")
            continue
        
        lead_time_days = sku_lead_time_row['Lead_Time_Days'].iloc[0]
        sku_shelf_life = sku_lead_time_row['Shelf_Life_Days'].iloc[0]
        min_order_quantity = sku_lead_time_row['Min_Order_Quantity'].iloc[0]
        order_multiple = sku_lead_time_row['Order_Multiple'].iloc[0]

        forecast_error_std = evaluation_metrics.get(sku_id, {}).get("RMSE", 0)
        
        if forecast_error_std > 0 and lead_time_days > 0:
            safety_stock = int(Z_SCORE * forecast_error_std * np.sqrt(lead_time_days))
        else:
            safety_stock = int(forecast_df.head(lead_time_days)['Forecasted_Quantity'].sum() * 0.1)
        
        capped_safety_stock = safety_stock
        if sku_shelf_life is not None:
            shelf_life_demand = forecast_df.head(int(sku_shelf_life))['Forecasted_Quantity'].sum()
            capped_safety_stock = min(safety_stock, int(shelf_life_demand * 0.2))

        forecast_during_lead_time = forecast_df.head(lead_time_days)['Forecasted_Quantity'].sum()
        reorder_point_days = 7 # Fixed for simplicity in this Streamlit example
        reorder_point_demand_buffer = forecast_df.head(reorder_point_days)['Forecasted_Quantity'].sum()

        target_stock_level = int(forecast_during_lead_time + capped_safety_stock)
        reorder_point = int(reorder_point_demand_buffer + capped_safety_stock)

        order_quantity = 0
        if current_stock < reorder_point:
            order_quantity = max(0, target_stock_level - current_stock)
            if order_quantity < min_order_quantity:
                order_quantity = min_order_quantity
            order_quantity = int(np.ceil(order_quantity / order_multiple)) * order_multiple

        estimated_holding_cost_30_days = target_stock_level * holding_cost_per_unit_per_day * 30
        estimated_ordering_cost = st.session_state.ORDERING_COST_PER_ORDER if order_quantity > 0 else 0

        indent_recommendations.append({
            "SKU_ID": sku_id,
            "Current_Stock": current_stock,
            "Forecasted_Demand_Lead_Time": forecast_during_lead_time,
            "Lead_Time_Days": lead_time_days,
            "SKU_Shelf_Life_Days": sku_shelf_life,
            "Calculated_Safety_Stock": safety_stock,
            "Capped_Safety_Stock": capped_safety_stock,
            "Target_Stock_Level": target_stock_level,
            "Reorder_Point": reorder_point,
            "Order_Quantity": order_quantity,
            "Min_Order_Quantity": min_order_quantity,
            "Order_Multiple": order_multiple,
            "Estimated_Holding_Cost_30_Days": estimated_holding_cost_30_days,
            "Estimated_Ordering_Cost": estimated_ordering_cost,
            "Recommendation_Date": current_date.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    return pd.DataFrame(indent_recommendations)

@st.cache_data
def calculate_bom_requirements(sku_indent_recommendations_df, bom_df, lead_times_df):
    component_requirements = {}
    current_date = datetime.now()

    if sku_indent_recommendations_df.empty:
        return pd.DataFrame()

    for _, row in sku_indent_recommendations_df.iterrows():
        sku_id = row['SKU_ID']
        sku_order_quantity = row['Order_Quantity']

        if sku_order_quantity > 0:
            components_for_sku = bom_df[bom_df['Parent_SKU_ID'] == sku_id]

            for _, comp_row in components_for_sku.iterrows():
                component_id = comp_row['Component_ID']
                qty_required_per_sku = comp_row['Quantity_Required']
                component_type = comp_row['Component_Type']
                shelf_life_days = comp_row['Shelf_Life_Days']

                total_component_qty = sku_order_quantity * qty_required_per_sku

                comp_lead_time_row = lead_times_df[(lead_times_df['Item_ID'] == component_id) & (lead_times_df['Item_Type'] == component_type)]
                if comp_lead_time_row.empty:
                    continue
                
                component_lead_time = comp_lead_time_row['Lead_Time_Days'].iloc[0]
                min_order_quantity = comp_lead_time_row['Min_Order_Quantity'].iloc[0]
                order_multiple = comp_lead_time_row['Order_Multiple'].iloc[0]

                adjusted_qty = total_component_qty
                if adjusted_qty < min_order_quantity:
                    adjusted_qty = min_order_quantity
                adjusted_qty = int(np.ceil(adjusted_qty / order_multiple)) * order_multiple
                
                shelf_life_critical_flag = False
                if shelf_life_days is not None and component_lead_time > shelf_life_days:
                    shelf_life_critical_flag = True

                earliest_order_date = (current_date + timedelta(days=row['Lead_Time_Days']) - timedelta(days=component_lead_time)).strftime("%Y-%m-%d")
                estimated_ordering_cost = st.session_state.ORDERING_COST_PER_ORDER

                if component_id not in component_requirements:
                    component_requirements[component_id] = {
                        "Component_ID": component_id,
                        "Total_Required_Quantity": 0,
                        "Component_Type": component_type,
                        "Lead_Time_Days": component_lead_time,
                        "Shelf_Life_Days": shelf_life_days,
                        "Shelf_Life_Critical": shelf_life_critical_flag,
                        "Min_Order_Quantity": min_order_quantity,
                        "Order_Multiple": order_multiple,
                        "Earliest_Order_Placement_Date": earliest_order_date,
                        "Estimated_Ordering_Cost": 0,
                        "Source_SKUs": []
                    }
                component_requirements[component_id]["Total_Required_Quantity"] += adjusted_qty
                component_requirements[component_id]["Estimated_Ordering_Cost"] += estimated_ordering_cost
                component_requirements[component_id]["Source_SKUs"].append(f"{sku_id} ({sku_order_quantity})")
    
    if not component_requirements:
        return pd.DataFrame()
        
    bom_indent_df = pd.DataFrame(list(component_requirements.values()))
    bom_indent_df['Source_SKUs'] = bom_indent_df['Source_SKUs'].apply(lambda x: ", ".join(x))
    return bom_indent_df

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Quantumflo: Demand Forecasting & Auto-Indent Solution")
st.markdown("A data-driven solution for supply chain planning using ML.")

# Sidebar for controls
st.sidebar.header("Application Settings")
run_on_sample_data = st.sidebar.button("🚀 Run Forecast on Sample Data")
run_on_uploaded_data = st.sidebar.button("🚀 Run Forecast on Uploaded Data")

st.sidebar.subheader("Configuration")
st.session_state.FORECAST_HORIZON_DAYS = st.sidebar.slider("Forecast Horizon (Days)", min_value=7, max_value=180, value=30)
st.session_state.SERVICE_LEVEL = st.sidebar.slider("Service Level", min_value=0.5, max_value=0.999, value=0.95, step=0.01)
model_to_use = st.sidebar.selectbox("Select ML Model", ('XGBoost', 'RandomForest'), index=0)

# File uploader section
st.sidebar.subheader("Upload Your Data (CSV)")
uploaded_files = {}
uploaded_files['sales_data'] = st.sidebar.file_uploader("Upload sales_data.csv", type="csv")
uploaded_files['inventory_data'] = st.sidebar.file_uploader("Upload inventory_data.csv", type="csv")
uploaded_files['promotions_data'] = st.sidebar.file_uploader("Upload promotions_data.csv", type="csv")
uploaded_files['external_factors_data'] = st.sidebar.file_uploader("Upload external_factors_data.csv", type="csv")
uploaded_files['lead_times_data'] = st.sidebar.file_uploader("Upload lead_times_data.csv", type="csv")
uploaded_files['bom_data'] = st.sidebar.file_uploader("Upload bom_data.csv", type="csv")

# Main execution logic
if run_on_sample_data or run_on_uploaded_data:
    st.info("Running the simulation. This may take a moment...")
    
    # Load data
    if run_on_sample_data:
        sales_df = generate_sales_data(st.session_state.NUM_SKUS, st.session_state.START_DATE, st.session_state.END_DATE)
        inventory_df = generate_inventory_data(sales_df, st.session_state.START_DATE)
        promotions_df = generate_promotion_data(sales_df)
        external_factors_df = generate_external_factors_data(st.session_state.START_DATE, st.session_state.END_DATE)
        lead_times_df = generate_lead_times_data(sales_df['SKU_ID'].unique())
        bom_df = generate_bom_data(sales_df['SKU_ID'].unique(), st.session_state.NUM_COMPONENTS_PER_SKU, lead_times_df)
        
        st.success("Using dummy data.")
        st.subheader("Generated Sample Data (first 5 rows)")
        st.dataframe(sales_df.head())
    else: # Run on uploaded data
        sales_df = load_data(uploaded_files.get('sales_data'), pd.DataFrame)
        inventory_df = load_data(uploaded_files.get('inventory_data'), pd.DataFrame)
        promotions_df = load_data(uploaded_files.get('promotions_data'), pd.DataFrame)
        external_factors_df = load_data(uploaded_files.get('external_factors_data'), pd.DataFrame)
        lead_times_df = load_data(uploaded_files.get('lead_times_data'), pd.DataFrame)
        bom_df = load_data(uploaded_files.get('bom_data'), pd.DataFrame)
        if sales_df.empty or inventory_df.empty or lead_times_df.empty or bom_df.empty:
            st.error("Missing required data files. Please upload all files and try again.")
            st.stop()

    # Process and forecast
    with st.spinner("Processing data and training models..."):
        processed_data_df = prepare_data_for_forecasting(sales_df, inventory_df, promotions_df, external_factors_df)
        if not processed_data_df.empty:
            trained_models, evaluation_metrics = train_forecast_model(processed_data_df, model_to_use)
            if trained_models:
                forecasted_demand_by_sku = predict_demand(trained_models, processed_data_df, st.session_state.FORECAST_HORIZON_DAYS, external_factors_df)
                if forecasted_demand_by_sku:
                    st.success("Forecasting complete!")
                    st.subheader("Model Evaluation Metrics (RMSE and MAE)")
                    st.json(evaluation_metrics)
                    
                    st.subheader("Future Demand Forecasts")
                    all_forecasts_df = pd.concat(forecasted_demand_by_sku.values(), ignore_index=True)
                    st.dataframe(all_forecasts_df)

                    st.subheader("SKU-Level Auto-Indent Recommendations")
                    sku_indent_recommendations_df = calculate_auto_indent(
                        forecasted_demand_by_sku, 
                        inventory_df, 
                        lead_times_df, 
                        evaluation_metrics, 
                        st.session_state.SERVICE_LEVEL, 
                        st.session_state.HOLDING_COST_PER_UNIT_PER_DAY
                    )
                    st.dataframe(sku_indent_recommendations_df)
                    
                    st.subheader("BOM-Level Component Requirements")
                    bom_requirements_df = calculate_bom_requirements(sku_indent_recommendations_df, bom_df, lead_times_df)
                    st.dataframe(bom_requirements_df)
                else:
                    st.error("Failed to generate demand forecasts.")
            else:
                st.error("No models were trained. Please check your data and try again.")
        else:
            st.error("Data processing failed.")

# --- Helper function for download links ---
def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

st.sidebar.subheader("Download Sample Data Templates")
st.sidebar.markdown("Click to download and populate with your own data.")
if st.sidebar.button("Download Sample Data"):
    dummy_sales_df = generate_sales_data(st.session_state.NUM_SKUS, st.session_state.START_DATE, st.session_state.END_DATE)
    dummy_inventory_df = generate_inventory_data(dummy_sales_df, st.session_state.START_DATE)
    dummy_promotions_df = generate_promotion_data(dummy_sales_df)
    dummy_external_factors_df = generate_external_factors_data(st.session_state.START_DATE, st.session_state.END_DATE)
    dummy_lead_times_df = generate_lead_times_data(dummy_sales_df['SKU_ID'].unique())
    dummy_bom_df = generate_bom_data(dummy_sales_df['SKU_ID'].unique(), st.session_state.NUM_COMPONENTS_PER_SKU, dummy_lead_times_df)

    st.sidebar.markdown(get_table_download_link(dummy_sales_df, 'sales_data.csv', 'Download sales_data.csv'), unsafe_allow_html=True)
    st.sidebar.markdown(get_table_download_link(dummy_inventory_df, 'inventory_data.csv', 'Download inventory_data.csv'), unsafe_allow_html=True)
    st.sidebar.markdown(get_table_download_link(dummy_promotions_df, 'promotions_data.csv', 'Download promotions_data.csv'), unsafe_allow_html=True)
    st.sidebar.markdown(get_table_download_link(dummy_external_factors_df, 'external_factors_data.csv', 'Download external_factors_data.csv'), unsafe_allow_html=True)
    st.sidebar.markdown(get_table_download_link(dummy_lead_times_df, 'lead_times_data.csv', 'Download lead_times_data.csv'), unsafe_allow_html=True)
    st.sidebar.markdown(get_table_download_link(dummy_bom_df, 'bom_data.csv', 'Download bom_data.csv'), unsafe_allow_html=True)
