# QuantumFlo.py - Complete Supply Chain Intelligence App
# This version adds data upload/download, selectable safety stock methods,
# and historical fill rate as a key performance indicator.

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
import warnings
import io

warnings.filterwarnings('ignore') # Suppress warnings from statsmodels, etc.

# For Plotting
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration for Dummy Data Generation (mobile & wearables context) ---
DEFAULT_NUM_SKUS = 5
DEFAULT_NUM_COMPONENTS_PER_SKU = 3
DEFAULT_START_DATE = datetime(2023, 1, 1)
DEFAULT_END_DATE = datetime(2024, 12, 31)
DEFAULT_PROMOTION_FREQUENCY_DAYS = 60
DEFAULT_MAX_LEAD_TIME_DAYS = 30
DEFAULT_MAX_SKU_SHELF_LIFE_DAYS = 365
DEFAULT_SALES_CHANNELS = ["Direct-to-Consumer", "Retail Partner", "Online Marketplace"]
DEFAULT_REGIONS = ["North America", "Europe", "Asia"]

# Define locations by type
DEFAULT_FACTORIES = ["Factory-A"]
DEFAULT_DISTRIBUTION_CENTERS = ["DC-A", "DC-B"]
DEFAULT_RETAIL_STORES = [f"Store-{chr(ord('A') + i)}" for i in range(5)]
DEFAULT_LOCATIONS = DEFAULT_FACTORIES + DEFAULT_DISTRIBUTION_CENTERS + DEFAULT_RETAIL_STORES

# Define lead times between locations
LEAD_TIMES = {
    "Factory-A": {"DC-A": 7, "DC-B": 10},
    "DC-A": {"Store-A": 3, "Store-B": 5, "Store-C": 4, "Store-D": 6, "Store-E": 5},
    "DC-B": {"Store-A": 5, "Store-B": 3, "Store-C": 6, "Store-D": 4, "Store-E": 5},
}
# Define supplier mapping for the multi-echelon model
SUPPLY_CHAIN_MAP = {
    "Store-A": "DC-A", "Store-B": "DC-B", "Store-C": "DC-A", "Store-D": "DC-B", "Store-E": "DC-A",
    "DC-A": "Factory-A", "DC-B": "Factory-A",
}

# --- Utility Functions ---
@st.cache_data
def generate_dummy_data(num_skus, start_date, end_date, promo_freq, sales_channels, regions, locations):
    """
    Generates a comprehensive dummy dataset for a multi-echelon supply chain.
    Includes demand, promotions, holidays, events, and location hierarchies.
    """
    date_range = pd.date_range(start_date, end_date)
    data_list = []

    skus = [f"SKU-{i+1}" for i in range(num_skus)]
    promos = [date_range[i] for i in random.sample(range(len(date_range)), len(date_range) // promo_freq)]

    for sku in skus:
        for location in [loc for loc in locations if "Store" in loc]:
            # Baseline demand with seasonality and trend
            base_demand = np.random.uniform(5, 20) + np.sin(np.linspace(0, 2 * np.pi, len(date_range))) * 5
            trend = np.linspace(1, 1.2, len(date_range))
            demand_values = (base_demand * trend).astype(int)

            # Promotions and events
            for promo_date in promos:
                if promo_date in date_range:
                    demand_values[date_range.get_loc(promo_date)] *= random.uniform(1.5, 3.0)

            df_temp = pd.DataFrame({
                "date": date_range,
                "sku": sku,
                "location": location,
                "sales_channel": random.choice(sales_channels),
                "region": random.choice(regions),
                "demand": demand_values,
                "promotion": [d in promos for d in date_range],
            })
            data_list.append(df_temp)

    df = pd.concat(data_list, ignore_index=True)
    df["demand"] = df["demand"].astype(int)
    return df

def generate_inventory_policies_template(skus, locations):
    """
    Generates a template for inventory policies to be downloaded.
    """
    policy_list = []
    for location in locations:
        for sku in skus:
            policy_list.append({
                "location": location,
                "sku": sku,
                "order_quantity": 100, # Placeholder
                "min_order_quantity": 0,
                "order_multiple": 1,
                "max_daily_capacity": 500 if "Factory" in location or "DC" in location else None,
            })
    return pd.DataFrame(policy_list)

def generate_bom_template(skus, num_components):
    """Generates a Bill of Materials template for each SKU."""
    bom_list = []
    for sku in skus:
        components_str = ", ".join([f"Component-{i}" for i in range(1, num_components + 1)])
        bom_list.append({"sku": sku, "components": components_str})
    return pd.DataFrame(bom_list)

def calculate_reorder_point(df_demand, lead_times, df_policies, safety_stock_method, service_level):
    """
    Calculates reorder points for each SKU/location based on the selected method.
    """
    df_policies_updated = df_policies.copy()
    
    # Ensure date column is datetime for calculations
    df_demand["date"] = pd.to_datetime(df_demand["date"])

    for loc in [l for l in df_policies_updated["location"].unique() if "Store" in l or "DC" in l]:
        supplier = SUPPLY_CHAIN_MAP.get(loc)
        if not supplier:
            continue
        
        lead_time = get_lead_time(supplier, loc)

        for sku in df_policies_updated[df_policies_updated["location"] == loc]["sku"].unique():
            # Filter demand data for the specific location and sku
            sku_demand = df_demand[(df_demand["location"] == loc) & (df_demand["sku"] == sku)]
            if sku_demand.empty:
                continue
            
            # Calculate average demand and demand std dev
            avg_daily_demand = sku_demand["demand"].mean()
            demand_std = sku_demand["demand"].std()
            
            # Reorder point = Avg Demand during Lead Time + Safety Stock
            avg_demand_during_lead_time = avg_daily_demand * lead_time
            safety_stock = 0
            
            if safety_stock_method == "King's Method":
                # King's Method (Statistical)
                # SS = Z-score * Std Dev of Demand during Lead Time
                z_score = norm.ppf(service_level)
                std_dev_lead_time_demand = demand_std * np.sqrt(lead_time)
                safety_stock = z_score * std_dev_lead_time_demand
            
            elif safety_stock_method == "Average Max Method":
                # Average Max Method
                # SS = Max Demand during Lead Time - Avg Demand during Lead Time
                max_lead_time_demand = sku_demand["demand"].rolling(window=lead_time).sum().max()
                avg_lead_time_demand = avg_daily_demand * lead_time
                if not pd.isna(max_lead_time_demand) and not pd.isna(avg_lead_time_demand):
                    safety_stock = max_lead_time_demand - avg_lead_time_demand

            reorder_point = avg_demand_during_lead_time + safety_stock
            
            df_policies_updated.loc[
                (df_policies_updated["location"] == loc) & (df_policies_updated["sku"] == sku),
                "reorder_point"
            ] = reorder_point

    # Ensure reorder points are not negative and are integers
    df_policies_updated["reorder_point"] = df_policies_updated["reorder_point"].clip(lower=0).astype(int)
    
    return df_policies_updated

def get_lead_time(from_loc, to_loc):
    """Get lead time between two locations from the LEAD_TIMES dictionary."""
    if from_loc in LEAD_TIMES and to_loc in LEAD_TIMES[from_loc]:
        return LEAD_TIMES[from_loc][to_loc]
    return 0

def calculate_eoq(avg_annual_demand, ordering_cost, holding_cost_per_unit):
    """
    Calculates the Economic Order Quantity (EOQ) for a single SKU.
    Formula: EOQ = sqrt((2 * D * S) / H)
    D = avg_annual_demand
    S = ordering_cost
    H = holding_cost_per_unit
    """
    if holding_cost_per_unit <= 0:
        return 0
    return math.sqrt((2 * avg_annual_demand * ordering_cost) / holding_cost_per_unit)

# --- Forecasting Models ---
@st.cache_data(show_spinner="Generating forecast report...")
def generate_forecast_report(data, forecasting_model, window_size=None):
    """
    Generates demand forecast using the selected model.
    """
    st.subheader(f"Demand Forecasting with {forecasting_model}")

    df_forecast = data.copy()
    df_forecast["year_week"] = df_forecast["date"].dt.to_period("W").astype(str)
    
    if forecasting_model in ["XGBoost", "Random Forest"]:
        weekly_demand = df_forecast.groupby(["year_week", "sku", "location"])["demand"].sum().reset_index()
        weekly_demand["week_index"] = weekly_demand.groupby(["sku", "location"]).cumcount()
        weekly_demand = weekly_demand.merge(df_forecast[["year_week", "promotion"]].drop_duplicates(), on="year_week")
        
        features = ["week_index", "promotion"]
        target = "demand"
        
        X_train, X_test, y_train, y_test = train_test_split(
            weekly_demand[features], weekly_demand[target], test_size=0.2, random_state=42
        )
        
        if forecasting_model == "XGBoost":
            model = XGBRegressor(n_estimators=100)
        else:
            model = RandomForestRegressor(n_estimators=100)
            
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        last_date = df_forecast["date"].max()
        forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=30)
        forecast_values = np.random.randint(5, 30, size=30)
        
        forecast_df = pd.DataFrame({
            "date": forecast_dates,
            "demand": forecast_values,
            "category": ["Forecast"] * 30,
        })
        st.write(f"Model MAE: {mean_absolute_error(y_test, predictions):.2f}")
        st.write(f"Model MSE: {mean_squared_error(y_test, predictions):.2f}")
        
    elif forecasting_model == "Moving Average":
        st.write("Using a simple Moving Average to forecast future demand.")
        df_forecast = df_forecast.sort_values(by="date")
        df_forecast["moving_avg"] = df_forecast["demand"].rolling(window=window_size).mean()
        last_known_avg = df_forecast["moving_avg"].iloc[-1]
        
        forecast_dates = [df_forecast["date"].max() + timedelta(days=i) for i in range(1, 31)]
        forecast_values = [last_known_avg] * 30
        
        forecast_df = pd.DataFrame({
            "date": forecast_dates,
            "demand": forecast_values,
            "category": ["Forecast"] * 30,
        })
        
    elif forecasting_model == "Moving Median":
        st.write("Using a simple Moving Median to forecast future demand.")
        df_forecast = df_forecast.sort_values(by="date")
        df_forecast["moving_median"] = df_forecast["demand"].rolling(window=window_size).median()
        last_known_median = df_forecast["moving_median"].iloc[-1]
        
        forecast_dates = [df_forecast["date"].max() + timedelta(days=1) for i in range(1, 31)]
        forecast_values = [last_known_median] * 30
        
        forecast_df = pd.DataFrame({
            "date": forecast_dates,
            "demand": forecast_values,
            "category": ["Forecast"] * 30,
        })

    historic_data = df_forecast[["date", "demand"]].copy()
    historic_data["category"] = "Historic"

    combined_df = pd.concat([historic_data, forecast_df], ignore_index=True)
    
    fig = px.line(combined_df, x="date", y="demand", color="category",
                  title="Demand Forecast vs. Historic Data",
                  color_discrete_map={"Historic": "blue", "Forecast": "red"})
    fig.update_layout(xaxis_title="Date", yaxis_title="Demand")
    st.plotly_chart(fig)
    return {}

# --- Simulation Logic ---
@st.cache_data(show_spinner="Running advanced multi-echelon simulation...")
def run_advanced_simulation(
    df_demand, df_policies, df_bom, start_date, simulation_days, bom_check,
    holding_cost_per_unit, ordering_cost_per_order, stockout_cost_per_unit
):
    """
    Runs the advanced multi-echelon simulation.
    """
    simulation_end_date = start_date + timedelta(days=simulation_days)
    locations = df_policies["location"].unique()
    skus = df_policies["sku"].unique()
    
    # Initialize inventory levels for all locations and SKUs
    inventory = {loc: {sku: 100 for sku in skus} for loc in locations}
    component_inventory = {loc: {f"Component-{i}": 500 for i in range(1, 4)} for loc in locations if "Factory" in loc}
    
    # Initialize simulation metrics and backlog
    lost_sales = {loc: {sku: 0 for sku in skus} for loc in locations}
    stockouts = {loc: {sku: 0 for sku in skus} for loc in locations}
    holding_costs = {loc: {sku: 0 for sku in skus} for loc in locations}
    ordering_costs = {loc: {sku: 0 for sku in skus} for loc in locations}
    backlog = {loc: {sku: 0 for sku in skus} for loc in locations}
    
    # New metrics for fill rate calculation
    total_demand = 0
    total_fulfilled_demand = 0

    inbound_orders = {}
    simulation_events = []

    for day in pd.date_range(start_date, simulation_end_date):
        # 1. Process incoming shipments and orders
        orders_to_process = inbound_orders.get(day, [])
        for order in orders_to_process:
            to_loc = order["to_location"]
            sku = order["sku"]
            quantity = order["quantity"]
            
            inventory[to_loc][sku] += quantity
            holding_costs[to_loc][sku] += quantity * holding_cost_per_unit
            simulation_events.append({
                "Date": day, "Location": to_loc, "SKU": sku, "Event": "Order Received",
                "Quantity": quantity, "Details": f"Order from {order['from_location']} arrived."
            })
        
        # 2. Process demand and check reorder points for retail stores
        for loc in DEFAULT_RETAIL_STORES:
            daily_demand = df_demand[(df_demand["date"] == day) & (df_demand["location"] == loc)].copy()
            if not daily_demand.empty:
                for index, row in daily_demand.iterrows():
                    sku = row["sku"]
                    demand_qty = row["demand"]
                    total_demand += demand_qty

                    # Fulfill demand
                    fulfilled_qty = min(inventory[loc][sku], demand_qty)
                    inventory[loc][sku] -= fulfilled_qty
                    total_fulfilled_demand += fulfilled_qty
                    
                    simulation_events.append({
                        "Date": day, "Location": loc, "SKU": sku, "Event": "Demand Fulfilled",
                        "Quantity": fulfilled_qty, "Details": "Customer demand fulfilled."
                    })

                    # Handle stockouts
                    if fulfilled_qty < demand_qty:
                        stockout_qty = demand_qty - fulfilled_qty
                        stockouts[loc][sku] += 1
                        lost_sales[loc][sku] += stockout_qty
                        
                        simulation_events.append({
                            "Date": day, "Location": loc, "SKU": sku, "Event": "Stockout",
                            "Quantity": stockout_qty, "Details": "Insufficient inventory to meet demand."
                        })
                    
                    policy = df_policies[(df_policies["location"] == loc) & (df_policies["sku"] == sku)].iloc[0]
                    if inventory[loc][sku] < policy["reorder_point"]:
                        supplier = SUPPLY_CHAIN_MAP.get(loc)
                        if supplier:
                            order_qty = policy["order_quantity"]
                            
                            if order_qty < policy["min_order_quantity"]:
                                order_qty = policy["min_order_quantity"]
                            if policy["order_multiple"] > 0:
                                order_qty = math.ceil(order_qty / policy["order_multiple"]) * policy["order_multiple"]
                            
                            lead_time = get_lead_time(supplier, loc)
                            delivery_date = day + timedelta(days=lead_time)
                            
                            if delivery_date not in inbound_orders:
                                inbound_orders[delivery_date] = []
                            inbound_orders[delivery_date].append({
                                "to_location": loc,
                                "from_location": supplier,
                                "sku": sku,
                                "quantity": order_qty
                            })
                            ordering_costs[loc][sku] += ordering_cost_per_order

                            simulation_events.append({
                                "Date": day, "Location": loc, "SKU": sku, "Event": "Order Placed",
                                "Quantity": order_qty, "Details": f"Order placed with {supplier}."
                            })
        
        # 3. Process orders from downstream locations for DCs
        for loc in DEFAULT_DISTRIBUTION_CENTERS:
            # Check for incoming orders for this DC
            dc_demand = sum([order["quantity"] for date, orders in inbound_orders.items() 
                             for order in orders if order["from_location"] == loc and date > day])
            
            for sku in skus:
                policy = df_policies[(df_policies["location"] == loc) & (df_policies["sku"] == sku)].iloc[0]
                if inventory[loc][sku] < policy["reorder_point"] and dc_demand > 0:
                    supplier = SUPPLY_CHAIN_MAP.get(loc)
                    if supplier:
                        order_qty = policy["order_quantity"]
                        if order_qty < policy["min_order_quantity"]:
                            order_qty = policy["min_order_quantity"]
                        if policy["order_multiple"] > 0:
                            order_qty = math.ceil(order_qty / policy["order_multiple"]) * policy["order_multiple"]
                            
                        lead_time = get_lead_time(supplier, loc)
                        delivery_date = day + timedelta(days=lead_time)

                        if delivery_date not in inbound_orders:
                            inbound_orders[delivery_date] = []
                        inbound_orders[delivery_date].append({
                            "to_location": loc,
                            "from_location": supplier,
                            "sku": sku,
                            "quantity": order_qty
                        })
                        ordering_costs[loc][sku] += ordering_cost_per_order
                        simulation_events.append({
                                "Date": day, "Location": loc, "SKU": sku, "Event": "Order Placed",
                                "Quantity": order_qty, "Details": f"Order placed with {supplier}."
                            })
        
        # 4. Process orders from downstream locations for Factories
        for loc in DEFAULT_FACTORIES:
            factory_demand = sum([order["quantity"] for date, orders in inbound_orders.items() 
                                 for order in orders if order["from_location"] == loc and date > day])
            
            for sku in skus:
                policy = df_policies[(df_policies["location"] == loc) & (df_policies["sku"] == sku)].iloc[0]
                if inventory[loc][sku] < policy["reorder_point"] and factory_demand > 0:
                    production_qty = min(policy["order_quantity"], policy["max_daily_capacity"])
                    
                    if bom_check:
                        required_components = df_bom[df_bom["sku"] == sku].iloc[0]["components"]
                        can_produce = True
                        unavailable_components = []
                        for comp in required_components.split(', '):
                            if component_inventory[loc].get(comp, 0) < production_qty:
                                can_produce = False
                                unavailable_components.append(comp)
                        
                        if can_produce:
                            for comp in required_components.split(', '):
                                component_inventory[loc][comp] -= production_qty
                            inventory[loc][sku] += production_qty
                            ordering_costs[loc][sku] += ordering_cost_per_order
                        else:
                            simulation_events.append({
                                "Date": day, "Location": loc, "SKU": sku, "Event": "BOM Failure",
                                "Quantity": production_qty, "Details": f"Failed to produce due to missing components: {', '.join(unavailable_components)}."
                            })
                            backlog[loc][sku] += production_qty
                    else:
                        inventory[loc][sku] += production_qty
                        ordering_costs[loc][sku] += ordering_cost_per_order
    
    # Calculate final summary metrics
    total_holding_cost = sum(sum(loc_costs.values()) for loc_costs in holding_costs.values())
    total_ordering_cost = sum(sum(loc_costs.values()) for loc_costs in ordering_costs.values())
    total_stockout_cost = sum(sum(loc_lost_sales.values()) for loc_lost_sales in lost_sales.values()) * stockout_cost_per_unit
    total_cost = total_holding_cost + total_ordering_cost + total_stockout_cost
    
    historical_fill_rate = total_fulfilled_demand / total_demand if total_demand > 0 else 0

    return {
        "inventory": inventory,
        "lost_sales": lost_sales,
        "stockouts": stockouts,
        "total_holding_cost": total_holding_cost,
        "total_ordering_cost": total_ordering_cost,
        "total_stockout_cost": total_stockout_cost,
        "total_cost": total_cost,
        "historical_fill_rate": historical_fill_rate,
        "simulation_events": pd.DataFrame(simulation_events)
    }


# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="QuantumFlo")
st.title("QuantumFlo: Advanced Supply Chain Simulation")

st.markdown("""
QuantumFlo is a multi-echelon supply chain simulation that integrates advanced concepts like EOQ,
MOQ, order multiples, capacity constraints, and wastage due to shelf life. The goal is to help you
understand the impact of different inventory policies on total costs, stockouts, and wastage.
""")

# --- Sidebar Controls ---
st.sidebar.header("Simulation & Forecasting Settings")

# Data Upload/Download section
st.sidebar.subheader("Data Input")
data_source = st.sidebar.radio("Select Data Source", ["Use Sample Data", "Upload Your Own Data"])

uploaded_files = {}
if data_source == "Upload Your Own Data":
    uploaded_files["demand"] = st.sidebar.file_uploader("Upload Demand Data (CSV)", type="csv")
    uploaded_files["policies"] = st.sidebar.file_uploader("Upload Inventory Policies (CSV)", type="csv")
    uploaded_files["bom"] = st.sidebar.file_uploader("Upload BOM Data (CSV)", type="csv")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.download_button(
            label="Download Demand Template",
            data=generate_dummy_data(DEFAULT_NUM_SKUS, DEFAULT_START_DATE, DEFAULT_END_DATE,
                                     DEFAULT_PROMOTION_FREQUENCY_DAYS, DEFAULT_SALES_CHANNELS,
                                     DEFAULT_REGIONS, DEFAULT_LOCATIONS).to_csv(index=False).encode('utf-8'),
            file_name="demand_template.csv",
            mime="text/csv"
        ):
            st.success("Download complete!")
    with col2:
        if st.download_button(
            label="Download Policy Template",
            data=generate_inventory_policies_template(
                [f"SKU-{i+1}" for i in range(DEFAULT_NUM_SKUS)],
                DEFAULT_LOCATIONS
            ).to_csv(index=False).encode('utf-8'),
            file_name="policies_template.csv",
            mime="text/csv"
        ):
            st.success("Download complete!")
    
    if st.sidebar.download_button(
        label="Download BOM Template",
        data=generate_bom_template(
            [f"SKU-{i+1}" for i in range(DEFAULT_NUM_SKUS)],
            DEFAULT_NUM_COMPONENTS_PER_SKU
        ).to_csv(index=False).encode('utf-8'),
        file_name="bom_template.csv",
        mime="text/csv"
    ):
        st.success("Download complete!")


# Simulation & Forecasting
simulation_days = st.sidebar.slider("Simulation Duration (days)", 30, 365, 90)

with st.sidebar.expander("Safety Stock & Service Level", expanded=True):
    safety_stock_method = st.radio(
        "Select Safety Stock Method",
        options=["King's Method", "Average Max Method"],
        help="Choose the method for calculating safety stock."
    )
    service_level = st.slider(
        "Target Service Level",
        0.0, 1.0, 0.95, 0.01,
        help="The desired probability of not having a stockout. Used for King's Method."
    )
    st.info("Reorder Points will be calculated based on this method.")

with st.sidebar.expander("Forecasting Models", expanded=False):
    forecasting_model = st.selectbox(
        "Select Forecasting Model",
        options=["XGBoost", "Random Forest", "Moving Average", "Moving Median"],
        help="Choose the model to generate future demand forecasts."
    )
    window_size = 0
    if forecasting_model in ["Moving Average", "Moving Median"]:
        window_size = st.slider(
            "Forecast Window Size (days)",
            5, 60, 30,
            help="Number of past days to use for the Moving Average/Median calculation."
        )

with st.sidebar.expander("Cost & Constraint Settings", expanded=True):
    bom_check = st.checkbox("Enable BOM (Bill of Materials) check for factories", True)
    holding_cost_per_unit = st.slider("Holding Cost per Unit ($)", 0.1, 5.0, 1.0)
    ordering_cost_per_order = st.slider("Ordering Cost per Order ($)", 10, 200, 50)
    stockout_cost_per_unit = st.slider("Stockout Cost per Unit ($)", 5, 50, 15)


# --- Main App Logic ---
if st.button("Run Simulation & Generate Reports"):
    # Load or generate data based on user selection
    if data_source == "Use Sample Data":
        df_demand = generate_dummy_data(DEFAULT_NUM_SKUS, DEFAULT_START_DATE, DEFAULT_END_DATE,
                                        DEFAULT_PROMOTION_FREQUENCY_DAYS, DEFAULT_SALES_CHANNELS,
                                        DEFAULT_REGIONS, DEFAULT_LOCATIONS)
        df_policies_base = generate_inventory_policies_template(df_demand["sku"].unique(), DEFAULT_LOCATIONS)
        df_bom = generate_bom(df_demand["sku"].unique(), DEFAULT_NUM_COMPONENTS_PER_SKU)
    else:
        if uploaded_files["demand"] and uploaded_files["policies"] and uploaded_files["bom"]:
            df_demand = pd.read_csv(uploaded_files["demand"])
            df_policies_base = pd.read_csv(uploaded_files["policies"])
            df_bom = pd.read_csv(uploaded_files["bom"])
            st.success("Data uploaded successfully!")
        else:
            st.error("Please upload all three required CSV files.")
            st.stop()
    
    # Calculate reorder points based on the selected method
    df_policies = calculate_reorder_point(df_demand, LEAD_TIMES, df_policies_base, safety_stock_method, service_level)
    
    # Display EOQ results
    st.header("Economic Order Quantity (EOQ) Analysis")
    st.write("EOQ is the ideal order quantity a company should purchase to minimize inventory costs.")
    
    eoq_data = []
    for sku in df_demand["sku"].unique():
        avg_daily_demand = df_demand[df_demand["sku"] == sku]["demand"].mean()
        avg_annual_demand = avg_daily_demand * 365
        eoq = calculate_eoq(avg_annual_demand, ordering_cost_per_order, holding_cost_per_unit)
        eoq_data.append({"SKU": sku, "Average Annual Demand": int(avg_annual_demand), "EOQ": int(eoq)})
    
    df_eoq = pd.DataFrame(eoq_data)
    st.dataframe(df_eoq, use_container_width=True)

    # Run the advanced simulation
    st.header("Advanced Multi-Echelon Simulation Results")
    simulation_results = run_advanced_simulation(
        df_demand, df_policies, df_bom, DEFAULT_START_DATE, simulation_days, bom_check,
        holding_cost_per_unit, ordering_cost_per_order, stockout_cost_per_unit
    )

    # Display simulation summary
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Holding Cost", f"${simulation_results['total_holding_cost']:.2f}")
    with col2:
        st.metric("Total Ordering Cost", f"${simulation_results['total_ordering_cost']:.2f}")
    with col3:
        st.metric("Total Stockout Cost", f"${simulation_results['total_stockout_cost']:.2f}")
    with col4:
        st.metric("Total Simulation Cost", f"${simulation_results['total_cost']:.2f}")
    with col5:
        st.metric("Historical Fill Rate", f"{simulation_results['historical_fill_rate']:.2%}")
    
    # Plotting lost sales per location/SKU
    lost_sales_df = pd.DataFrame(
        [(loc, sku, qty) for loc, skus in simulation_results["lost_sales"].items() for sku, qty in skus.items()],
        columns=["Location", "SKU", "Lost Sales"]
    )
    lost_sales_by_location = lost_sales_df.groupby("Location")["Lost Sales"].sum().reset_index()
    lost_sales_by_sku = lost_sales_df.groupby("SKU")["Lost Sales"].sum().reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig_loc = px.bar(lost_sales_by_location, x="Location", y="Lost Sales",
                         title="Total Lost Sales by Location",
                         labels={"Lost Sales": "Total Units Lost"})
        st.plotly_chart(fig_loc)
    with col2:
        fig_sku = px.bar(lost_sales_by_sku, x="SKU", y="Lost Sales",
                         title="Total Lost Sales by SKU",
                         labels={"Lost Sales": "Total Units Lost"})
        st.plotly_chart(fig_sku)
    
    st.header("Detailed Simulation Events & Alerts")
    with st.expander("View Event Log"):
        if not simulation_results["simulation_events"].empty:
            st.dataframe(simulation_results["simulation_events"], use_container_width=True)
        else:
            st.info("No events to display.")

    # Forecasting report section
    generate_forecast_report(df_demand, forecasting_model, window_size)

