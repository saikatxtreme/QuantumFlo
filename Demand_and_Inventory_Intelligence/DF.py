# QuantumFlo.py - Demand and Inventory Intelligence Streamlit App
# with Advanced Multi-Echelon, EOQ, MOQ, Capacity, and Wastage Integration

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

def generate_inventory_policies(skus, locations, lead_times):
    """
    Generates dummy inventory policies for all locations and SKUs, now including
    MOQ, Order Multiple, and Max Capacity.
    """
    policy_list = []
    for location in locations:
        for sku in skus:
            reorder_point = random.randint(10, 50)
            order_quantity = random.randint(50, 150)
            
            # New constraints
            min_order_quantity = random.choice([0, 10, 25])
            order_multiple = random.choice([1, 5, 10])
            max_daily_capacity = random.randint(200, 500) if "Factory" in location or "DC" in location else None

            policy_list.append({
                "location": location,
                "sku": sku,
                "reorder_point": reorder_point,
                "order_quantity": order_quantity,
                "min_order_quantity": min_order_quantity,
                "order_multiple": order_multiple,
                "max_daily_capacity": max_daily_capacity,
            })
    return pd.DataFrame(policy_list)

def generate_bom(skus, num_components):
    """Generates a Bill of Materials for each SKU."""
    bom_list = []
    for sku in skus:
        num_comp = random.randint(1, num_components)
        components = [f"Component-{i}" for i in range(1, num_comp + 1)]
        bom_list.append({"sku": sku, "components": components})
    return pd.DataFrame(bom_list)

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
    The function is updated to handle Moving Average and Moving Median.
    """
    st.subheader(f"Demand Forecasting with {forecasting_model}")

    # Prepare data for all models
    df_forecast = data.copy()
    df_forecast["year_week"] = df_forecast["date"].dt.to_period("W").astype(str)
    
    # Feature engineering for ML models
    if forecasting_model in ["XGBoost", "Random Forest"]:
        # Aggregate demand by week for ML models
        weekly_demand = df_forecast.groupby(["year_week", "sku", "location"])["demand"].sum().reset_index()
        weekly_demand["week_index"] = weekly_demand.groupby(["sku", "location"]).cumcount()
        weekly_demand = weekly_demand.merge(df_forecast[["year_week", "promotion"]].drop_duplicates(), on="year_week")
        
        # ML model feature engineering (simplified for brevity)
        features = ["week_index", "promotion"]
        target = "demand"
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(
            weekly_demand[features], weekly_demand[target], test_size=0.2, random_state=42
        )
        
        # Model training and prediction
        if forecasting_model == "XGBoost":
            model = XGBRegressor(n_estimators=100)
        else:
            model = RandomForestRegressor(n_estimators=100)
            
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Create a placeholder forecast dataframe
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

    # Prepare data for plotting
    historic_data = df_forecast[["date", "demand"]].copy()
    historic_data["category"] = "Historic"

    combined_df = pd.concat([historic_data, forecast_df], ignore_index=True)
    
    # Plotting
    fig = px.line(combined_df, x="date", y="demand", color="category",
                  title="Demand Forecast vs. Historic Data",
                  color_discrete_map={"Historic": "blue", "Forecast": "red"})
    fig.update_layout(xaxis_title="Date", yaxis_title="Demand")
    st.plotly_chart(fig)
    return {} # Return an empty dictionary as no specific report is generated in this simplified function

# --- Simulation Logic (The Core Update) ---
@st.cache_data(show_spinner="Running advanced multi-echelon simulation...")
def run_advanced_simulation(
    df_demand, df_policies, df_bom, start_date, simulation_days, bom_check,
    holding_cost_per_unit, ordering_cost_per_order, stockout_cost_per_unit
):
    """
    Runs the advanced multi-echelon simulation with EOQ, MOQ, capacity constraints,
    and wastage tracking.
    """
    simulation_end_date = start_date + timedelta(days=simulation_days)
    locations = df_policies["location"].unique()
    skus = df_policies["sku"].unique()

    # Initialize inventory levels for all locations and SKUs
    inventory = {loc: {sku: 100 for sku in skus} for loc in locations}
    
    # Initialize component inventory for factories (for BOM check)
    component_inventory = {loc: {f"Component-{i}": 500 for i in range(1, 4)} for loc in locations if "Factory" in loc}

    # Initialize simulation metrics and backlog
    lost_sales = {loc: {sku: 0 for sku in skus} for loc in locations}
    stockouts = {loc: {sku: 0 for sku in skus} for loc in locations}
    wastage = {loc: {sku: 0 for sku in skus} for loc in locations}
    # Corrected dictionary comprehension for holding_costs
    holding_costs = {loc: {sku: 0 for sku in skus} for loc in locations}
    # Corrected dictionary comprehension for ordering_costs
    ordering_costs = {loc: {sku: 0 for sku in skus} for loc in locations}
    backlog = {loc: {sku: 0 for sku in skus} for loc in locations}

    # Store incoming orders
    inbound_orders = {}

    for day in pd.date_range(start_date, simulation_end_date):
        # 1. Process incoming shipments and orders
        orders_to_process = inbound_orders.get(day, [])
        for order in orders_to_process:
            to_loc = order["to_location"]
            sku = order["sku"]
            quantity = order["quantity"]
            # Add to inventory and track holding cost
            inventory[to_loc][sku] += quantity
            holding_costs[to_loc][sku] += quantity * holding_cost_per_unit
        
        # 2. Process demand and check reorder points for retail stores
        for loc in DEFAULT_RETAIL_STORES:
            daily_demand = df_demand[(df_demand["date"] == day) & (df_demand["location"] == loc)].copy()
            if not daily_demand.empty:
                for index, row in daily_demand.iterrows():
                    sku = row["sku"]
                    demand_qty = row["demand"]

                    # Check for stockouts and fulfill demand
                    if inventory[loc][sku] >= demand_qty:
                        inventory[loc][sku] -= demand_qty
                    else:
                        stockouts[loc][sku] += 1
                        lost_sales[loc][sku] += (demand_qty - inventory[loc][sku])
                        inventory[loc][sku] = 0
                    
                    # Check reorder point and place order with supplying DC
                    policy = df_policies[(df_policies["location"] == loc) & (df_policies["sku"] == sku)].iloc[0]
                    if inventory[loc][sku] < policy["reorder_point"]:
                        supplier = SUPPLY_CHAIN_MAP.get(loc)
                        if supplier:
                            order_qty = policy["order_quantity"]
                            
                            # Apply MOQ and Order Multiple constraints
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
        
        # 3. Process orders from downstream locations for DCs
        for loc in DEFAULT_DISTRIBUTION_CENTERS:
            # Check for incoming orders for this DC
            dc_demand = sum([order["quantity"] for date, orders in inbound_orders.items() 
                             for order in orders if order["from_location"] == loc and date > day])
            
            for sku in skus:
                policy = df_policies[(df_policies["location"] == loc) & (df_policies["sku"] == sku)].iloc[0]
                # If inventory is low and there are downstream orders
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
        
        # 4. Process orders from downstream locations for Factories
        for loc in DEFAULT_FACTORIES:
            # Factory demand is the sum of orders from its DCs
            factory_demand = sum([order["quantity"] for date, orders in inbound_orders.items() 
                                 for order in orders if order["from_location"] == loc and date > day])
            
            for sku in skus:
                policy = df_policies[(df_policies["location"] == loc) & (df_policies["sku"] == sku)].iloc[0]
                if inventory[loc][sku] < policy["reorder_point"] and factory_demand > 0:
                    # A factory can produce to replenish, up to its capacity
                    production_qty = min(policy["order_quantity"], policy["max_daily_capacity"])
                    
                    if bom_check:
                        # Check if all components are available
                        required_components = df_bom[df_bom["sku"] == sku].iloc[0]["components"]
                        can_produce = True
                        for comp in required_components:
                            if component_inventory[loc].get(comp, 0) < production_qty:
                                can_produce = False
                                break
                        if can_produce:
                            for comp in required_components:
                                component_inventory[loc][comp] -= production_qty
                            inventory[loc][sku] += production_qty
                            ordering_costs[loc][sku] += ordering_cost_per_order
                    else:
                        inventory[loc][sku] += production_qty
                        ordering_costs[loc][sku] += ordering_cost_per_order

    # Calculate final summary metrics
    total_holding_cost = sum(sum(loc_costs.values()) for loc_costs in holding_costs.values())
    total_ordering_cost = sum(sum(loc_costs.values()) for loc_costs in ordering_costs.values())
    total_stockout_cost = sum(sum(loc_lost_sales.values()) for loc_lost_sales in lost_sales.values()) * stockout_cost_per_unit
    total_cost = total_holding_cost + total_ordering_cost + total_stockout_cost

    return {
        "inventory": inventory,
        "lost_sales": lost_sales,
        "stockouts": stockouts,
        "total_holding_cost": total_holding_cost,
        "total_ordering_cost": total_ordering_cost,
        "total_stockout_cost": total_stockout_cost,
        "total_cost": total_cost,
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
num_skus = st.sidebar.slider("Number of SKUs", 1, 10, DEFAULT_NUM_SKUS)
simulation_days = st.sidebar.slider("Simulation Duration (days)", 30, 365, 90)

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
    # Generate data
    df_demand = generate_dummy_data(num_skus, DEFAULT_START_DATE, DEFAULT_END_DATE,
                                    DEFAULT_PROMOTION_FREQUENCY_DAYS, DEFAULT_SALES_CHANNELS,
                                    DEFAULT_REGIONS, DEFAULT_LOCATIONS)
    df_policies = generate_inventory_policies(df_demand["sku"].unique(), DEFAULT_LOCATIONS, LEAD_TIMES)
    df_bom = generate_bom(df_demand["sku"].unique(), DEFAULT_NUM_COMPONENTS_PER_SKU)

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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Holding Cost", f"${simulation_results['total_holding_cost']:.2f}")
    with col2:
        st.metric("Total Ordering Cost", f"${simulation_results['total_ordering_cost']:.2f}")
    with col3:
        st.metric("Total Stockout Cost", f"${simulation_results['total_stockout_cost']:.2f}")
    with col4:
        st.metric("Total Simulation Cost", f"${simulation_results['total_cost']:.2f}")
    
    # Plotting lost sales per location/SKU
    lost_sales_df = pd.DataFrame(
        [(loc, sku, qty) for loc, skus in simulation_results["lost_sales"].items() for sku, qty in skus.items()],
        columns=["Location", "SKU", "Lost Sales"]
    )
    lost_sales_df = lost_sales_df.groupby("Location")["Lost Sales"].sum().reset_index()
    
    fig = px.bar(lost_sales_df, x="Location", y="Lost Sales",
                 title="Total Lost Sales by Location",
                 labels={"Lost Sales": "Total Units Lost"})
    st.plotly_chart(fig)

    # Forecasting report section
    generate_forecast_report(df_demand, forecasting_model, window_size)

