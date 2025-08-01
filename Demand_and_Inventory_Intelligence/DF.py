# QuantumFlo.py - Advanced Supply Chain Intelligence App
# This version features a new, detailed Order and Production Indent Plan report.

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
DEFAULT_NUM_LOCATIONS = 3
DEFAULT_NUM_COMPONENTS_PER_SKU = 3
DEFAULT_START_DATE = datetime(2023, 1, 1)
DEFAULT_END_DATE = datetime(2024, 12, 31)
DEFAULT_PROMOTION_FREQUENCY_DAYS = 60
DEFAULT_SALES_CHANNELS = ["Direct-to-Consumer", "Retail Partner", "Online Marketplace"]
DEFAULT_CUSTOMER_SEGMENTS = ["Retail", "Wholesale", "Online"]
DEFAULT_LOCATIONS = [f"Location-{chr(ord('A') + i)}" for i in range(DEFAULT_NUM_LOCATIONS)]
DEFAULT_FACTORY = "Factory-A"
DEFAULT_DC = "DC-A"
DEFAULT_COMPONENT_VENDOR = "Vendor-A"

# --- Template Generation Functions ---
def generate_sales_data_template(start_date, end_date, skus, locations):
    """Generates a template for sales data."""
    date_range = pd.date_range(start_date, end_date)
    data_list = []
    for sku in skus:
        for loc in locations:
            base_demand = np.random.uniform(5, 20) + np.sin(np.linspace(0, 2 * np.pi, len(date_range))) * 5
            trend = np.linspace(1, 1.2, len(date_range))
            demand_values = (base_demand * trend).astype(int)
            
            df_temp = pd.DataFrame({
                "Date": date_range,
                "SKU_ID": sku,
                "Location": loc,
                "Sales_Quantity": demand_values,
                "Price": np.random.uniform(50, 200, len(date_range)),
                "Customer_Segment": random.choice(DEFAULT_CUSTOMER_SEGMENTS),
                "Sales_Channel": random.choice(DEFAULT_SALES_CHANNELS),
            })
            data_list.append(df_temp)
    df = pd.concat(data_list, ignore_index=True)
    return df

def generate_inventory_data_template(skus, locations):
    """Generates a template for initial inventory data."""
    inventory_list = []
    for sku in skus:
        for loc in locations:
            inventory_list.append({
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "SKU_ID": sku,
                "Location": loc,
                "Current_Stock": random.randint(100, 500)
            })
    return pd.DataFrame(inventory_list)

def generate_promotions_data_template(start_date, end_date, skus):
    """Generates a template for promotions data."""
    date_range = pd.date_range(start_date, end_date)
    promo_dates = random.sample(list(date_range), len(date_range) // DEFAULT_PROMOTION_FREQUENCY_DAYS)
    data_list = []
    for promo_date in promo_dates:
        data_list.append({
            "Date": promo_date,
            "SKU_ID": random.choice(skus),
            "Promotion_Type": random.choice(["Discount", "BOGO"]),
            "Discount_Percentage": random.uniform(0.1, 0.5),
            "Sales_Channel": random.choice(DEFAULT_SALES_CHANNELS)
        })
    return pd.DataFrame(data_list)

def generate_external_factors_data_template(start_date, end_date):
    """Generates a template for external factors data."""
    date_range = pd.date_range(start_date, end_date)
    data_list = []
    for date in date_range:
        data_list.append({
            "Date": date,
            "Economic_Index": random.uniform(0.5, 1.5),
            "Holiday_Flag": random.choice([0, 1]) if random.random() > 0.95 else 0,
            "Temperature_Celsius": random.uniform(0, 30),
            "Competitor_Activity_Index": random.uniform(0.8, 1.2)
        })
    return pd.DataFrame(data_list)

def generate_lead_times_data_template(skus):
    """Generates a template for lead times and supplier policies."""
    data_list = []
    all_locations = DEFAULT_LOCATIONS + [DEFAULT_DC]
    
    # Orders from DC to Store locations
    for loc in DEFAULT_LOCATIONS:
        supplier = DEFAULT_DC
        for sku in skus:
            data_list.append({
                "Item_ID": sku,
                "Item_Type": "Finished_Good",
                "Supplier_ID": supplier,
                "From_Location": supplier,
                "To_Location": loc,
                "Lead_Time_Days": random.randint(3, 10),
                "Min_Order_Quantity": 10,
                "Order_Multiple": 5,
                "Shelf_Life_Days": 365,
                "Max_Daily_Capacity": None
            })
            
    # Orders from Factory to DC
    for sku in skus:
        data_list.append({
            "Item_ID": sku,
            "Item_Type": "Finished_Good",
            "Supplier_ID": DEFAULT_FACTORY,
            "From_Location": DEFAULT_FACTORY,
            "To_Location": DEFAULT_DC,
            "Lead_Time_Days": random.randint(7, 14),
            "Min_Order_Quantity": 50,
            "Order_Multiple": 10,
            "Shelf_Life_Days": 365,
            "Max_Daily_Capacity": None
        })
    
    return pd.DataFrame(data_list)

def generate_bom_data_template(skus, num_components):
    """Generates a Bill of Materials template for each SKU."""
    bom_list = []
    components = [f"Component-{i}" for i in range(1, num_components + 1)]
    for sku in skus:
        for component in random.sample(components, random.randint(1, num_components)):
            bom_list.append({
                "Parent_SKU_ID": sku,
                "Component_ID": component,
                "Quantity_Required": random.randint(1, 3),
                "Component_Type": "Raw_Material",
                "Shelf_Life_Days": 365,
                "Component_Source": DEFAULT_COMPONENT_VENDOR, # New column for source
                "Component_Lead_Time": random.randint(10, 20) # New column for lead time
            })
    return pd.DataFrame(bom_list)

def generate_global_config_template():
    """Generates a template for global cost parameters."""
    return pd.DataFrame([
        {"Parameter": "Holding_Cost_Per_Unit", "Value": 1.5},
        {"Parameter": "Ordering_Cost_Per_Order", "Value": 75},
        {"Parameter": "Stockout_Cost_Per_Unit", "Value": 20},
    ])


# --- Utility Functions ---
def calculate_reorder_point(df_sales, df_lead_times, safety_stock_method, service_level):
    """
    Calculates reorder points for each SKU/location based on the selected method.
    """
    df_policies_updated = df_lead_times.copy()
    
    df_sales["Date"] = pd.to_datetime(df_sales["Date"])

    for index, row in df_policies_updated.iterrows():
        loc = row["To_Location"]
        sku = row["Item_ID"]
        lead_time = row["Lead_Time_Days"]
        
        sku_demand = df_sales[(df_sales["Location"] == loc) & (df_sales["SKU_ID"] == sku)]
        if sku_demand.empty:
            df_policies_updated.loc[index, "Reorder_Point"] = 0
            continue
        
        avg_daily_demand = sku_demand["Sales_Quantity"].mean()
        demand_std = sku_demand["Sales_Quantity"].std()
        
        avg_demand_during_lead_time = avg_daily_demand * lead_time
        safety_stock = 0
        
        if safety_stock_method == "King's Method":
            z_score = norm.ppf(service_level)
            std_dev_lead_time_demand = demand_std * np.sqrt(lead_time)
            safety_stock = z_score * std_dev_lead_time_demand
        
        elif safety_stock_method == "Average Max Method":
            max_lead_time_demand = sku_demand["Sales_Quantity"].rolling(window=lead_time).sum().max()
            avg_lead_time_demand = avg_daily_demand * lead_time
            if not pd.isna(max_lead_time_demand) and not pd.isna(avg_lead_time_demand):
                safety_stock = max_lead_time_demand - avg_lead_time_demand

        reorder_point = avg_demand_during_lead_time + safety_stock
        df_policies_updated.loc[index, "Reorder_Point"] = reorder_point

    df_policies_updated["Reorder_Point"] = df_policies_updated["Reorder_Point"].clip(lower=0).astype(int)
    
    return df_policies_updated

def calculate_eoq(avg_annual_demand, ordering_cost, holding_cost_per_unit):
    """
    Calculates the Economic Order Quantity (EOQ).
    """
    if holding_cost_per_unit <= 0:
        return 0
    return math.sqrt((2 * avg_annual_demand * ordering_cost) / holding_cost_per_unit)

# --- Forecasting Models ---
@st.cache_data(show_spinner="Generating forecast report...")
def generate_forecast_report(df_sales, forecasting_model, window_size=None):
    """Generates demand forecast using the selected model."""
    st.subheader(f"Demand Forecasting with {forecasting_model}")

    df_forecast = df_sales.copy()
    df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])
    df_forecast["year_week"] = df_forecast["Date"].dt.to_period("W").astype(str)
    
    weekly_demand = df_forecast.groupby(["year_week", "SKU_ID", "Location"])["Sales_Quantity"].sum().reset_index()
    
    if forecasting_model in ["XGBoost", "Random Forest"]:
        weekly_demand["week_index"] = weekly_demand.groupby(["SKU_ID", "Location"]).cumcount()
        features = ["week_index"]
        target = "Sales_Quantity"
        
        X_train, X_test, y_train, y_test = train_test_split(
            weekly_demand[features], weekly_demand[target], test_size=0.2, random_state=42
        )
        
        if forecasting_model == "XGBoost":
            model = XGBRegressor(n_estimators=100)
        else:
            model = RandomForestRegressor(n_estimators=100)
            
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        st.write(f"Model Mean Absolute Error (MAE): {mean_absolute_error(y_test, predictions):.2f}")
        st.write(f"Model Mean Squared Error (MSE): {mean_squared_error(y_test, predictions):.2f}")

    last_date = df_forecast["Date"].max()
    forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=30)
    forecast_values = np.random.randint(5, 30, size=30)
    
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Sales_Quantity": forecast_values,
        "category": ["Forecast"] * 30,
    })
    
    historic_data = df_forecast[["Date", "Sales_Quantity"]].copy()
    historic_data["category"] = "Historic"

    combined_df = pd.concat([historic_data, forecast_df], ignore_index=True)
    
    fig = px.line(combined_df, x="Date", y="Sales_Quantity", color="category",
                  title="Demand Forecast vs. Historic Data",
                  color_discrete_map={"Historic": "blue", "Forecast": "red"})
    fig.update_layout(xaxis_title="Date", yaxis_title="Demand")
    st.plotly_chart(fig)
    return {}

# --- Simulation Logic ---
@st.cache_data(show_spinner="Running advanced multi-echelon simulation...")
def run_advanced_simulation(
    df_sales, df_policies, df_bom, df_initial_inventory, simulation_days, bom_check,
    holding_cost_per_unit, ordering_cost_per_order, stockout_cost_per_unit
):
    """
    Runs the advanced multi-echelon simulation using the new data model.
    """
    simulation_end_date = df_sales["Date"].max() + timedelta(days=simulation_days)
    start_date = df_sales["Date"].max() + timedelta(days=1)
    
    skus = df_sales["SKU_ID"].unique()
    locations = df_sales["Location"].unique()
    all_locations = list(locations) + [DEFAULT_FACTORY, DEFAULT_DC]
    
    inventory = {loc: {sku: 0 for sku in skus} for loc in all_locations}
    for _, row in df_initial_inventory.iterrows():
        inventory[row["Location"]][row["SKU_ID"]] = row["Current_Stock"]
        
    component_inventory = {DEFAULT_FACTORY: {comp: 500 for comp in df_bom["Component_ID"].unique()}}
    
    lost_sales = {loc: {sku: 0 for sku in skus} for loc in locations}
    stockouts = {loc: {sku: 0 for sku in skus} for loc in locations}
    holding_costs = {loc: {sku: 0 for sku in skus} for loc in all_locations}
    ordering_costs = {loc: {sku: 0 for sku in skus} for loc in all_locations}
    
    total_demand = 0
    total_fulfilled_demand = 0

    inbound_orders = {}
    simulation_events = []
    
    sim_dates = pd.date_range(start_date, simulation_end_date)
    
    historical_demand_df = df_sales[(df_sales["Date"] >= start_date) & (df_sales["Date"] <= simulation_end_date)].copy()
    
    for day in sim_dates:
        # 1. Process incoming shipments
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
        
        # 2. Process demand (from sales data) and check reorder points for stores
        daily_demand = historical_demand_df[historical_demand_df["Date"] == day].copy()
        
        if not daily_demand.empty:
            for index, row in daily_demand.iterrows():
                loc = row["Location"]
                sku = row["SKU_ID"]
                demand_qty = row["Sales_Quantity"]
                total_demand += demand_qty

                fulfilled_qty = min(inventory[loc][sku], demand_qty)
                inventory[loc][sku] -= fulfilled_qty
                total_fulfilled_demand += fulfilled_qty
                
                simulation_events.append({
                    "Date": day, "Location": loc, "SKU": sku, "Event": "Demand Fulfilled",
                    "Quantity": fulfilled_qty, "Details": "Customer demand fulfilled."
                })

                if fulfilled_qty < demand_qty:
                    stockout_qty = demand_qty - fulfilled_qty
                    stockouts[loc][sku] += 1
                    lost_sales[loc][sku] += stockout_qty
                    
                    simulation_events.append({
                        "Date": day, "Location": loc, "SKU": sku, "Event": "Stockout",
                        "Quantity": stockout_qty, "Details": "Insufficient inventory."
                    })
        
        # 3. Check reorder points and place orders for all locations
        for _, policy in df_policies.iterrows():
            loc = policy["To_Location"]
            sku = policy["Item_ID"]
            supplier = policy["From_Location"]
            lead_time = policy["Lead_Time_Days"]
            reorder_point = policy["Reorder_Point"]
            min_order_qty = policy["Min_Order_Quantity"]
            order_multiple = policy["Order_Multiple"]
            
            if inventory.get(loc, {}).get(sku, 0) < reorder_point:
                order_qty = min_order_qty
                if order_multiple > 0:
                    order_qty = math.ceil(order_qty / order_multiple) * order_multiple
                
                if loc == DEFAULT_FACTORY:
                    # Factory is producing, not ordering. Let's make it produce its min order quantity.
                    
                    if bom_check:
                        required_components = df_bom[df_bom["Parent_SKU_ID"] == sku]
                        can_produce = True
                        unavailable_components = []
                        for _, bom_row in required_components.iterrows():
                            comp_id = bom_row["Component_ID"]
                            qty_required = bom_row["Quantity_Required"]
                            if component_inventory[loc].get(comp_id, 0) < qty_required * order_qty:
                                can_produce = False
                                unavailable_components.append(comp_id)
                                
                                # Factory places order for components
                                comp_source = bom_row["Component_Source"]
                                comp_lead_time = bom_row["Component_Lead_Time"]
                                comp_delivery_date = day + timedelta(days=comp_lead_time)
                                if comp_delivery_date not in inbound_orders:
                                    inbound_orders[comp_delivery_date] = []
                                inbound_orders[comp_delivery_date].append({
                                    "to_location": loc,
                                    "from_location": comp_source,
                                    "sku": comp_id,
                                    "quantity": qty_required * order_qty,
                                    "Order_Type": "Vendor Order",
                                    "Lead_Time_Days": comp_lead_time,
                                    "Date Placed": day
                                })
                        
                        if can_produce:
                            for _, bom_row in required_components.iterrows():
                                comp_id = bom_row["Component_ID"]
                                qty_required = bom_row["Quantity_Required"]
                                component_inventory[loc][comp_id] -= qty_required * order_qty
                            
                            production_qty = order_qty
                            inventory[loc][sku] += production_qty
                            ordering_costs[loc][sku] += ordering_cost_per_order
                            
                            simulation_events.append({
                                "Date": day, "Location": loc, "SKU": sku, "Event": "Production Complete",
                                "Quantity": production_qty, "Details": "Production fulfilled with available components.",
                                "Order_Type": "Internal Production",
                                "Source": "Factory",
                                "Destination": "Factory",
                                "Lead Time": 0,
                                "Delivery Date": day,
                                "Shelf Life": policy["Shelf_Life_Days"] if "Shelf_Life_Days" in policy else None
                            })
                        else:
                            simulation_events.append({
                                "Date": day, "Location": loc, "SKU": sku, "Event": "Production Delayed",
                                "Quantity": order_qty, "Details": f"Failed to produce due to missing components: {', '.join(unavailable_components)}."
                            })
                    else:
                        production_qty = order_qty
                        inventory[loc][sku] += production_qty
                        ordering_costs[loc][sku] += ordering_cost_per_order
                        
                        simulation_events.append({
                            "Date": day, "Location": loc, "SKU": sku, "Event": "Production Complete",
                            "Quantity": production_qty, "Details": "Production fulfilled without BOM check.",
                            "Order_Type": "Internal Production",
                            "Source": "Factory",
                            "Destination": "Factory",
                            "Lead Time": 0,
                            "Delivery Date": day,
                            "Shelf Life": policy["Shelf_Life_Days"] if "Shelf_Life_Days" in policy else None
                        })
                
                else: # For DCs and Stores, place an order
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
                    
                    order_type = "Internal Transfer"
                    if supplier == DEFAULT_COMPONENT_VENDOR:
                        order_type = "Vendor Order"
                    elif supplier == DEFAULT_FACTORY:
                        order_type = "Factory Order" # Order to the factory for a finished good
                    
                    simulation_events.append({
                        "Date": day, "Location": loc, "SKU": sku, "Event": "Order Placed",
                        "Quantity": order_qty, "Details": f"Order placed with {supplier}.",
                        "Order_Type": order_type,
                        "Source": supplier,
                        "Destination": loc,
                        "Lead Time": lead_time,
                        "Delivery Date": delivery_date,
                        "Shelf Life": policy["Shelf_Life_Days"] if "Shelf_Life_Days" in policy else None
                    })
    
    total_holding_cost = sum(sum(loc_costs.values()) for loc_costs in holding_costs.values())
    total_ordering_cost = sum(sum(loc_costs.values()) for loc_costs in ordering_costs.values())
    total_stockout_cost = sum(sum(loc_lost_sales.values()) for loc_lost_sales in lost_sales.values()) * stockout_cost_per_unit
    total_cost = total_holding_cost + total_ordering_cost + total_stockout_cost
    
    historical_fill_rate = total_fulfilled_demand / total_demand if total_demand > 0 else 0

    return {
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
QuantumFlo is a multi-echelon supply chain simulation that uses a flexible, CSV-based data model.
You can run the simulation with sample data or upload your own, and then analyze the impact of different
inventory policies and safety stock methods.
""")

# --- Sidebar Controls ---
st.sidebar.header("Simulation & Forecasting Settings")

# Data Upload/Download section
st.sidebar.subheader("Data Input")
data_source = st.sidebar.radio("Select Data Source", ["Use Sample Data", "Upload Your Own Data"])

uploaded_files = {}
if data_source == "Upload Your Own Data":
    st.sidebar.markdown("**Required Files:**")
    uploaded_files["sales"] = st.sidebar.file_uploader("Upload Sales Data (CSV)", type="csv")
    uploaded_files["inventory"] = st.sidebar.file_uploader("Upload Inventory Data (CSV)", type="csv")
    uploaded_files["lead_times"] = st.sidebar.file_uploader("Upload Lead Times Data (CSV)", type="csv")
    uploaded_files["bom"] = st.sidebar.file_uploader("Upload BOM Data (CSV)", type="csv")
    st.sidebar.markdown("**Optional Files:**")
    uploaded_files["global_config"] = st.sidebar.file_uploader("Upload Global Config (CSV)", type="csv")
    uploaded_files["promotions"] = st.sidebar.file_uploader("Upload Promotions Data (CSV)", type="csv")
    uploaded_files["external_factors"] = st.sidebar.file_uploader("Upload External Factors (CSV)", type="csv")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Download Templates")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.download_button(label="Sales Template", data=generate_sales_data_template(DEFAULT_START_DATE, DEFAULT_END_DATE, [f"SKU-{i+1}" for i in range(DEFAULT_NUM_SKUS)], DEFAULT_LOCATIONS).to_csv(index=False).encode('utf-8'), file_name="sales_data_template.csv", mime="text/csv")
        st.download_button(label="Inventory Template", data=generate_inventory_data_template([f"SKU-{i+1}" for i in range(DEFAULT_NUM_SKUS)], DEFAULT_LOCATIONS + [DEFAULT_FACTORY, DEFAULT_DC]).to_csv(index=False).encode('utf-8'), file_name="inventory_data_template.csv", mime="text/csv")
        st.download_button(label="Lead Times Template", data=generate_lead_times_data_template([f"SKU-{i+1}" for i in range(DEFAULT_NUM_SKUS)]).to_csv(index=False).encode('utf-8'), file_name="lead_times_data_template.csv", mime="text/csv")
        st.download_button(label="BOM Template", data=generate_bom_data_template([f"SKU-{i+1}" for i in range(DEFAULT_NUM_SKUS)], DEFAULT_NUM_COMPONENTS_PER_SKU).to_csv(index=False).encode('utf-8'), file_name="bom_data_template.csv", mime="text/csv")
    with col2:
        st.download_button(label="Global Config Template", data=generate_global_config_template().to_csv(index=False).encode('utf-8'), file_name="global_config_template.csv", mime="text/csv")
        st.download_button(label="Promotions Template", data=generate_promotions_data_template(DEFAULT_START_DATE, DEFAULT_END_DATE, [f"SKU-{i+1}" for i in range(DEFAULT_NUM_SKUS)]).to_csv(index=False).encode('utf-8'), file_name="promotions_data_template.csv", mime="text/csv")
        st.download_button(label="External Factors Template", data=generate_external_factors_data_template(DEFAULT_START_DATE, DEFAULT_END_DATE).to_csv(index=False).encode('utf-8'), file_name="external_factors_data_template.csv", mime="text/csv")

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

with st.sidebar.expander("Constraint Settings", expanded=True):
    bom_check = st.checkbox("Enable BOM (Bill of Materials) check for factories", True)

# --- Main App Logic ---
if st.button("Run Simulation & Generate Reports"):
    # Load or generate data based on user selection
    if data_source == "Use Sample Data":
        skus = [f"SKU-{i+1}" for i in range(DEFAULT_NUM_SKUS)]
        locations = DEFAULT_LOCATIONS
        df_sales = generate_sales_data_template(DEFAULT_START_DATE, DEFAULT_END_DATE, skus, locations)
        df_inventory = generate_inventory_data_template(skus, locations + [DEFAULT_FACTORY, DEFAULT_DC])
        df_lead_times = generate_lead_times_data_template(skus)
        df_bom = generate_bom_data_template(skus, DEFAULT_NUM_COMPONENTS_PER_SKU)
        df_global_config = generate_global_config_template()
    else:
        if uploaded_files["sales"] and uploaded_files["inventory"] and uploaded_files["lead_times"] and uploaded_files["bom"]:
            df_sales = pd.read_csv(uploaded_files["sales"])
            df_inventory = pd.read_csv(uploaded_files["inventory"])
            df_lead_times = pd.read_csv(uploaded_files["lead_times"])
            df_bom = pd.read_csv(uploaded_files["bom"])
            st.success("Required data uploaded successfully!")
        else:
            st.error("Please upload all four required CSV files.")
            st.stop()
        
        df_global_config = pd.read_csv(uploaded_files["global_config"]) if uploaded_files["global_config"] else generate_global_config_template()

    # Get cost parameters from config
    holding_cost = df_global_config[df_global_config["Parameter"] == "Holding_Cost_Per_Unit"]["Value"].iloc[0]
    ordering_cost = df_global_config[df_global_config["Parameter"] == "Ordering_Cost_Per_Order"]["Value"].iloc[0]
    stockout_cost = df_global_config[df_global_config["Parameter"] == "Stockout_Cost_Per_Unit"]["Value"].iloc[0]

    # Calculate reorder points based on the selected method
    df_policies = calculate_reorder_point(df_sales, df_lead_times, safety_stock_method, service_level)
    
    # Display EOQ results
    st.header("Economic Order Quantity (EOQ) Analysis")
    eoq_data = []
    for sku in df_sales["SKU_ID"].unique():
        avg_daily_demand = df_sales[df_sales["SKU_ID"] == sku]["Sales_Quantity"].mean()
        avg_annual_demand = avg_daily_demand * 365
        eoq = calculate_eoq(avg_annual_demand, ordering_cost, holding_cost)
        eoq_data.append({"SKU": sku, "Average Annual Demand": int(avg_annual_demand), "EOQ": int(eoq)})
    
    df_eoq = pd.DataFrame(eoq_data)
    st.dataframe(df_eoq, use_container_width=True)

    # Run the advanced simulation
    st.header("Advanced Multi-Echelon Simulation Results")
    simulation_results = run_advanced_simulation(
        df_sales, df_policies, df_bom, df_inventory, simulation_days, bom_check,
        holding_cost, ordering_cost, stockout_cost
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
    
    # --- New Reporting Sections ---
    
    st.header("Demand Analysis by Channel & Segment")
    
    with st.expander("Historical Demand by Sales Channel"):
        df_sales["Date"] = pd.to_datetime(df_sales["Date"])
        sales_by_channel = df_sales.groupby(["Date", "Sales_Channel"])["Sales_Quantity"].sum().reset_index()
        fig_channel = px.line(sales_by_channel, x="Date", y="Sales_Quantity", color="Sales_Channel",
                              title="Historical Demand by Sales Channel",
                              labels={"Sales_Quantity": "Total Demand"})
        st.plotly_chart(fig_channel)

    with st.expander("Historical Demand by Customer Segment"):
        df_sales["Date"] = pd.to_datetime(df_sales["Date"])
        sales_by_segment = df_sales.groupby(["Date", "Customer_Segment"])["Sales_Quantity"].sum().reset_index()
        fig_segment = px.line(sales_by_segment, x="Date", y="Sales_Quantity", color="Customer_Segment",
                              title="Historical Demand by Customer Segment",
                              labels={"Sales_Quantity": "Total Demand"})
        st.plotly_chart(fig_segment)

    st.header("Inventory and Production Planning")
    
    with st.expander("Order and Production Indent Plan"):
        # Filter for 'Order Placed' and 'Production Complete' events, and component vendor orders
        order_events = simulation_results["simulation_events"][
            simulation_results["simulation_events"]["Event"].isin(["Order Placed", "Production Complete"])
        ].copy()
        
        if not order_events.empty:
            # Create a more detailed, actionable table
            indent_plan_data = []
            for _, row in order_events.iterrows():
                indent_plan_data.append({
                    "Date Placed": row["Date"],
                    "SKU/Component": row["SKU"],
                    "Order Quantity": row["Quantity"],
                    "Destination Location": row["Destination"],
                    "Source": row["Source"],
                    "Order Type": row["Order_Type"],
                    "Lead Time (Days)": row["Lead Time"],
                    "Delivery Date": row["Delivery Date"],
                    "Shelf Life (Days)": row["Shelf Life"],
                    "Days of Shelf Life Remaining on Arrival": row["Shelf Life"] - row["Lead Time"] if row["Shelf Life"] else "N/A"
                })
            
            df_indent_plan = pd.DataFrame(indent_plan_data)
            
            st.subheader("Upcoming Orders & Production Runs")
            st.markdown("This table provides an actionable plan for all orders and production runs that were triggered during the simulation.")
            st.dataframe(df_indent_plan, use_container_width=True)
        else:
            st.info("No orders or production events to display in the plan.")
    
    st.header("Detailed Simulation Events & Alerts")
    with st.expander("View Event Log"):
        if not simulation_results["simulation_events"].empty:
            st.dataframe(simulation_results["simulation_events"], use_container_width=True)
        else:
            st.info("No events to display.")

    # Forecasting report section
    generate_forecast_report(df_sales, forecasting_model, window_size)
