# QuantumFlo.py - Advanced Supply Chain Intelligence App
# This version features realistic dummy data, drill-down filters, and a more robust simulation.
# New: Time aggregation for insights and a dedicated FAQ tab.

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
# New, more specific configuration for a multi-echelon network
DEFAULT_NUM_SKUS = 10
DEFAULT_NUM_COMPONENTS_PER_SKU = 3
DEFAULT_START_DATE = datetime(2023, 1, 1)
DEFAULT_END_DATE = datetime(2024, 12, 31)
DEFAULT_PROMOTION_FREQUENCY_DAYS = 60
DEFAULT_SALES_CHANNELS = ["Direct-to-Consumer", "Retail Partner", "Online Marketplace"]
DEFAULT_CUSTOMER_SEGMENTS = ["Retail", "Wholesale", "Online"]
DEFAULT_FACTORY = ["Factory-A", "Factory-B"]
DEFAULT_DC = ["DC-A", "DC-B", "DC-C"]
DEFAULT_RETAIL_STORES = [f"Store-{i}" for i in range(1, 11)]
DEFAULT_COMPONENT_VENDORS = ["Vendor-A", "Vendor-B"]

# --- Template Generation Functions ---

def generate_realistic_data(num_skus, num_components):
    """
    Generates a complete set of realistic dummy data for the simulation.
    This includes SKUs, BOM, network, and demand.
    """
    st.info("Generating a complete set of realistic dummy data...")
    
    # 1. Generate realistic SKUs
    product_names = [
        "Eco-Tracker Smartwatch", "Aura Wireless Headphones", "Chrono-Fit Band", "Fusion Powerbank",
        "Apex Pro E-Reader", "Quantum Bluetooth Speaker", "Zenith VR Headset", "Sonic-Pulse Earbuds",
        "Vortex Dash Cam", "Cyber-Pen Stylus", "Terra Tablet", "Luna Laptop", "Solar Charger"
    ]
    product_categories = ["Smartwatch", "Audio", "E-Reader", "VR", "Accessories", "Tablet", "Laptop", "Power"]
    
    sku_data = []
    for i in range(num_skus):
        sku_id = f"SKU-{i+1}"
        sku_name = random.choice(product_names)
        category = random.choice(product_categories)
        sku_data.append({
            "SKU_ID": sku_id,
            "Name": sku_name,
            "Category": category,
            "Price": round(random.uniform(50, 1000), 2),
            "Shelf_Life_Days": random.randint(180, 730)
        })
    df_skus = pd.DataFrame(sku_data)
    
    # 2. Generate Bill of Materials (BOM)
    component_names = [
        "Lithium-ion Battery", "OLED Display", "Microprocessor", "Plastic Casing",
        "Circuit Board", "Charging Cable", "Touch Screen", "Accelerometer",
        "Speaker Driver", "Microphone", "VR Lens", "SSD Drive", "RAM Stick", "Solar Panel"
    ]
    all_components = list(set(component_names))
    
    bom_data = []
    for sku_id in df_skus["SKU_ID"]:
        components_for_sku = random.sample(all_components, min(num_components, len(all_components)))
        for comp in components_for_sku:
            bom_data.append({
                "Parent_SKU_ID": sku_id,
                "Component_ID": comp,
                "Quantity_Required": random.randint(1, 3),
                "Component_Source": random.choice(DEFAULT_COMPONENT_VENDORS),
                "Component_Lead_Time": random.randint(10, 30)
            })
    df_bom = pd.DataFrame(bom_data)
    
    # 3. Generate Supply Chain Network
    network_data = []
    
    # Factory to DC connections
    for dc in DEFAULT_DC:
        source_factory = random.choice(DEFAULT_FACTORY)
        for sku_id in df_skus["SKU_ID"]:
            network_data.append({
                "Item_ID": sku_id,
                "Item_Type": "Finished_Good",
                "From_Location": source_factory,
                "To_Location": dc,
                "Lead_Time_Days": random.randint(3, 14)
            })
            
    # DC to Store connections
    for store in DEFAULT_RETAIL_STORES:
        source_dc = random.choice(DEFAULT_DC)
        for sku_id in df_skus["SKU_ID"]:
            network_data.append({
                "Item_ID": sku_id,
                "Item_Type": "Finished_Good",
                "From_Location": source_dc,
                "To_Location": store,
                "Lead_Time_Days": random.randint(1, 7)
            })
            
    df_network = pd.DataFrame(network_data)
    
    # 4. Generate Demand History
    date_range = pd.date_range(start=DEFAULT_START_DATE, end=DEFAULT_END_DATE)
    demand_data = []
    
    # Holiday demand spikes
    holidays = {
        (12, 25): 1.8, # Christmas
        (11, 23): 2.5, # Black Friday (approx)
        (7, 4): 1.5,  # 4th of July
    }
    
    # Trend and seasonality
    total_days = (DEFAULT_END_DATE - DEFAULT_START_DATE).days
    day_of_year_sin = np.sin(np.linspace(0, 2 * np.pi, total_days)) * 0.4 + 1.0 # Yearly seasonality
    day_of_year_cos = np.cos(np.linspace(0, 4 * np.pi, total_days)) * 0.1 + 1.0 # Quarterly
    
    for _, sku_row in df_skus.iterrows():
        sku_id = sku_row['SKU_ID']
        for location in DEFAULT_RETAIL_STORES:
            base_demand = random.randint(10, 50)
            
            for i, date in enumerate(date_range):
                demand_factor = 1.0
                
                demand_factor *= day_of_year_sin[i] * day_of_year_cos[i]
                
                # Apply holiday effect
                holiday_key = (date.month, date.day)
                if holiday_key in holidays:
                    demand_factor *= holidays[holiday_key]
                
                # Apply promotion effect
                if random.random() < 0.05: # 5% chance of a promo day
                    demand_factor *= random.uniform(1.5, 2.5)
                
                daily_demand = max(0, int(base_demand * demand_factor * np.random.normal(1, 0.2)))
                
                demand_data.append({
                    "Date": date,
                    "SKU_ID": sku_id,
                    "Location": location,
                    "Sales_Quantity": daily_demand,
                    "Price": sku_row['Price'],
                    "Sales_Channel": random.choice(DEFAULT_SALES_CHANNELS),
                    "Customer_Segment": random.choice(DEFAULT_CUSTOMER_SEGMENTS),
                })
    
    df_sales = pd.DataFrame(demand_data)
    
    # 5. Generate initial inventory and reorder policies
    df_initial_inventory = pd.DataFrame()
    df_policies = df_network.copy()
    
    all_sim_locations = list(set(DEFAULT_RETAIL_STORES + DEFAULT_DC + DEFAULT_FACTORY))
    initial_inventory_data = []
    
    for loc in all_sim_locations:
        for sku in df_skus["SKU_ID"]:
            initial_inventory_data.append({
                "Location": loc,
                "SKU_ID": sku,
                "Current_Stock": random.randint(100, 500) if loc in DEFAULT_RETAIL_STORES + DEFAULT_DC else random.randint(1000, 2000)
            })
    df_initial_inventory = pd.DataFrame(initial_inventory_data)
    
    # Reorder points and quantities are now set during the main app run
    df_policies["Min_Order_Quantity"] = 0
    df_policies["Order_Multiple"] = 0
    
    # 6. Global Config
    df_global_config = pd.DataFrame([
        {"Parameter": "Holding_Cost_Per_Unit", "Value": 1.5},
        {"Parameter": "Ordering_Cost_Per_Order", "Value": 75},
        {"Parameter": "Stockout_Cost_Per_Unit", "Value": 20},
    ])
    
    return df_sales, df_initial_inventory, df_policies, df_bom, df_global_config, df_skus

# --- Utility Functions ---

def calculate_reorder_point_and_eoq(df_sales, df_policies, service_level, holding_cost_per_unit, ordering_cost_per_order):
    """
    Calculates reorder points and EOQ for each SKU/location based on the selected method.
    """
    df_policies_updated = df_policies.copy()
    
    df_sales["Date"] = pd.to_datetime(df_sales["Date"])
    
    for index, row in df_policies_updated.iterrows():
        loc = row["To_Location"]
        sku = row["Item_ID"]
        lead_time = row["Lead_Time_Days"]
        
        sku_demand = df_sales[(df_sales["Location"] == loc) & (df_sales["SKU_ID"] == sku)]["Sales_Quantity"]
        
        if sku_demand.empty:
            df_policies_updated.loc[index, "Reorder_Point"] = 0
            df_policies_updated.loc[index, "Min_Order_Quantity"] = 0
            continue
        
        avg_daily_demand = sku_demand.mean()
        demand_std = sku_demand.std()
        
        # Calculate Reorder Point (King's Method)
        z_score = norm.ppf(service_level)
        std_dev_lead_time_demand = demand_std * np.sqrt(lead_time) if not pd.isna(demand_std) else 0
        safety_stock = z_score * std_dev_lead_time_demand
        reorder_point = (avg_daily_demand * lead_time) + safety_stock
        df_policies_updated.loc[index, "Reorder_Point"] = reorder_point
        
        # Calculate EOQ
        days_in_year = 365
        annual_demand = avg_daily_demand * days_in_year
        # Assuming Price is a good proxy for value for holding cost calculation
        if holding_cost_per_unit > 0:
            eoq = math.sqrt((2 * annual_demand * ordering_cost_per_order) / (holding_cost_per_unit * row["Price"]))
        else:
            eoq = 0
        
        # Update Min_Order_Quantity with EOQ
        df_policies_updated.loc[index, "Min_Order_Quantity"] = eoq

    df_policies_updated["Reorder_Point"] = df_policies_updated["Reorder_Point"].clip(lower=0).astype(int)
    df_policies_updated["Min_Order_Quantity"] = df_policies_updated["Min_Order_Quantity"].clip(lower=0).astype(int)
    
    return df_policies_updated

# --- Simulation Logic ---
@st.cache_data(show_spinner="Running advanced multi-echelon simulation...")
def run_advanced_simulation(
    df_sales, df_policies, df_bom, df_initial_inventory, simulation_days, bom_check,
    holding_cost_per_unit, ordering_cost_per_order, stockout_cost_per_unit, df_skus
):
    """
    Runs the advanced multi-echelon simulation.
    """
    st.info(f"Starting a {simulation_days}-day simulation...")

    sim_start_date = df_sales["Date"].max()
    sim_end_date = sim_start_date + timedelta(days=simulation_days)
    
    skus = df_skus["SKU_ID"].unique()
    locations = df_sales["Location"].unique()
    all_sim_locations = list(set(DEFAULT_RETAIL_STORES + DEFAULT_DC + DEFAULT_FACTORY))
    
    # Initialize inventory and other metrics
    inventory = {loc: {sku: 0 for sku in skus} for loc in all_sim_locations}
    for _, row in df_initial_inventory.iterrows():
        inventory[row["Location"]][row["SKU_ID"]] = row["Current_Stock"]
    
    # Initialize component inventory for factories
    component_inventory = {factory: {comp: random.randint(1000, 5000) for comp in df_bom["Component_ID"].unique()} for factory in DEFAULT_FACTORY}
    
    inbound_orders = {}
    simulation_events = []
    
    # Simulation metrics
    total_holding_cost = 0
    total_ordering_cost = 0
    total_stockout_cost = 0
    total_demand_units = 0
    total_fulfilled_units = 0

    sim_dates = pd.date_range(sim_start_date, sim_end_date)
    
    for day in sim_dates:
        # 1. Process incoming shipments
        orders_to_process = inbound_orders.get(day, [])
        for order in orders_to_process:
            to_loc = order["to_location"]
            sku = order["sku"]
            quantity = order["quantity"]
            
            if to_loc in inventory and sku in inventory[to_loc]:
                inventory[to_loc][sku] += quantity
                
                sku_price = df_skus[df_skus['SKU_ID'] == sku]['Price'].iloc[0] if not df_skus[df_skus['SKU_ID'] == sku].empty else 1.0
                total_holding_cost += quantity * sku_price * holding_cost_per_unit
                
                simulation_events.append({
                    "Date": day, "Location": to_loc, "SKU": sku, "Event": "Order Received",
                    "Quantity": quantity, "Details": f"Order from {order['from_location']} arrived."
                })
        
        # 2. Process demand (at retail stores only)
        daily_demand = df_sales[df_sales["Date"] == day].copy()
        
        for _, row in daily_demand.iterrows():
            loc = row["Location"]
            sku = row["SKU_ID"]
            demand_qty = row["Sales_Quantity"]
            
            if loc in DEFAULT_RETAIL_STORES:
                total_demand_units += demand_qty
                
                fulfilled_qty = min(inventory.get(loc, {}).get(sku, 0), demand_qty)
                
                if loc in inventory and sku in inventory[loc]:
                    inventory[loc][sku] -= fulfilled_qty
                total_fulfilled_units += fulfilled_qty
                
                if fulfilled_qty < demand_qty:
                    stockout_qty = demand_qty - fulfilled_qty
                    total_stockout_cost += stockout_qty * stockout_cost_per_unit
                    simulation_events.append({
                        "Date": day, "Location": loc, "SKU": sku, "Event": "Stockout",
                        "Quantity": stockout_qty, "Details": "Insufficient inventory."
                    })

        # 3. Check reorder points and place orders
        for _, policy in df_policies.iterrows():
            loc = policy["To_Location"]
            sku = policy["Item_ID"]
            supplier = policy["From_Location"]
            lead_time = policy["Lead_Time_Days"]
            reorder_point = policy["Reorder_Point"]
            order_qty = policy["Min_Order_Quantity"] # EOQ is used here
            
            # Check inventory level against reorder point
            if inventory.get(loc, {}).get(sku, 0) < reorder_point:
                
                # Special case: factory production
                if loc in DEFAULT_FACTORY:
                    if bom_check:
                        required_components = df_bom[df_bom["Parent_SKU_ID"] == sku]
                        can_produce = True
                        for _, bom_row in required_components.iterrows():
                            comp_id = bom_row["Component_ID"]
                            qty_required = bom_row["Quantity_Required"]
                            if component_inventory[loc].get(comp_id, 0) < qty_required * order_qty:
                                can_produce = False
                                # Factory orders components if needed
                                comp_delivery_date = day + timedelta(days=bom_row["Component_Lead_Time"])
                                if comp_delivery_date not in inbound_orders: inbound_orders[comp_delivery_date] = []
                                inbound_orders[comp_delivery_date].append({
                                    "to_location": loc, "from_location": bom_row["Component_Source"],
                                    "sku": comp_id, "quantity": qty_required * order_qty
                                })
                                break
                        
                        if can_produce:
                            # Consume components
                            for _, bom_row in required_components.iterrows():
                                comp_id = bom_row["Component_ID"]
                                qty_required = bom_row["Quantity_Required"]
                                component_inventory[loc][comp_id] -= qty_required * order_qty
                            
                            # Produce finished goods
                            inventory[loc][sku] += order_qty
                            total_ordering_cost += ordering_cost_per_order
                            simulation_events.append({
                                "Date": day, "Location": loc, "SKU": sku, "Event": "Production Complete",
                                "Quantity": order_qty, "Details": "Production fulfilled with available components."
                            })
                    else: # No BOM check
                        inventory[loc][sku] += order_qty
                        total_ordering_cost += ordering_cost_per_order
                        simulation_events.append({
                            "Date": day, "Location": loc, "SKU": sku, "Event": "Production Complete",
                            "Quantity": order_qty, "Details": "Production fulfilled without BOM check."
                        })
                else: # DC or Store places an order
                    delivery_date = day + timedelta(days=lead_time)
                    if delivery_date not in inbound_orders: inbound_orders[delivery_date] = []
                    
                    inbound_orders[delivery_date].append({
                        "to_location": loc,
                        "from_location": supplier,
                        "sku": sku,
                        "quantity": order_qty
                    })
                    total_ordering_cost += ordering_cost_per_order
                    simulation_events.append({
                        "Date": day, "Location": loc, "SKU": sku, "Event": "Order Placed",
                        "Quantity": order_qty, "Details": f"Order placed with {supplier}."
                    })

    # Prepare final results
    sim_results_df = pd.DataFrame(simulation_events)
    fill_rate = (total_fulfilled_units / total_demand_units) if total_demand_units > 0 else 0
    
    return {
        "total_holding_cost": total_holding_cost,
        "total_ordering_cost": total_ordering_cost,
        "total_stockout_cost": total_stockout_cost,
        "total_cost": total_holding_cost + total_ordering_cost + total_stockout_cost,
        "fill_rate": fill_rate,
        "simulation_events": sim_results_df
    }

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="QuantumFlo")

st.title("QuantumFlo: Advanced Supply Chain Simulation")
st.markdown("""
    QuantumFlo is a powerful tool for simulating and optimizing your multi-echelon supply chain.
    Use the controls on the left to configure the simulation and analyze the results.
""")

# --- Sidebar Filters & Configuration ---
with st.sidebar:
    st.header("App Configuration")
    
    # Data Generation
    st.subheader("Data Generation")
    with st.expander("Generate New Data", expanded=False):
        st.write("Click below to generate a new, random dataset.")
        if st.button("Generate New Dataset"):
            st.session_state["data"] = generate_realistic_data(DEFAULT_NUM_SKUS, DEFAULT_NUM_COMPONENTS_PER_SKU)
            st.success("New data generated!")
    
    # Simulation Parameters
    st.subheader("Simulation Parameters")
    simulation_days = st.slider("Simulation Duration (days)", 30, 365, 90)
    service_level = st.slider("Desired Service Level (%)", 80, 99, 95) / 100
    bom_check = st.checkbox("Enable BOM (Bill of Materials) Check", value=True)

    # Time aggregation
    st.subheader("Time Aggregation")
    aggregation_level = st.radio("Group results by:", ["Day", "Week", "Month"])
    
    # Filters
    st.subheader("Data Filters")
    # This check ensures data has been generated
    if "data" in st.session_state and st.session_state["data"]:
        df_sales = st.session_state["data"][0]
        
        all_skus = ["All"] + sorted(df_sales["SKU_ID"].unique().tolist())
        selected_sku = st.selectbox("Select SKU", all_skus)
        
        all_locations = ["All"] + sorted(df_sales["Location"].unique().tolist())
        selected_location = st.selectbox("Select Location", all_locations)
        
        start_date = df_sales["Date"].min()
        end_date = df_sales["Date"].max()
        selected_date_range = st.slider(
            "Select Date Range",
            min_value=start_date.date(),
            max_value=end_date.date(),
            value=(start_date.date(), end_date.date())
        )
    else:
        st.warning("Please generate data first.")
        selected_sku, selected_location, selected_date_range = "All", "All", (datetime.now().date(), datetime.now().date())


# --- Main Content Area ---
if "data" not in st.session_state or not st.session_state["data"]:
    st.warning("Please generate a new dataset using the sidebar to begin.")
else:
    df_sales, df_initial_inventory, df_policies, df_bom, df_global_config, df_skus = st.session_state["data"]
    
    # Get costs from global config
    holding_cost_per_unit = df_global_config[df_global_config["Parameter"] == "Holding_Cost_Per_Unit"]["Value"].iloc[0]
    ordering_cost_per_order = df_global_config[df_global_config["Parameter"] == "Ordering_Cost_Per_Order"]["Value"].iloc[0]
    stockout_cost_per_unit = df_global_config[df_global_config["Parameter"] == "Stockout_Cost_Per_Unit"]["Value"].iloc[0]

    # Calculate reorder points and EOQ
    df_sales = df_sales.merge(df_skus[['SKU_ID', 'Price']], on='SKU_ID', how='left')
    df_policies = df_policies.merge(df_skus[['SKU_ID', 'Price']], left_on='Item_ID', right_on='SKU_ID', how='left')
    df_policies_with_policy = calculate_reorder_point_and_eoq(df_sales, df_policies, service_level, holding_cost_per_unit, ordering_cost_per_order)

    # --- Run Simulation ---
    if st.button("Run Simulation"):
        st.session_state["simulation_results"] = run_advanced_simulation(
            df_sales, df_policies_with_policy, df_bom, df_initial_inventory, simulation_days, bom_check,
            holding_cost_per_unit, ordering_cost_per_order, stockout_cost_per_unit, df_skus
        )
        st.success("Simulation complete! View results below.")

    tab1, tab2 = st.tabs(["Dashboard", "FAQ"])

    with tab1:
        if "simulation_results" in st.session_state:
            simulation_results = st.session_state["simulation_results"]
            
            # --- Apply filters to data for display ---
            filtered_sim_events = simulation_results["simulation_events"].copy()
            
            # Date range filter
            start_date_filter, end_date_filter = pd.to_datetime(selected_date_range[0]), pd.to_datetime(selected_date_range[1])
            filtered_sim_events = filtered_sim_events[
                (filtered_sim_events["Date"] >= start_date_filter) & (filtered_sim_events["Date"] <= end_date_filter)
            ]
            
            # Location filter
            if selected_location != "All":
                filtered_sim_events = filtered_sim_events[filtered_sim_events["Location"] == selected_location]
                
            # SKU filter
            if selected_sku != "All":
                filtered_sim_events = filtered_sim_events[filtered_sim_events["SKU"] == selected_sku]

            # --- Time aggregation logic ---
            filtered_sim_events['Period'] = filtered_sim_events['Date']
            if aggregation_level == "Week":
                # Get the year and week number for aggregation
                filtered_sim_events['Period'] = filtered_sim_events['Date'].dt.to_period('W').astype(str)
            elif aggregation_level == "Month":
                filtered_sim_events['Period'] = filtered_sim_events['Date'].dt.to_period('M').astype(str)
            
            # --- Display KPIs ---
            st.header("Simulation Results & KPIs")
            col1, col2, col3, col4 = st.columns(4)
            
            total_cost = simulation_results["total_cost"]
            fill_rate = simulation_results["fill_rate"]
            
            with col1:
                st.metric("Total Cost", f"${total_cost:,.2f}")
            with col2:
                st.metric("Total Holding Cost", f"${simulation_results['total_holding_cost']:,.2f}")
            with col3:
                st.metric("Total Stockout Cost", f"${simulation_results['total_stockout_cost']:,.2f}")
            with col4:
                st.metric("Historical Fill Rate", f"{fill_rate * 100:.2f}%")

            # --- Plots & Reports ---
            st.header("Detailed Reports")

            st.subheader("Inventory Levels Over Time")
            # Aggregate inventory levels from filtered events
            inventory_over_time = filtered_sim_events[filtered_sim_events["Event"].isin(["Order Received", "Demand Fulfilled", "Production Complete"])].copy()
            inventory_over_time["net_change"] = inventory_over_time.apply(
                lambda row: row["Quantity"] if row["Event"] in ["Order Received", "Production Complete"] else -row["Quantity"], axis=1
            )
            # Group by period, location, and SKU
            inventory_over_time = inventory_over_time.groupby(["Period", "Location", "SKU"])["net_change"].sum().reset_index()
            inventory_over_time["inventory_level"] = inventory_over_time.groupby(["Location", "SKU"])["net_change"].cumsum()
            
            fig_inv = px.line(
                inventory_over_time,
                x="Period",
                y="inventory_level",
                color="SKU",
                line_dash="Location",
                title=f"Inventory Levels ({aggregation_level} Aggregation)"
            )
            st.plotly_chart(fig_inv, use_container_width=True)

            st.subheader("Stockout Events and Lost Sales")
            stockout_events = filtered_sim_events[filtered_sim_events["Event"] == "Stockout"].copy()
            if not stockout_events.empty:
                # Aggregate stockout data by period
                stockout_by_period = stockout_events.groupby(["Period", "SKU"])["Quantity"].sum().reset_index()
                fig_stockout = px.bar(
                    stockout_by_period,
                    x="Period",
                    y="Quantity",
                    color="SKU",
                    title=f"Lost Sales by {aggregation_level}",
                    hover_data=["SKU"]
                )
                st.plotly_chart(fig_stockout, use_container_width=True)
            else:
                st.info("No stockout events occurred in the selected date range and filters.")

            st.subheader("Detailed Simulation Event Log")
            st.dataframe(filtered_sim_events, use_container_width=True)

    with tab2:
        st.header("❓ Frequently Asked Questions")
        st.markdown("""
        ---
        ### **Q: How does the simulation work?**
        **A:** The `QuantumFlo` simulation runs on a daily basis, modeling the flow of goods through your supply chain. It starts at the retail store level, fulfilling customer demand for each SKU. If a store's inventory falls below its reorder point, it places an order with its supplying distribution center. The simulation then moves to the next day, and these orders become the demand for the distribution centers. The same logic applies from the DCs to the factories, creating a realistic, time-sensitive simulation of the entire supply chain.
        ---
        ### **Q: What does the `BOM` check do?**
        **A:** When enabled, the simulation will only allow a factory to fulfill an order if it has all the necessary components for that SKU in its `component_inventory`. If a component is missing, the order cannot be fulfilled, which will lead to a delayed shipment and a potential stockout at the downstream location, accurately reflecting real-world production constraints.
        ---
        ### **Q: How are inventory policies like Reorder Point and EOQ calculated?**
        **A:**
        * **Reorder Point (ROP):** The ROP is a trigger to place a new order. It is calculated using the formula:
            $ROP = (\text{Average Daily Demand} \times \text{Lead Time}) + \text{Safety Stock}$
            The `Safety Stock` is determined by your chosen service level (e.g., 95%), which dictates the buffer inventory needed to prevent stockouts during the lead time.
        * **Economic Order Quantity (EOQ):** The EOQ is the optimal order quantity that minimizes the total inventory costs (holding and ordering costs). It is calculated using the formula:
            $EOQ = \sqrt{\frac{2 \times \text{Annual Demand} \times \text{Ordering Cost}}{\text{Holding Cost per Unit}}}$
        ---
        ### **Q: How can I use the cost data?**
        **A:** The app provides three key cost metrics: `Total_Holding_Cost`, `Total_Ordering_Cost`, and `Total_Stockout_Cost`. You can use these values to evaluate the performance of your inventory policy (reorder point and order quantity). The goal of a comprehensive optimization model would be to find a policy that minimizes the sum of these three costs, which represents the total cost of your supply chain operations.
        """)
