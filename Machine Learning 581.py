#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
from scipy.optimize import linprog

# Load the datasets
import pandas as pd
import numpy as np
from scipy.optimize import linprog

# Load the datasets
warehouse_loc_path = 'warehouse_loc.csv'
sales_transactions_path = 'Productlevel_Sales_Transactions_Dataset_Weekly.csv'

warehouse_loc = pd.read_csv(warehouse_loc_path)
sales_transactions = pd.read_csv(sales_transactions_path)

# Convert the FreightCost columns to numerical values by removing the '$' sign and converting to float
cost_columns = ['FreightCost100', 'FreightCost200', 'FreightCost300', 'FreightCost400', 'FreightCost500']
for col in cost_columns:
    warehouse_loc[col] = warehouse_loc[col].replace('[\$,]', '', regex=True).astype(float)

# Calculate Total_Demand by summing the weekly sales columns (Wk0 to Wk103)
weekly_columns = [col for col in sales_transactions.columns if col.startswith('Wk')]
sales_transactions['Total_Demand'] = sales_transactions[weekly_columns].sum(axis=1)

# Verify the calculated total demand
total_demand_calculated = sales_transactions['Total_Demand'].sum()
print("Calculated Total Demand:", total_demand_calculated)

# Group by Customer_Id to get the total demand per customer
total_demand_per_customer = sales_transactions.groupby('Customer_Id')['Total_Demand'].sum().reset_index()

# Merge the total demand data with the warehouse location data
merged_data = warehouse_loc.merge(total_demand_per_customer, on='Customer_Id', how='left')

# Calculate total demand after merge
total_demand_after_merge = merged_data['Total_Demand'].sum()
print("Total demand after merge:", total_demand_after_merge)

# Calculate the necessary capacity per plant to meet the total demand
num_plants = len(cost_columns)
capacity_per_plant = total_demand_after_merge / num_plants
print("Required capacity per plant to meet the total demand:", capacity_per_plant)

# Optimization Setup
plants = cost_columns
num_customers = merged_data.shape[0]

# Extract the cost matrix
cost_matrix = merged_data[plants].values

# Define the objective function (minimize cost)
c = cost_matrix.flatten()

# Define the constraints for the demand of each customer
A_eq = np.zeros((num_customers, num_customers * num_plants))
for i in range(num_customers):
    for j in range(num_plants):
        A_eq[i, i + j * num_customers] = 1

b_eq = merged_data['Total_Demand'].values

# Define the constraints for the capacity of each plant
A_ub = np.zeros((num_plants, num_customers * num_plants))
for i in range(num_plants):
    for j in range(num_customers):
        A_ub[i, i * num_customers + j] = 1

# Set the capacity of each plant to the calculated capacity per plant
b_ub = [capacity_per_plant] * num_plants

# Define bounds for the decision variables (non-negative)
x_bounds = (0, None)

# Solve the linear programming problem
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=x_bounds, method='highs')

# Check the result status and details
print("Optimization result success:", result.success)
print("Optimization result status:", result.status)
print("Optimization result message:", result.message)

# Extract the results if successful
shipping_plan = pd.DataFrame(columns=['Customer_Id', 'Plant', 'Shipped_Units', 'Cost'])

if result.success:
    x = result.x.reshape((num_plants, num_customers))
    shipping_plan_list = []
    for i in range(num_plants):
        for j in range(num_customers):
            if x[i, j] > 0:
                shipping_plan_list.append({
                    'Customer_Id': merged_data.loc[j, 'Customer_Id'],
                    'Plant': plants[i],
                    'Shipped_Units': x[i, j],
                    'Cost': merged_data.loc[j, plants[i]] * x[i, j]
                })
    shipping_plan = pd.concat([shipping_plan, pd.DataFrame(shipping_plan_list)], ignore_index=True)

# Format the Cost column as dollar figures
shipping_plan['Cost'] = shipping_plan['Cost'].apply(lambda x: "${:,.2f}".format(x))

# Display the shipping plan
print(shipping_plan)

# Initialize a new DataFrame for weekly shipments
columns = ['Plant_ID', 'Pcode', 'State', 'Customer_Id', 'Scode', 'Price'] + [f'Wk{i}' for i in range(104)]
weekly_shipments = pd.DataFrame(columns=columns)

# Populate the weekly shipments DataFrame
for _, row in shipping_plan.iterrows():
    # Find the corresponding rows in the sales_transactions DataFrame
    customer_id = row['Customer_Id']
    plant_id = row['Plant']
    shipped_units = row['Shipped_Units']
    
    # Filter the sales transactions for the specific customer and plant
    customer_data = sales_transactions[sales_transactions['Customer_Id'] == customer_id]
    if customer_data.empty:
        continue
    
    # Distribute the shipped units evenly across the 103 weeks
    weekly_units = shipped_units / 104
    
    for _, c_row in customer_data.iterrows():
        new_row = {
            'Plant_ID': plant_id,
            'Pcode': c_row['Pcode'],
            'State': c_row['State'],
            'Customer_Id': customer_id,
            'Scode': c_row['Scode'],
            'Price': c_row['Price']
        }
        
        for week in range(104):
            new_row[f'Wk{week}'] = weekly_units
        
        weekly_shipments = pd.concat([weekly_shipments, pd.DataFrame([new_row])], ignore_index=True)

# Save the weekly shipments DataFrame to an Excel file
output_path = 'weekly_shipments.xlsx'
weekly_shipments.to_excel(output_path, index=False)

print(f"Weekly shipments saved to {output_path}")

# Display the capacity of each plant
print(f"Capacity of each plant: {capacity_per_plant}")


# In[39]:


import pandas as pd
import numpy as np
from scipy.optimize import linprog

# Load the datasets
warehouse_loc_path = 'warehouse_loc.csv'
sales_transactions_path = 'Productlevel_Sales_Transactions_Dataset_Weekly.csv'

warehouse_loc = pd.read_csv(warehouse_loc_path)
sales_transactions = pd.read_csv(sales_transactions_path)

# Convert the FreightCost columns to numerical values by removing the '$' sign and converting to float
cost_columns = ['FreightCost100', 'FreightCost200', 'FreightCost300', 'FreightCost400', 'FreightCost500']
for col in cost_columns:
    warehouse_loc[col] = warehouse_loc[col].replace('[\$,]', '', regex=True).astype(float)

# Calculate Total_Demand by summing the weekly sales columns (Wk0 to Wk103)
weekly_columns = [col for col in sales_transactions.columns if col.startswith('Wk')]
sales_transactions['Total_Demand'] = sales_transactions[weekly_columns].sum(axis=1)

# Group by Customer_Id to get the total demand per customer
total_demand_per_customer = sales_transactions.groupby('Customer_Id')['Total_Demand'].sum().reset_index()

# Merge the total demand data with the warehouse location data
merged_data = warehouse_loc.merge(total_demand_per_customer, on='Customer_Id', how='left')

# Calculate total demand after merge
total_demand_after_merge = merged_data['Total_Demand'].sum()
print("Total demand after merge:", total_demand_after_merge)

# Calculate the necessary capacity per plant to meet the total demand
num_plants = len(cost_columns)
capacity_per_plant = total_demand_after_merge / num_plants
print("Required capacity per plant to meet the total demand:", capacity_per_plant)

# Optimization Setup
plants = cost_columns
num_customers = merged_data.shape[0]

# Extract the cost matrix
cost_matrix = merged_data[plants].values

# Define the objective function (minimize cost)
c = cost_matrix.flatten()

# Define the constraints for the demand of each customer
A_eq = np.zeros((num_customers, num_customers * num_plants))
for i in range(num_customers):
    for j in range(num_plants):
        A_eq[i, i + j * num_customers] = 1

b_eq = merged_data['Total_Demand'].values

# Define the constraints for the capacity of each plant
A_ub = np.zeros((num_plants, num_customers * num_plants))
for i in range(num_plants):
    for j in range(num_customers):
        A_ub[i, i * num_customers + j] = 1

# Set the capacity of each plant to the calculated capacity per plant
b_ub = [capacity_per_plant] * num_plants

# Define bounds for the decision variables (non-negative)
x_bounds = (0, None)

# Solve the linear programming problem
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=x_bounds, method='highs')

# Check the result status and details
print("Optimization result success:", result.success)
print("Optimization result status:", result.status)
print("Optimization result message:", result.message)

# Extract the results if successful
shipping_plan = pd.DataFrame(columns=['Customer_Id', 'Plant', 'Shipped_Units', 'Cost'])

if result.success:
    x = result.x.reshape((num_plants, num_customers))
    shipping_plan_list = []
    for i in range(num_plants):
        for j in range(num_customers):
            if x[i, j] > 0:
                shipping_plan_list.append({
                    'Customer_Id': merged_data.loc[j, 'Customer_Id'],
                    'Plant': plants[i],
                    'Shipped_Units': x[i, j],
                    'Cost': merged_data.loc[j, plants[i]] * x[i, j]
                })
    shipping_plan = pd.concat([shipping_plan, pd.DataFrame(shipping_plan_list)], ignore_index=True)

# Format the Cost column as dollar figures
shipping_plan['Cost'] = shipping_plan['Cost'].apply(lambda x: "${:,.2f}".format(x))

# Display the shipping plan
print(shipping_plan)

# Initialize a new DataFrame for weekly shipments
columns = ['Plant_ID', 'Pcode', 'State', 'Customer_Id', 'Scode', 'Price'] + [f'Wk{i}' for i in range(104)]
weekly_shipments = pd.DataFrame(columns=columns)

# Populate the weekly shipments DataFrame
for _, row in shipping_plan.iterrows():
    # Find the corresponding rows in the sales_transactions DataFrame
    customer_id = row['Customer_Id']
    plant_id = row['Plant']
    shipped_units = row['Shipped_Units']
    
    # Filter the sales transactions for the specific customer and plant
    customer_data = sales_transactions[sales_transactions['Customer_Id'] == customer_id]
    if customer_data.empty:
        continue
    
    # Distribute the shipped units based on the original weekly demand
    for _, c_row in customer_data.iterrows():
        new_row = {
            'Plant_ID': plant_id,
            'Pcode': c_row['Pcode'],
            'State': c_row['State'],
            'Customer_Id': customer_id,
            'Scode': c_row['Scode'],
            'Price': c_row['Price']
        }
        
        for week in weekly_columns:
            new_row[week] = c_row[week] * (shipped_units / c_row['Total_Demand'])
        
        weekly_shipments = pd.concat([weekly_shipments, pd.DataFrame([new_row])], ignore_index=True)

# Save the weekly shipments DataFrame to an Excel file
output_path = 'weekly_shipments_corrected.xlsx'
weekly_shipments.to_excel(output_path, index=False)

print(f"Weekly shipments saved to {output_path}")

# Display the capacity of each plant
print(f"Capacity of each plant: {capacity_per_plant}")


# In[44]:


import matplotlib.pyplot as plt
import seaborn as sns

# Ensure plots are shown inline in the Jupyter Notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Set up the visualization style
sns.set(style="whitegrid")

# Visualization 1: Total Demand vs. Total Capacity
def plot_demand_vs_capacity():
    plt.figure(figsize=(10, 6))
    customers = sales_transactions['Customer_Id'].unique()
    total_demand = sales_transactions.groupby('Customer_Id')['Total_Demand'].sum()
    total_capacity = pd.Series(capacity_per_plant, index=cost_columns)

    plt.bar(customers, total_demand, label='Total Demand')
    plt.bar(range(len(total_capacity)), total_capacity, label='Total Capacity per Plant')
    
    plt.xlabel('Customer/Plant')
    plt.ylabel('Units')
    plt.title('Total Demand vs. Total Capacity')
    plt.legend()
    plt.show()

# Visualization 3: Shipment Distribution
def plot_shipment_distribution():
    plt.figure(figsize=(12, 8))
    shipment_dist = shipping_plan.pivot(index='Customer_Id', columns='Plant', values='Shipped_Units').fillna(0)
    shipment_dist.plot(kind='bar', stacked=True, figsize=(14, 8))
    
    plt.xlabel('Customer ID')
    plt.ylabel('Shipped Units')
    plt.title('Shipment Distribution from Plants to Customers')
    plt.legend(title='Plant')
    plt.show()

# Visualization 4: Weekly Shipments Before and After Optimization
def plot_weekly_shipments(customer_id):
    plt.figure(figsize=(14, 8))
    original_weekly = sales_transactions[sales_transactions['Customer_Id'] == customer_id][weekly_columns].sum()
    optimized_weekly = weekly_shipments[weekly_shipments['Customer_Id'] == customer_id][weekly_columns].sum()
    
    plt.plot(original_weekly, label='Original Weekly Shipments')
    plt.plot(optimized_weekly, label='Optimized Weekly Shipments')
    
    plt.xlabel('Week')
    plt.ylabel('Shipped Units')
    plt.title(f'Weekly Shipments for Customer {customer_id} Before and After Optimization')
    plt.legend()
    plt.show()

# Visualization 5: Cost Efficiency
def plot_cost_efficiency():
    plt.figure(figsize=(10, 6))
    cost_per_unit = shipping_plan.copy()
    
    # Debugging: Print raw cost values
    print("Raw cost values:", cost_per_unit['Cost'])
    
    cost_per_unit['Cost'] = cost_per_unit['Cost'].apply(lambda x: float(x.replace('$', '').replace(',', '')))
    
    # Debugging: Print converted cost values
    print("Converted cost values:", cost_per_unit['Cost'])
    
    cost_per_unit['Cost per Unit'] = cost_per_unit['Cost'] / cost_per_unit['Shipped_Units']
    
    sns.histplot(cost_per_unit['Cost per Unit'], kde=True)
    
    plt.xlabel('Cost per Unit Shipped ($)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Cost per Unit Shipped')
    plt.show()

# Plot the visualizations
plot_demand_vs_capacity()
plot_shipment_distribution()
plot_weekly_shipments(customer_id=1)  # Example customer ID
plot_cost_efficiency()


# In[ ]:




