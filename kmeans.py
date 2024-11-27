import pymysql
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')  # Or 'Qt5Agg', depending on your environment
#%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Database connection details
db_config = {
    'host': 'localhost',  # Host where the database is running
    'user': 'root',    # Your database username
    'password': '',    # Your database password
    'database': 'Retaildw'  # The database you're connecting to
}

# Step 1: Connect to the MySQL database
try:
    # Establish connection
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()
    print("Connected to the database.")
   
    # Step 2: Load data from the RetailDW database
    query = """
    SELECT Sales_Fact.customer_id, Sales_Fact.product_id, Sales_Fact.quantity, Sales_Fact.total_amount, Date_Dim.date
    FROM Sales_Fact
    JOIN Date_Dim ON Sales_Fact.date_id = Date_Dim.date_id
    """
   
    # Load the data into a pandas DataFrame
    data = pd.read_sql(query, conn)
   
    # Step 3: Display the first few rows of the dataset
    print("First few rows of the dataset:")
    print(data.head())

    # Step 4: Prepare the data for clustering
    # For clustering, we need customer-level data, so we'll aggregate the data by customer_id
    customer_data = data.groupby('customer_id').agg({
        'quantity': 'sum',
        'total_amount': 'sum'
    }).reset_index()

    print("Customer data aggregated by customer_id:")
    print(customer_data.head())

    # Step 5: Data Preprocessing
    # Normalize the data (scaling) to bring the features on the same scale
    scaler = StandardScaler()
    customer_data_scaled = scaler.fit_transform(customer_data[['quantity', 'total_amount']])

    # Step 6: Elbow Method to determine the optimal number of clusters
    # We will test a range of cluster numbers to find the optimal one
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(customer_data_scaled)
        wcss.append(kmeans.inertia_)
   
    # Plotting the elbow graph to find the optimal number of clusters
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS (Within-cluster sum of squares)')
    plt.show()

    # Based on the elbow plot, choose the optimal number of clusters (let's assume it's 4 here)
    optimal_clusters = 4

    # Step 7: Apply K-Means Clustering
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
    customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

    # Step 8: Display the clustering results
    print("\nCustomer data with assigned clusters:")
    print(customer_data.head())

    # Step 9: Visualize the clustering (optional)
    plt.scatter(customer_data[customer_data['Cluster'] == 0]['quantity'], customer_data[customer_data['Cluster'] == 0]['total_amount'], s=100, c='red', label='Cluster 1')
    plt.scatter(customer_data[customer_data['Cluster'] == 1]['quantity'], customer_data[customer_data['Cluster'] == 1]['total_amount'], s=100, c='blue', label='Cluster 2')
    plt.scatter(customer_data[customer_data['Cluster'] == 2]['quantity'], customer_data[customer_data['Cluster'] == 2]['total_amount'], s=100, c='green', label='Cluster 3')
    plt.scatter(customer_data[customer_data['Cluster'] == 3]['quantity'], customer_data[customer_data['Cluster'] == 3]['total_amount'], s=100, c='yellow', label='Cluster 4')
   
    # Plot the centroids
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', label='Centroids')
   
    plt.title('Customer Clusters based on Purchasing Behavior')
    plt.xlabel('Quantity Purchased')
    plt.ylabel('Total Amount Spent')
    plt.legend()
    plt.show()

except pymysql.MySQLError as e:
    print(f"Error while connecting to MySQL: {e}")

finally:
    # Close the database connection if it was successfully opened
    if conn:
        conn.close()
        print("MySQL connection is closed.")
