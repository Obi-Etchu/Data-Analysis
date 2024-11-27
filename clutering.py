import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import mysql.connector

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="Retaildw"
)

# Query for Decision Tree Classification
classification_query = """
SELECT 
    s.customer_id,
    p.category,
    SUM(s.quantity) as total_quantity,
    SUM(s.total_amount) as total_spent,
    COUNT(DISTINCT s.sale_id) as number_of_transactions
FROM Sales_Fact s
JOIN Product_Dim p ON s.product_id = p.product_id
GROUP BY s.customer_id, p.category
"""

# Query for K-means Clustering
clustering_query = """
SELECT 
    s.customer_id,
    COUNT(DISTINCT s.sale_id) as number_of_transactions,
    SUM(s.total_amount) as total_spent,
    AVG(s.total_amount) as avg_transaction_value,
    SUM(s.quantity) as total_items_purchased
FROM Sales_Fact s
GROUP BY s.customer_id
"""

# Load data for Decision Tree
classification_data = pd.read_sql(classification_query, db)

# Prepare data for Decision Tree
X = classification_data[['total_quantity', 'total_spent', 'number_of_transactions']]
le = LabelEncoder()
y = le.fit_transform(classification_data['category'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Evaluate Decision Tree model
dt_accuracy = dt_classifier.score(X_test, y_test)
print(f"Decision Tree Accuracy: {dt_accuracy:.2f}")

# Load data for K-means
clustering_data = pd.read_sql(clustering_query, db)

# Prepare data for K-means
scaler = StandardScaler()
X_clustering = scaler.fit_transform(clustering_data.drop('customer_id', axis=1))

# Determine optimal number of clusters using elbow method
inertias = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_clustering)
    inertias.append(kmeans.inertia_)

# Apply K-means clustering
optimal_k = 4  # This should be determined from elbow curve
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clustering_data['Cluster'] = kmeans.fit_predict(X_clustering)

# Print cluster characteristics
for cluster in range(optimal_k):
    print(f"\nCluster {cluster} characteristics:")
    cluster_data = clustering_data[clustering_data['Cluster'] == cluster]
    print(cluster_data.describe().round(2))

# Function to predict product category for new customer
def predict_category(quantity, amount, transactions):
    features = np.array([[quantity, amount, transactions]])
    prediction = dt_classifier.predict(features)
    return le.inverse_transform(prediction)[0]

# Function to assign customer to cluster
def assign_cluster(transactions, total_spent, avg_transaction, total_items):
    features = np.array([[transactions, total_spent, avg_transaction, total_items]])
    scaled_features = scaler.transform(features)
    cluster = kmeans.predict(scaled_features)[0]
    return cluster

# Close database connection
db.close()