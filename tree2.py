import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import mysql.connector
import matplotlib.pyplot as plt

# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="Retaildw"
)

# Simplified query to start with
query = """
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

# Load data
cursor = db.cursor()
cursor.execute(query)
results = cursor.fetchall()

# Convert to DataFrame
data = pd.DataFrame(results, columns=['customer_id', 'category', 'total_quantity', 
                                    'total_spent', 'number_of_transactions'])

# Data cleaning
def clean_data(df):
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with median
    for column in df.select_dtypes(include=[np.number]).columns:
        df[column] = df[column].fillna(df[column].median())
    
    # Remove extreme outliers (values beyond 3 standard deviations)
    for column in df.select_dtypes(include=[np.number]).columns:
        mean = df[column].mean()
        std = df[column].std()
        df = df[df[column].between(mean - 3*std, mean + 3*std)]
    
    return df

# Clean the data
data = clean_data(data)

# Prepare features and target
X = data[['total_quantity', 'total_spent', 'number_of_transactions']]
le = LabelEncoder()
y = le.fit_transform(data['category'])

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt_classifier = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10
)

# Fit the model
dt_classifier.fit(X_train, y_train)

# Evaluate model
train_accuracy = dt_classifier.score(X_train, y_train)
test_accuracy = dt_classifier.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualize decision tree
plt.figure(figsize=(10,10))
plot_tree(dt_classifier, feature_names=X.columns, class_names=le.classes_, filled=True,fontsize=6)
plt.show()

# Function to predict category for new customers
def predict_category(quantity, amount, transactions):
    # Scale the input features
    features = scaler.transform([[quantity, amount, transactions]])
    prediction = dt_classifier.predict(features)
    return le.inverse_transform(prediction)[0]

# Example prediction
try:
    example_prediction = predict_category(10, 1000, 5)
    print(f"\nPredicted category: {example_prediction}")
except Exception as e:
    print(f"Prediction error: {e}")

db.close()