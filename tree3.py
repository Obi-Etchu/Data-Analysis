import pymysql
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Database connection details
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'Retaildw'
}

try:
    # Establish connection
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()
    print("Connected to the database.")

    # Load data from the RetailDW database
    query = """
    SELECT Sales_Fact.*, Date_Dim.date, Product_Dim.category
    FROM Sales_Fact
    JOIN Date_Dim ON Sales_Fact.date_id = Date_Dim.date_id
    JOIN Product_Dim ON Sales_Fact.product_id = Product_Dim.product_id
    """
    data = pd.read_sql(query, conn)
    print("First few rows of the dataset:")
    print(data.head())

    # Check for missing data and fill if necessary
    data.fillna(method='ffill', inplace=True)

    # Encode 'customer_id' and 'product_id' using LabelEncoder
    le_customer = LabelEncoder()
    le_product = LabelEncoder()
    data['customer_id_encoded'] = le_customer.fit_transform(data['customer_id'])
    data['product_id_encoded'] = le_product.fit_transform(data['product_id'])

    # Define Features and Target Variable
    X = data[['customer_id_encoded', 'product_id_encoded', 'quantity']]
    y = data['category']

    # Encode target variable if it’s categorical
    le_category = LabelEncoder()
    y = le_category.fit_transform(y)

    # Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Make Predictions
    y_pred = clf.predict(X_test)

    print('------------')
    print('Tested category:')
    print(y_test)
    print('------------')
    print('Predicted category:')
    print(y_pred)
    print('------------')

    # Calculate Accuracy and Print Results
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Classification Report
    print("\nClassification Report:")
    print(metrics.classification_report(y_test, y_pred))

    # Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(confusion_matrix)

except pymysql.MySQLError as e:
    print(f"Error while connecting to MySQL: {e}")

finally:
    if conn:
        conn.close()
        print("MySQL connection is closed.")
