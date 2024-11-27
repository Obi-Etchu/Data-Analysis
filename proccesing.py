import pymysql
import pandas as pd
from scipy.stats import zscore  # Importing zscore for outlier removal

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

    # Query to load data from Sales_Fact table, including associated date and category
    query = """
    SELECT Sales_Fact.*, Date_Dim.date, Product_Dim.category
    FROM Sales_Fact
    JOIN Date_Dim ON Sales_Fact.date_id = Date_Dim.date_id
    JOIN Product_Dim ON Sales_Fact.product_id = Product_Dim.product_id
    """
    data = pd.read_sql(query, conn)
   
    # Check for missing data
    print("Missing values per column:")
    print(data.isnull().sum())
   
    # Fill missing values using forward fill method
    data.fillna(method='ffill', inplace=True)
    print("\nData after filling missing values:")
    print(data.isnull().sum())  # Confirm there are no more missing values
   
    # Normalize total_amount and remove outliers based on z-score
    data['zscore_total_amount'] = zscore(data['total_amount'])
    data = data[data['zscore_total_amount'].abs() < 3]  # Retain only values within 3 standard deviations
   
    # Drop the zscore column after filtering (optional cleanup)
    data.drop(columns=['zscore_total_amount'], inplace=True)
   
    print("\nData after removing outliers:")
    print(data.describe())  # Display summary statistics for verification

    # Further processing and data mining code can go here

except pymysql.MySQLError as e:
    print(f"Error while connecting to MySQL: {e}")

finally:
    # Close the database connection if it was successfully opened
    if conn:
        conn.close()
        print("MySQL connection is closed.")