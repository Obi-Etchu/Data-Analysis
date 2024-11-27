import mysql.connector
import pandas as pd

# Paths to your CSV files
# customer_csv = 'Customer_Dim.csv'
# product_csv = 'Product_Dim.csv'
# date_csv = 'Date_Dim.csv'
# sales_csv = 'Sales_Fact.csv'

try:
    # Connect to MySQL Database
    conn = mysql.connector.connect(
        host='localhost',
        user='root',  # Change if using a different user
        password='',  # Change if using a password
        database='Retaildw' 
        
    )
    if conn.is_connected():
        print("Connected to the database.")
        cursor = conn.cursor()

        # Test query to ensure the connection is fully functional
        cursor.execute("SELECT DATABASE();")
        db_name = cursor.fetchone()
        print(f"Connected to database: {db_name[0]}")

except mysql.connector.Error as e:
    print(f"Error while connecting to MySQL: {e}")

finally:
    if conn.is_connected():
        cursor.close()
        conn.close()
        print("MySQL connection is closed.")
