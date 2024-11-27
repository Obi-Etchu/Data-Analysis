import mysql.connector
import pandas as pd

# Database connection details
try:
    # Connect to MySQL Database
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='Retaildw'
    )
    cursor = conn.cursor()
    print("Connected to the database.")

except mysql.connector.Error as e:
    print(f"Error while connecting to MySQL: {e}")

finally:
    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()
        print("MySQL connection is closed.")
