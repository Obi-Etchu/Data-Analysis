import pandas as pd
from sqlalchemy import create_engine

# Database connection details
username = 'root'
password = ''
host = 'localhost'
database = 'datawarehouse'

# Connect to MySQL
engine = create_engine(f"mysql+mysqlconnector://{username}:{password}@{host}/{database}")

# 1. Extract
# Load data from MySQL into a pandas DataFrame
def extract_data(sales_fact):
    query = f"SELECT * FROM {sales_fact}"
    df = pd.read_sql(query, engine)
    return df

# 2. Transform
# Example transformation: Clean data, calculate new columns, handle nulls, etc.
def transform_data(df):
    # Drop rows with any null values
    df.dropna(inplace=True)

    # Example: Add a calculated column (e.g., total_score if you have 'score1' and 'score2' columns)
    if 'score1' in df.columns and 'score2' in df.columns:
        df['total_amount'] = df['quantity'] + df['total_amount']

    # Example: Standardize date formats if there's a 'date' column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Additional transformations as needed
    return df

# 3. Load
# Load transformed data back into a new table in MySQL
def load_data(df, transformed_sales):
    df.to_sql(transformed_sales, con=engine, if_exists='replace', index=False)
    print(f"Data loaded into table '{transformed_sales}'")

# Run ETL process
def run_etl(source_table, target_table):
    # Extract
    df = extract_data(source_table)
    print("Data extracted successfully.")

    # Transform
    df_transformed = transform_data(df)
    print("Data transformed successfully.")

    # Load
    load_data(df_transformed, target_table)
    print("ETL process completed successfully.")

# Specify source and target table names
source_table = 'your_source_table'
target_table = 'transformed_data'

# Execute ETL
run_etl(source_table, target_table)
