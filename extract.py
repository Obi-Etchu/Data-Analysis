
import pandas as pd
from sqlalchemy.orm import Session

def extract_data(session: Session):
    """Extracts data from the source table."""
    query = "SELECT product_id, quantity FROM sales_fact"
    df = pd.read_sql(query, session.bind)  # Use the SQLAlchemy session's engine for querying
    print("Data extracted successfully")
    return df