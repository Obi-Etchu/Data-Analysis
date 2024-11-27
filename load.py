from sqlalchemy.orm import Session

def load_data(session: Session, transformed_df):
    """Loads the transformed data into the target table."""
    # Load data into the target table using SQLAlchemy
    transformed_df.to_sql('transformed_sales', session.bind, if_exists='replace', index=False)
    print("Data loaded successfully into transformed_sales table")



