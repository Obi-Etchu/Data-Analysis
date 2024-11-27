def transform_data(df):
    """Transform the data by calculating total sales and filtering records."""
    # Example transformation: calculate total sales and filter records with quantity > 5
    df['total_sales'] = df['quantity'] * df['total_amount']
    transformed_df = df[df['quantity'] > 5]
    print("Data transformed successfully")
    return transformed_df
