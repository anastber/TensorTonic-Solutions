import pandas as pd

def set_index_column(data, index_col):
    """
    Returns: dict with 'index_values', 'columns', 'data'
    """
    df = pd.DataFrame(data).set_index(index_col)
    return {
        "index_values": df.index.tolist(),
        "columns": df.columns.tolist(),
        "data": df.to_dict(orient="list")
    }