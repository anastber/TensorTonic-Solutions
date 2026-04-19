import pandas as pd

def select_column(data, column):
    """
    Returns: dict with 'values' (list) and 'length' (int)
    """
    df= pd.DataFrame(data)
    liste=df[column].values.tolist()
    return {
        "values": liste,
        "length": len(liste),
    }