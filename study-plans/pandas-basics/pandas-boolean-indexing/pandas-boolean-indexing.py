import pandas as pd

def boolean_filter(data, column, threshold):
    """
    Returns: dict with 'filtered_data' (dict) and 'count' (int)
    """
    df = pd.DataFrame(data)
    mask= df[column] > threshold
    filtered=df[mask]

    return {
        "filtered_data": filtered.to_dict(orient="list"),
        "count": len(filtered)
    }