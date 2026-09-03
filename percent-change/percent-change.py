def percent_change(series: list) -> list:
    """
    Returns the fractional change between consecutive values.
    """
    result=[]
    for i in range(1, len(series)):
        if series[i-1]!= 0:
            result.append((series[i]-series[i-1])/series[i-1])
        else:
            result.append(0.0)
    return result