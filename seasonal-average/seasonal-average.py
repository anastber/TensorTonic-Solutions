def seasonal_average(series: list, period: int) -> list:
    """
    Returns the average for each position in the seasonal cycle.
    """
    result=[]
    for position in range(period):
        values=[]
        for i in range (position, len(series), period):
            values.append(series[i])
        result.append(sum(values)/len(values))

    return result
        
        
        
            