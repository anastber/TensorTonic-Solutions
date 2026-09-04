def user_based_cf_prediction(similarities: list, ratings: list) -> float:
    """
    Returns the positive-similarity weighted rating prediction.
    """
    prediction = 0.0
    sum_sim=0.0
    for i,similarity in enumerate(similarities):
        if similarity>0:
            prediction+=similarities[i]*ratings[i]
            sum_sim+=similarities[i]
    if sum_sim!=0:
        return prediction/sum_sim
    return 0.0