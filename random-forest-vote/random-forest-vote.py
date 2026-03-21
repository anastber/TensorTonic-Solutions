import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    arr = np.array(predictions) 
    n_samples = arr.shape[1]
    result=[]
    for i in range(n_samples):
        sample = arr[:, i]
        classes, counts = np.unique(sample, return_counts=True)
        max_count = counts.max()
        candidates = classes[counts == max_count]
        result.append(int(candidates.min()))
    return result