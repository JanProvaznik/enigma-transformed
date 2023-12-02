from scipy.spatial.distance import jensenshannon

def js_divergence(dict1, dict2):
    """Calculate the Jensen-Shannon divergence between two dictionaries/counters.
    """
    # Get the union of keys from both dictionaries
    keys = set(dict1.keys()).union(set(dict2.keys()))
    
    # Convert frequency dictionaries to probability distributions
    total1 = sum(dict1.values())
    total2 = sum(dict2.values())
    prob_dist1 = [dict1.get(key, 0) / total1 for key in keys]
    prob_dist2 = [dict2.get(key, 0) / total2 for key in keys]
    
    # Calculate Jensen-Shannon divergence
    divergence = jensenshannon(prob_dist1, prob_dist2)
    
    return divergence