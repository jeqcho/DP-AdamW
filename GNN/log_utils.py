import numpy as np

def log_uniform_percentiles(min_val: float, max_val: float, count: int):
    """
    Generate log-uniform percentile points to split a distribution.
    
    Args:
        min_val: The minimum value of the range
        max_val: The maximum value of the range
        count: Number of points to generate (including min and max)
        
    Returns:
        A list of percentile points evenly spaced in log space
    
    Example:
        >>> log_uniform_percentiles(1e-6, 1e-2, 5)
        [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    """
    if min_val <= 0 or max_val <= 0:
        raise ValueError("Both min_val and max_val must be positive for log-uniform distribution")
    
    if count < 2:
        raise ValueError("Count must be at least 2 to include min and max values")
    
    # Calculate evenly spaced points in log space
    log_min = np.log10(min_val)
    log_max = np.log10(max_val)
    
    # Generate points in log space
    log_points = np.linspace(log_min, log_max, count)
    
    # Convert back to original scale
    points = 10 ** log_points
    
    return points.tolist() 