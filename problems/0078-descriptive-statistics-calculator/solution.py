import numpy as np 
def descriptive_statistics(data):
	# Your code here
	data = np.array(data)
    
    # Calculate mean
    mean = np.mean(data)
    
    # Calculate median
    median = np.median(data)
    
    # Calculate mode (using scipy.stats.mode or manual calculation)
    # For simplicity, we'll use numpy's unique function to find mode
    values, counts = np.unique(data, return_counts=True)
    mode = values[np.argmax(counts)]
    
    # Calculate variance
    variance = np.var(data)
    
    # Calculate standard deviation
    std_dev = np.std(data)
    
    # Calculate percentiles (25th, 50th, 75th)
    percentiles = np.percentile(data, [25, 50, 75])
    
    # Calculate interquartile range (IQR)
    iqr = percentiles[2] - percentiles[0]
    
    stats_dict = {
        "mean": mean,
        "median": median,
        "mode": mode,
        "variance": np.round(variance, 4),
        "standard_deviation": np.round(std_dev, 4),
        "25th_percen