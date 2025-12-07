import numpy as np

# Generate random data
x = np.random.normal(0, 1, 1000)

# Calculate statistics
x_min = np.min(x)
x_max = np.max(x)
x_range = x_max - x_min
x_len = len(x)

#   Calculate mean and variance
x_mean = np.mean(x)
x_var = np.var(x, ddof=1) # ddof=1 for sample variance

# Calculate median, quartiles, and spread
x_median = np.median(x)
x_quartile_lo = np.quantile(x, 0.25)  
x_quartile_hi = np.quantile(x, 0.75)
x_spread = x_quartile_hi - x_quartile_lo

# Histogram
bin_count, bin_range = np.histogram(x, bins=10)

# Print results
print(f'Min: {x_min:.2f}, Max: {x_max:.2f}, Range: {x_range:.2f}, Length: {x_len}')
print(f'Mean: {x_mean:.2f}, Variance: {x_var:.2f}')
print(f'Median: {x_median:.2f}, Quartile Low: {x_quartile_lo:.2f}, Quartile High: {x_quartile_hi:.2f}, Spread: {x_spread:.2f}')
print(f'Bin Count: {bin_count}, Bin Range: {bin_range}')

# Plot histogram
import matplotlib.pyplot as plt

plt.hist(x, bins=10, edgecolor='black')
plt.title('Histogram of x')
plt.xlabel('x')
plt.ylabel('frequency')
plt.show()