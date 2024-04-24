import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pyswarm import pso

# Load the data
data = pd.read_csv("RELIANCE_NS.csv")  # Replace with your dataset
stock_prices = data["Adj Close"].values

# Normalize the stock prices between 0 and 1
normalized_prices = (stock_prices - np.min(stock_prices)) / (np.max(stock_prices) - np.min(stock_prices))

# Define the function to minimize using PSO
def objective_function(params, *args):
    observed_prices = args[0]
    
    # Calculate predicted prices using the PSO parameters
    predicted_prices = []
    for i in range(len(observed_prices)):
        if i == 0:
            predicted_prices.append(observed_prices[0])
        else:
            predicted_price = params[0] * observed_prices[i - 1] + params[1]
            predicted_prices.append(predicted_price)
    
    # Calculate the error metrics
    mae = mean_absolute_error(observed_prices, predicted_prices)
    return mae

# Define bounds for the PSO parameters
bounds = [(0, 1), (-1, 1)]

# Initialize lists to store predicted prices and error metrics
predicted_prices = []
mae_list = []
mse_list = []
rmse_list = []

# Perform predictions using PSO
for i in range(len(normalized_prices)):
    observed_sequence = normalized_prices[:i + 1]
    
    # Use PSO to optimize the parameters
    best_params, _ = pso(objective_function, bounds, args=(observed_sequence,))
    
    # Calculate the predicted price using the optimized parameters
    if i == 0:
        predicted_price = observed_sequence[0]
    else:
        predicted_price = best_params[0] * observed_sequence[-1] + best_params[1]
    
    predicted_prices.append(predicted_price)
    
    # Calculate the error metrics
    true_price = stock_prices[i]
    mae = mean_absolute_error([true_price], [predicted_price])
    mse = mean_squared_error([true_price], [predicted_price])
    rmse = np.sqrt(mse)
    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)

# Calculate the mean error metrics
mean_mae = np.mean(mae_list)
mean_mse = np.mean(mse_list)
mean_rmse = np.mean(rmse_list)

# Print the mean error metrics
print("Mean Absolute Error:", mean_mae)
print("Mean Squared Error:", mean_mse)
print("Root Mean Squared Error:", mean_rmse)
