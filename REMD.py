import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the data
data = pd.read_csv("TCS_NS1.csv")  # Replace with your dataset
stock_prices = data["Adj Close"].values

# Normalize the stock prices between 0 and 1
normalized_prices = (stock_prices - np.min(stock_prices)) / (np.max(stock_prices) - np.min(stock_prices))

# Define the parameters for REMD model
window_size = 10  # Number of previous observations to consider
alpha = 0.5  # Weight parameter

# Initialize lists to store predicted prices and error metrics
predicted_prices = []
mae_list = []
mse_list = []
rmse_list = []

# Initialize the first predicted price as the first observed price
predicted_prices.append(normalized_prices[0])

# Perform predictions using REMD
for i in range(1, len(normalized_prices)):
    observed_sequence = normalized_prices[max(0, i - window_size + 1) : i + 1]
    
    # Calculate the recursive exponential moving deviation
    remd = alpha * (normalized_prices[i] - observed_sequence[-2]) + (1 - alpha) * (predicted_prices[-1] - observed_sequence[-1])
    
    # Predict the next price using the REMD
    predicted_price = normalized_prices[i] + remd
    
    predicted_prices.append(predicted_price)
    
    # Print the predicted value for each iteration
    print(f"Predicted Price at index {i}: {predicted_price}")
    
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
