import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the data
data = pd.read_csv("TCS_NS1.csv")  # Replace with your dataset
stock_prices = data["Adj Close"].values

# Normalize the stock prices between 0 and 1
normalized_prices = (stock_prices - np.min(stock_prices)) / (np.max(stock_prices) - np.min(stock_prices))

# Define the parameters for SAHM model
window_size = 10  # Number of previous observations to consider
p = 2  # Order of the autoregressive model
q = 1  # Order of the heteroskedasticity model

# Initialize lists to store predicted prices and error metrics
predicted_prices = []
mae_list = []
mse_list = []
rmse_list = []

# Perform predictions using SAHM
for i in range(window_size, len(normalized_prices)):
    observed_sequence = normalized_prices[i - window_size : i]
    
    # Fit an autoregressive model of order p
    autoregressive_model = np.polyfit(np.arange(window_size), observed_sequence, p)
    
    # Calculate the residuals of the autoregressive model
    residuals = observed_sequence - np.polyval(autoregressive_model, np.arange(window_size))
    
    # Fit a heteroskedasticity model of order q to the residuals
    heteroskedasticity_model = np.polyfit(np.arange(window_size), residuals**2, q)
    
    # Predict the next price using the SAHM model
    predicted_residual = np.polyval(heteroskedasticity_model, window_size)  # Using the last coefficient
    predicted_price = observed_sequence[-1] + predicted_residual
    
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
