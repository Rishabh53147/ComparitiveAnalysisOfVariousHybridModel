import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import deque

# Load the data
data = pd.read_csv("TCS_NS1.csv")  # Replace with your dataset
stock_prices = data["Adj Close"].values

# Normalize the stock prices between 0 and 1
normalized_prices = (stock_prices - np.min(stock_prices)) / (np.max(stock_prices) - np.min(stock_prices))

# Define the parameters for RDAWA model
window_size = 10  # Number of previous observations to consider
alpha = 0.5  # Weight parameter

# Initialize a deque to store the window of past observations
observation_window = deque(maxlen=window_size)
for i in range(window_size):
    observation_window.append(normalized_prices[i])

# Initialize lists to store predicted prices and error metrics
predicted_prices = []
mae_list = []
mse_list = []
rmse_list = []
correct_predictions = 0  # Counter for correct predictions

# Perform predictions using RDAWA
for i in range(window_size, len(normalized_prices)):
    observed_sequence = np.array(observation_window)
    
    # Calculate the weighted average
    weighted_average = np.sum(observed_sequence * np.arange(1, window_size + 1)) / np.sum(np.arange(1, window_size + 1))
    
    # Predict the next price using the weighted average
    predicted_price = weighted_average * (np.max(stock_prices) - np.min(stock_prices)) + np.min(stock_prices)
    predicted_prices.append(predicted_price)
    
    # Calculate the error metrics
    true_price = stock_prices[i]
    mae = mean_absolute_error([true_price], [predicted_price])
    mse = mean_squared_error([true_price], [predicted_price])
    rmse = np.sqrt(mse)
    mae_list.append(mae)
    mse_list.append(mse)
    rmse_list.append(rmse)
    
    # Update the observation window with the latest observation
    observation_window.append(normalized_prices[i])

    # Check if the prediction is correct within a tolerance (adjust as needed)
    tolerance = 0.02  # Adjust this value based on the scale of your stock prices
    if abs(predicted_price - true_price) <= tolerance:
        correct_predictions += 1

# Calculate the mean error metrics
mean_mae = np.mean(mae_list)
mean_mse = np.mean(mse_list)
mean_rmse = np.mean(rmse_list)

# Calculate accuracy percentage
accuracy_percentage = (correct_predictions / (len(normalized_prices) - window_size)) * 100

# Print the mean error metrics and accuracy percentage
print("Mean Absolute Error:", mean_mae)
print("Mean Squared Error:", mean_mse)
print("Root Mean Squared Error:", mean_rmse)
print("Accuracy Percentage:", accuracy_percentage, "%")

# Plotting the actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(stock_prices[window_size:], label='Actual Prices', color='blue')
plt.plot(predicted_prices, label='Predicted Prices', color='red', linestyle='dashed')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
