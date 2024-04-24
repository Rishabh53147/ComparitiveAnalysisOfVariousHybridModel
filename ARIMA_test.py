import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from hmmlearn import hmm

def prepare_observations(stock_prices):
    price_diff = np.diff(stock_prices)
    price_direction = np.where(price_diff > 0, 1, 0).reshape(-1, 1)
    return price_direction

def train_hmm_model(train_data, n_hidden_states):
    model = hmm.GaussianHMM(n_components=n_hidden_states)
    model.fit(train_data)
    return model

def predict_direction(model, test_data):
    next_direction = model.predict(test_data)
    return next_direction

def calculate_accuracy(predicted_direction, test_data):
    actual_direction = test_data
    count = 0
    for i in range(len(actual_direction)):
        try:
            if actual_direction[i] == predicted_direction[i]:
                count = count + 1
        except:
            break
    accuracy = (count / len(actual_direction)) * 100.0
    return accuracy, count

data = pd.read_csv("INFY_NS2.csv")
stock_prices = data["Adj Close"].values

price_direction = prepare_observations(stock_prices)

n_hidden_states = 2

# Split the data into training and testing sets
train_size = int(0.8 * len(data))
train_data = price_direction[:train_size]
test_data = price_direction[train_size:]

# Train the HMM model
hmm_model = train_hmm_model(train_data, n_hidden_states)

# Forecast using ARIMA
model = ARIMA(train_data, order=(5, 0, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test_data))

# Make predictions on the test set using HMM
window_size = 10
hmm_predictions = []
for i in range(len(test_data) - window_size + 1):
    sliding_window = test_data[i:i + window_size]
    predicted_direction = predict_direction(hmm_model, sliding_window.reshape(-1, 1))
    hmm_predictions.append(predicted_direction[-1])

# Calculate MAE and MSE for ARIMA
mae_arima = mean_absolute_error(test_data, forecast)
mse_arima = mean_squared_error(test_data, forecast)
rmse_arima = sqrt(mse_arima)

# Calculate accuracy for HMM
accuracy_hmm, count_hmm = calculate_accuracy(hmm_predictions, test_data)

print("ARIMA Mean Absolute Error (MAE):", mae_arima)
print("ARIMA Mean Squared Error (MSE):", mse_arima)
print("ARIMA Root Mean Squared Error (RMSE):", rmse_arima)
print("HMM Accuracy:", accuracy_hmm)
print("HMM Days Accurate:", count_hmm)
