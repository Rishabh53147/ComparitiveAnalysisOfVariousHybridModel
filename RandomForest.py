import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from hmmlearn import hmm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the data
data = pd.read_csv("TCS_NS1.csv")
stock_prices = data["Adj Close"].values

# Prepare features and target for RF
window_size = 10
X_rf = []
y_rf = []
for i in range(len(stock_prices) - window_size):
    X_rf.append(stock_prices[i:i+window_size])
    y_rf.append(stock_prices[i+window_size])

X_rf = np.array(X_rf)
y_rf = np.array(y_rf)

# Split the data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(X_rf))
X_rf_train, X_rf_test = X_rf[:split_index], X_rf[split_index:]
y_rf_train, y_rf_test = y_rf[:split_index], y_rf[split_index:]

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_rf_train, y_rf_train)

# Make predictions with RF
rf_predictions = rf_model.predict(X_rf_test)

# Prepare observed sequences for HMM using RF predictions
price_diff = np.diff(rf_predictions)
price_direction = np.where(price_diff > 0, 1, 0).reshape(-1, 1)

# Train the HMM model
hmm_model = hmm.GaussianHMM(n_components=2)
hmm_model.fit(price_direction)

# Predict the next state with HMM
next_state = hmm_model.predict(price_direction[-1].reshape(1, -1))

# Use the next state to determine whether to buy, sell, or hold
if next_state == 1:
    hybrid_prediction = "Buy"
else:
    hybrid_prediction = "Sell"

# Calculate errors for RF predictions
mae_rf = mean_absolute_error(y_rf_test, rf_predictions)
mse_rf = mean_squared_error(y_rf_test, rf_predictions)
rmse_rf = np.sqrt(mse_rf)

print("Hybrid Prediction:", hybrid_prediction)
print("Random Forest Mean Absolute Error:", mae_rf)
print("Random Forest Mean Squared Error:", mse_rf)
print("Random Forest Root Mean Squared Error:", rmse_rf)
