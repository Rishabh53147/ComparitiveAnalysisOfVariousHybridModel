import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load your stock market data from a CSV file
data = pd.read_csv('RELIANCE_NS.csv')

# Select the relevant features (Open, High, Low, Volume, Close)
selected_features = ['Open', 'High', 'Low', 'Volume', 'Adj Close']
data = data[selected_features]

# Normalize the data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Define the sequence length (number of time steps to consider for prediction)
sequence_length = 10  # You can adjust this value based on your data

# Create sequences of data and labels
X = []
y = []
for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length])
    y.append(data[i+sequence_length][4])  # Using the 'Close' price as the target variable

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create the LSTM-CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, len(selected_features))))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1))  # For regression tasks

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions to get them back to the original scale
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate MAE, MSE, and RMSE
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = math.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

