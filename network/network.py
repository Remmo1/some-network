import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_recall_fscore_support, confusion_matrix
from tensorflow.keras.layers import Dense, Input, LSTM, Reshape

# Load data from file
data = pd.read_csv("tomcat-data.csv", sep=';', decimal=',')

# Process input strings into numeric features
def process_string(s):
    parts = [int(part) for part in s.split('.')]
    return parts + [sum(parts)]  # Adding sum as an additional feature

input_features = np.array([process_string(s) for s in data.iloc[:, 0].astype(str)])
output_values = data.iloc[:, 1].astype(float).values

# Normalize inputs and outputs
input_scaler = StandardScaler()
output_scaler = StandardScaler()
input_features = input_scaler.fit_transform(input_features)
output_values = output_scaler.fit_transform(output_values.reshape(-1, 1))

# Reshape inputs for LSTM
input_features = input_features.reshape((input_features.shape[0], 1, input_features.shape[1]))

# Split data into train and test sets
split_idx = int(0.8 * len(input_features))
train_inputs, test_inputs = input_features[:split_idx], input_features[split_idx:]
train_values, test_values = output_values[:split_idx], output_values[split_idx:]

# Define LSTM model
model = keras.Sequential([
    Input(shape=(1, train_inputs.shape[2])),  # LSTM expects 3D input
    LSTM(64, activation='relu', return_sequences=True),
    LSTM(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
model.fit(train_inputs, train_values, epochs=100, batch_size=4, verbose=1, validation_data=(test_inputs, test_values))

# Evaluate model
test_predictions = model.predict(test_inputs)
test_predictions = output_scaler.inverse_transform(test_predictions)  # Convert back to original scale
test_values = output_scaler.inverse_transform(test_values)

mae = mean_absolute_error(test_values, test_predictions)
mse = mean_squared_error(test_values, test_predictions)
r2 = r2_score(test_values, test_predictions)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Convert to classification (correct if within ±0.5 range)
threshold = 0.5
y_true = (np.abs(test_values - test_predictions) <= threshold).astype(int).flatten()
y_pred = np.ones_like(y_true)  # Assume model always predicts 1 (correct)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
conf_matrix = confusion_matrix(y_true, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)

# Predict on new data
test_strings = ["10.1.40", "10.1.42"]
test_inputs = np.array([process_string(s) for s in test_strings])
test_inputs = input_scaler.transform(test_inputs)
test_inputs = test_inputs.reshape((test_inputs.shape[0], 1, test_inputs.shape[1]))
predictions = model.predict(test_inputs)
predictions = output_scaler.inverse_transform(predictions)  # Convert back to original scale

print("Predictions:", predictions)
