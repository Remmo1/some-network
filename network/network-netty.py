import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Embedding, Dense, Flatten, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Load data from file
data = pd.read_csv("netty-data.csv", sep=';', decimal=',')
input_strings = data.iloc[:, 0].astype(str).tolist()
output_values = data.iloc[:, 1].astype(float).values

# Feature extraction: Convert string to numerical representation
def process_string(s):
    parts = [int(part) for part in s.split('.')]
    return parts + [sum(parts)]  # Adding sum as an additional feature

input_features = np.array([process_string(s) for s in input_strings])

# Normalize outputs
scaler = MinMaxScaler()
output_values = scaler.fit_transform(output_values.reshape(-1, 1))

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in kf.split(input_features):
    train_features, test_features = input_features[train_idx], input_features[test_idx]
    train_values, test_values = output_values[train_idx], output_values[test_idx]

    # Define model
    model = keras.Sequential([
        Input(shape=(len(train_features[0]),)),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),  # L2 Regularization
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train model
    model.fit(train_features, train_values, epochs=100, batch_size=2, verbose=1,
              validation_data=(test_features, test_values), callbacks=[early_stopping])

    # Evaluate model
    test_predictions = model.predict(test_features)
    test_predictions = scaler.inverse_transform(test_predictions)
    test_values = scaler.inverse_transform(test_values)

    mae = mean_absolute_error(test_values, test_predictions)
    mse = mean_squared_error(test_values, test_predictions)
    r2 = r2_score(test_values, test_predictions)

    # Convert regression output into categorical labels
    test_values_rounded = np.round(test_values).astype(int)
    test_predictions_rounded = np.round(test_predictions).astype(int)

    # Ensure classification metrics only run if multiple unique classes exist
    if len(np.unique(test_values_rounded)) > 1:
        precision = precision_score(test_values_rounded, test_predictions_rounded, average='weighted', zero_division=1)
        recall = recall_score(test_values_rounded, test_predictions_rounded, average='weighted', zero_division=1)
        f1 = f1_score(test_values_rounded, test_predictions_rounded, average='weighted', zero_division=1)
        conf_matrix = confusion_matrix(test_values_rounded, test_predictions_rounded)
    else:
        precision, recall, f1, conf_matrix = None, None, None, None

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared Score:", r2)
    if precision is not None:
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Confusion Matrix:\n", conf_matrix)
    else:
        print("Classification metrics skipped due to lack of class variability.")

# Predict on new data
test_strings = ["4.1.124", "4.1.125"] # f1=0,720; predictions: [[5.9780254] [5.9971867]]
test_features = np.array([process_string(s) for s in test_strings])
predictions = model.predict(test_features)
predictions = scaler.inverse_transform(predictions)

print("Predictions:", predictions)
