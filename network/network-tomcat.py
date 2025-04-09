import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime

# Load full dataset
data = pd.read_csv('tomcat-data.csv', sep=';')

# Preprocessing
# Convert date to datetime and extract year, month, day
data['Data'] = pd.to_datetime(data['Data'], format='%b %d, %Y')
data['year'] = data['Data'].dt.year
data['month'] = data['Data'].dt.month
data['day'] = data['Data'].dt.day

# Encode version into 3 numerical features (major, minor, patch)
def split_version(ver):
    major, minor, patch = map(int, ver.split('.'))
    return pd.Series([major, minor, patch])

data[['ver_major', 'ver_minor', 'ver_patch']] = data['Wersja'].apply(split_version)

# Convert median (string with comma) to float
data['Mediana'] = data['Mediana'].str.replace(',', '.').astype(float)

# Define features and target
X = data[['year', 'month', 'day', 'ver_major', 'ver_minor', 'ver_patch']]
y = data['Mediana']

# Optionally scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Binarize y or make it multi-class if needed
# For this case, treat it as classification: round Mediana to nearest 0.5 for class labels
y_class = (y * 2).round() / 2  # e.g., 8.1 → 8.0

# Encode y classes
le = LabelEncoder()
y_encoded = le.fit_transform(y_class)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Neural Network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y_encoded)), activation='softmax')  # Multi-class classification
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# Predict
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Evaluation
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
mae = mean_absolute_error(le.inverse_transform(y_test), le.inverse_transform(y_pred))
mse = mean_squared_error(le.inverse_transform(y_test), le.inverse_transform(y_pred))
r2 = r2_score(le.inverse_transform(y_test), le.inverse_transform(y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nPrecision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")
print("Confusion Matrix:")
print(conf_matrix)

# Predict the value for 10.1.40, Apr 8, 2025
input_version = '10.1.40'
input_date = 'Apr 8, 2025'

# Preprocess the input data
input_date = pd.to_datetime(input_date, format='%b %d, %Y')
input_year = input_date.year
input_month = input_date.month
input_day = input_date.day

# Extract version numbers
ver_major, ver_minor, ver_patch = map(int, input_version.split('.'))

# Prepare input features in the same way as the training data
input_features = np.array([[input_year, input_month, input_day, ver_major, ver_minor, ver_patch]])

# Scale the input features
input_scaled = scaler.transform(input_features)

# Make the prediction
prediction_prob = model.predict(input_scaled)
predicted_class = np.argmax(prediction_prob, axis=1)

# Convert the predicted class back to the original median value
predicted_value = le.inverse_transform(predicted_class)

print(f"\nPredicted median value for {input_version} on {input_date.strftime('%b %d, %Y')}: {predicted_value[0]}")
