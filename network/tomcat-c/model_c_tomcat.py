
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Wczytanie danych
data = pd.read_csv('tomcat-data.csv', sep=';')
data['Data'] = pd.to_datetime(data['Data'], format='%b %d, %Y')
data['year'] = data['Data'].dt.year
data['month'] = data['Data'].dt.month
data['day'] = data['Data'].dt.day

def split_version(ver):
    major, minor, patch = map(int, ver.split('.'))
    return pd.Series([major, minor, patch])

data[['ver_major', 'ver_minor', 'ver_patch']] = data['Wersja'].apply(split_version)
data['Mediana'] = data['Mediana'].str.replace(',', '.').astype(float)

X = data[['year', 'month', 'day', 'ver_major', 'ver_minor', 'ver_patch']]
y = data['Mediana']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_class = (y * 2).round() / 2
le = LabelEncoder()
y_encoded = le.fit_transform(y_class)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Model C
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
mae = mean_absolute_error(le.inverse_transform(y_test), le.inverse_transform(y_pred))
mse = mean_squared_error(le.inverse_transform(y_test), le.inverse_transform(y_pred))
r2 = r2_score(le.inverse_transform(y_test), le.inverse_transform(y_pred))

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")


import matplotlib.pyplot as plt
import seaborn as sns

# Wykres strat
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0)
loss_values = history.history['loss']
val_loss_values = history.history['val_loss']

plt.figure()
plt.plot(loss_values, label='train loss')
plt.plot(val_loss_values, label='val loss')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.title('Wykres funkcji strat')
plt.legend()
plt.savefig('loss.png')

# Macierz pomyłek
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Macierz pomyłek')
plt.xlabel('Predykcja')
plt.ylabel('Rzeczywiste')
plt.savefig('cmatrix.png')

# Predykcja dla Tomcat 10.1.39 z datą 2025-05-01
new_sample = pd.DataFrame([{
    'year': 2025, 'month': 5, 'day': 1,
    'ver_major': 10, 'ver_minor': 1, 'ver_patch': 39
}])
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)
pred_class = np.argmax(prediction)
print(f"Prognozowana liczba CVE (mediana) dla Tomcat 10.1.39: {le.inverse_transform([pred_class])[0]}")
