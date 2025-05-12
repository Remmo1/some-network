import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
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

# Prepare input data
X = data[['year', 'month', 'day', 'ver_major', 'ver_minor', 'ver_patch']]
y = data['Mediana']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y_class = (y * 2).round() / 2
le = LabelEncoder()
y_encoded = le.fit_transform(y_class)

# stratified split: remove class with one occurrence 
value_counts = pd.Series(y_encoded).value_counts()
valid_classes = value_counts[value_counts >= 2].index
mask = pd.Series(y_encoded).isin(valid_classes)

X_filtered = X_scaled[mask]
y_class_filtered = y_class[mask]  # używamy oryginalnych klas
le = LabelEncoder()
y_filtered = le.fit_transform(y_class_filtered)

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
)

# Better model C
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y_filtered)), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6)
]

history = model.fit(X_train, y_train, epochs=150, batch_size=8,
                    validation_split=0.2, callbacks=callbacks, verbose=1)

# Results and predictions
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

# Loss function
plt.figure()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.title('Funkcja straty')
plt.legend()
plt.grid(True)
plt.savefig('loss.png')

# Confusion matrix
cm = tf.math.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Macierz pomyłek')
plt.xlabel('Predykcja')
plt.ylabel('Rzeczywiste')
plt.savefig('cmatrix.png')

# Prediction
new_data = pd.DataFrame([{
    'year': 2025, 'month': 5, 'day': 12,
    'ver_major': 10, 'ver_minor': 1, 'ver_patch': 39
}])
new_scaled = scaler.transform(new_data)
pred_class = model.predict(new_scaled)
pred_label = le.inverse_transform(np.argmax(pred_class, axis=1))
print(f"Prognozowana mediana CVE: {pred_label[0]}")

