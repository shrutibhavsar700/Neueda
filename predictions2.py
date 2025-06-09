import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
df = pd.read_csv('HCLTECH.NS.xls')

# Parse dates and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Use relevant features for multivariate prediction
features = ['Open', 'High', 'Low', 'Close', 'Volume']
target = 'Close'

# Fill missing values if any
df[features] = df[features].fillna(method='ffill')

# Normalize features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, features.index(target)])
    return np.array(X), np.array(y)

SEQ_LEN = 20  # Number of days to look back
X, y = create_sequences(scaled_data, SEQ_LEN)

# Split into train and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(features))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[es],
    verbose=1
)

# Predict on test set
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
close_scaler = MinMaxScaler()
close_scaler.min_, close_scaler.scale_ = scaler.min_[features.index(target)], scaler.scale_[features.index(target)]
y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_inv = close_scaler.inverse_transform(y_pred).flatten()

# --- Predict next 30 days ---
future_steps = 30
last_seq = scaled_data[-SEQ_LEN:].copy()
future_preds = []

for _ in range(future_steps):
    input_seq = last_seq.reshape(1, SEQ_LEN, len(features))
    next_pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
    # Prepare next input sequence
    next_row = last_seq[-1].copy()
    next_row[features.index(target)] = next_pred_scaled
    future_preds.append(next_pred_scaled)
    last_seq = np.vstack([last_seq[1:], next_row])

# Inverse transform future predictions
future_preds_inv = close_scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

# Prepare future dates
last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_steps, freq='B')  # 'B' for business days

# Plot actual vs predicted and future predictions
plt.figure(figsize=(14,7))
plt.plot(df['Date'].iloc[-len(y_test):], y_test_inv, label='Actual Close Price')
plt.plot(df['Date'].iloc[-len(y_test):], y_pred_inv, label='Predicted Close Price')
plt.plot(future_dates, future_preds_inv, label='Next 30 Days Prediction', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Actual, Predicted, and Next 30 Days Close Price')
plt.legend()
plt.tight_layout()
plt.show()

# Display predicted values in a table
result_df = pd.DataFrame({
    'Date': df['Date'].iloc[-len(y_test):].values,
    'Actual Close': y_test_inv,
    'Predicted Close': y_pred_inv
})

future_df = pd.DataFrame({
    'Date': future_dates,
    'Actual Close': [np.nan]*future_steps,
    'Predicted Close': [np.nan]*future_steps,
    'Next 30 Days Prediction': future_preds_inv
})

# Combine for display
display_df = pd.concat([result_df, future_df], ignore_index=True)
print(display_df.tail(40).to_string(index=False))