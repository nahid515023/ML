import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

# Load data
data = pd.read_csv('/home/nahid/Desktop/ML/predict_sales/sales_train.csv')
data['date'] = pd.to_datetime(data['date'], dayfirst=True)  # Fix date parsing

# Aggregate daily sales
data.set_index(['date'], inplace=True)
data = data['item_cnt_day'].resample('D').sum()
df = pd.DataFrame(data)

# Plot sales data
plt.figure(figsize=(16, 8))
plt.plot(df.index, df['item_cnt_day'], label='Daily Sales')
plt.xlabel('Date')
plt.ylabel('Number of Products Sold')
plt.title('Daily Sales Data')
plt.legend()
plt.savefig('daily_sales_plot.png')  # Save the plot

# Prepare data for CNN
values = df['item_cnt_day'].values
values = values.astype('float32').reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(values)

# Function to create sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Sequence length
sequence_length = 20

# Create sequences
X, y = create_sequences(scaled, sequence_length)


# Train-test split
train_size = int(len(X) * 0.8)
Xtrain, Xtest = X[:train_size], X[train_size:]
Ytrain, Ytest = y[:train_size], y[train_size:]

# Reshape for CNN input
Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], 1)
Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], 1)

# Build CNN model
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(sequence_length, 1)))
model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(Xtrain, Ytrain, epochs=50, verbose=1, validation_data=(Xtest, Ytest))

# Predictions
preds = model.predict(Xtest)
preds = scaler.inverse_transform(preds)

# Inverse transform Ytest
Ytest = scaler.inverse_transform(Ytest.reshape(-1, 1))


# Evaluate model using MSE, MAE, and R²
mse = mean_squared_error(Ytest, preds)
mae = mean_absolute_error(Ytest, preds)
r2 = r2_score(Ytest, preds)
print(f"Mean Squared Error (MSE) on Test Set: {mse:.4f}")
print(f"Mean Absolute Error (MAE) on Test Set: {mae:.4f}")
print(f"R² Score on Test Set: {r2:.4f}")

# # Plot actual vs predicted
# plt.figure(figsize=(20, 10))
# plt.plot(Ytest, label='Actual Sales', color='blue')
# plt.plot(preds, label='Predicted Sales', color='red')
# plt.title('Actual vs Predicted Sales')
# plt.xlabel('Time Steps')
# plt.ylabel('Sales')
# plt.legend()
# plt.savefig('actual_vs_predicted_sales.png')  # Save the plot

# # Plot Training Loss
# plt.figure(figsize=(10, 5))
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('training_validation_loss.png')  # Save the plot

