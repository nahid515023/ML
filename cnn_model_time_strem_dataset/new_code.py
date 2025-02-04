import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    print("First five rows of the dataset:")
    print(data.head())

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Scaled_Production'] = scaler.fit_transform(data[['Monthly beer production']])

    return data, scaler

# Create sequences for the CNN model
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Split data into train and test sets
def split_data(X, y, train_ratio=0.8):
    train_size = int(train_ratio * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test

# Build CNN Model
def build_cnn_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1)  # Output for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Plot predictions
def plot_predictions(y_train_actual, y_train_pred, y_test_actual, y_test_pred, sequence_length):
    plt.figure(figsize=(14, 7))
    plt.plot(y_train_actual, label='Actual Training Data', color='blue')
    plt.plot(np.arange(sequence_length, len(y_train_pred) + sequence_length),
             y_train_pred, label='Predicted Training Data', color='orange')
    plt.plot(np.arange(len(y_train_pred) + sequence_length * 2,
                       len(y_train_pred) + sequence_length * 2 + len(y_test_actual)),
             y_test_actual, label='Actual Test Data', color='green')
    plt.plot(np.arange(len(y_train_pred) + sequence_length * 2,
                       len(y_train_pred) + sequence_length * 2 + len(y_test_pred)),
             y_test_pred, label='Predicted Test Data', color='red')
    plt.title('Training vs Test Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Beer Production')
    plt.legend()
    plt.grid()
    plt.savefig('cnn_model_time_strem_dataset/output_plot.png')
    plt.show()

# Calculate R² score
def calculate_r2(y_actual, y_pred):
    r2 = r2_score(y_actual, y_pred)
    print(f"R² Score: {r2:.4f}")
    return r2

# Plot Actual vs Predicted Scatter Plot
def scatter_actual_vs_predicted(y_actual, y_pred, title):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_actual, y_pred, color='blue', alpha=0.5)
    plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'k--', lw=2)
    plt.title(title)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid()
    plt.savefig(f'cnn_model_time_strem_dataset/{title.replace(" ", "_").lower()}.png')
    plt.show()

# Plot Training and Validation Loss
def plot_loss_curve(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('cnn_model_time_strem_dataset/loss_curve.png')
    plt.show()

# Create Correlation Matrix
def create_correlation_matrix(y_actual, y_pred, title):
    data = pd.DataFrame({'Actual': y_actual.flatten(), 'Predicted': y_pred.flatten()})
    corr = data.corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.savefig(f'cnn_model_time_strem_dataset/{title.replace(" ", "_").lower()}_correlation_matrix.png')
    plt.show()

# Main execution
if __name__ == "__main__":
    file_path = '/home/nahid/Desktop/ML/cnn_model_time_strem_dataset/monthly-beer-production-in-austr.csv'
    sequence_length = 12  # Sequence length (e.g., 12 months)

    # Load and preprocess data
    data, scaler = load_and_preprocess_data(file_path)
    scaled_values = data['Scaled_Production'].values

    # Prepare sequences
    X, y = create_sequences(scaled_values, sequence_length)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Reshape for CNN
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build and train the model
    model = build_cnn_model((X_train.shape[1], 1))
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Evaluate model
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Inverse transform predictions
    y_train_pred_inv = scaler.inverse_transform(y_train_pred)
    y_test_pred_inv = scaler.inverse_transform(y_test_pred)

    y_train_actual_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot results
    plot_predictions(y_train_actual_inv, y_train_pred_inv, y_test_actual_inv, y_test_pred_inv, sequence_length)

    # Calculate R² Score
    print("Training Data:")
    r2_train = calculate_r2(y_train_actual_inv, y_train_pred_inv)

    print("Test Data:")
    r2_test = calculate_r2(y_test_actual_inv, y_test_pred_inv)

    # Plot Loss Curve
    plot_loss_curve(history)

    # Plot Scatter Actual vs Predicted
    scatter_actual_vs_predicted(y_train_actual_inv, y_train_pred_inv, "Training Data: Actual vs Predicted")
    scatter_actual_vs_predicted(y_test_actual_inv, y_test_pred_inv, "Test Data: Actual vs Predicted")

    # Correlation Matrix
    create_correlation_matrix(y_train_actual_inv, y_train_pred_inv, "Training Data Correlation Matrix")
    create_correlation_matrix(y_test_actual_inv, y_test_pred_inv, "Test Data Correlation Matrix")
