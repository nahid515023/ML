import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from random import choice  # Import choice for selecting random elements

# Load data
data = pd.read_csv('/home/nahid/Desktop/ML/predict_sales/sales_train.csv')
data['date'] = pd.to_datetime(data['date'], dayfirst=True)  # Fix date parsing

# Aggregate daily sales
data.set_index('date', inplace=True)
data = data['item_cnt_day'].resample('D').sum()
df = pd.DataFrame(data)

# Prepare data for LSTM
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

# Reshape for LSTM input
Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], 1)
Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], 1)

# Define configurations
activationFunctions = ['relu', 'sigmoid', 'tanh', 'softmax', 'softplus', 'softsign', 'selu', 'elu', 'exponential', 'linear']
optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam']
epochs = [50,100,150,200]
nodes = [8, 16, 32, 64, 128, 256, 512]
layers = [2, 3, 4, 5]

results = pd.DataFrame()

for _ in range(1, 20):
    # Randomly choose settings
    layer = choice(layers)
    node = choice(nodes)
    epo = choice(epochs)
    act = choice(activationFunctions)
    opt = choice(optimizers)
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(node, input_shape=(sequence_length, 1), return_sequences=(layer > 1)))

    for i in range(1, layer):
        if i == layer - 1:
            # Last LSTM layer does not need to return sequences
            model.add(LSTM(node, return_sequences=False))
        else:
            # Intermediate LSTM layers should return sequences
            model.add(LSTM(node, return_sequences=True))

    model.add(Dense(100, activation=act))  # Dense layer after LSTM layers
    model.add(Dense(1))  # Output layer
    model.compile(optimizer=opt, loss='mse')

    # Train model
    history = model.fit(Xtrain, Ytrain, epochs=epo, verbose=1, validation_data=(Xtest, Ytest))

    # Predictions
    preds = model.predict(Xtest)
    preds = scaler.inverse_transform(preds)

    # Inverse transform Ytest
    Ytest = scaler.inverse_transform(Ytest.reshape(-1, 1))

    # Evaluate model
    mse = mean_squared_error(Ytest, preds)
    mae = mean_absolute_error(Ytest, preds)
    r2 = r2_score(Ytest, preds)
    
    # Store results in DataFrame
    new_row = pd.DataFrame({
        'Activation Function': [act],
        'Optimizer': [opt],
        'Epochs': [epo],
        'Layers': [layer],
        'Nodes': [node],
        'MSE': [mse],
        'MAE': [mae],
        'R2': [r2]
    })
    print(new_row)
    results = pd.concat([results, new_row], ignore_index=True)

# Save results to CSV
results.to_csv('/model_results.csv', index=False)
