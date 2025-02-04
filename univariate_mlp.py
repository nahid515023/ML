from numpy import array
from keras.models import Sequential
from keras.layers import Dense

# Split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Define a new input sequence
raw_seq = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

# Choose a number of time steps
n_steps = 3

# Split into samples
X, y = split_sequence(raw_seq, n_steps)

# Define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit model
model.fit(X, y, epochs=2000, verbose=0)

# Demonstrate prediction with a new input sequence
x_input = array([75, 85, 95])  # Change the input to a new sequence
x_input = x_input.reshape((1, n_steps))
yhat = model.predict(x_input, verbose=0)

# Present output
print("\n### Model Prediction ###")
print(f"Input Sequence: {x_input.flatten().tolist()}")
print(f"Predicted Next Value: {yhat[0][0]:.2f}")
