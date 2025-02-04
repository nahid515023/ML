from numpy import array
from numpy import hstack
from keras.models import Model
from keras.layers import Input, Dense

# Split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences)-1:
            break
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Define new input sequences
in_seq1 = array([5, 15, 25, 35, 45, 55, 65, 75, 85])
in_seq2 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

# Convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# Horizontally stack columns
dataset = hstack((in_seq1, in_seq2, out_seq))

# Choose a number of time steps
n_steps = 3

# Convert into input/output
X, y = split_sequences(dataset, n_steps)

# Flatten input
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))

# Separate output
y1 = y[:, 0].reshape((y.shape[0], 1))
y2 = y[:, 1].reshape((y.shape[0], 1))
y3 = y[:, 2].reshape((y.shape[0], 1))

# Define model
visible = Input(shape=(n_input,))
dense = Dense(100, activation='relu')(visible)

# Define outputs
output1 = Dense(1)(dense)
output2 = Dense(1)(dense)
output3 = Dense(1)(dense)

# Tie together
model = Model(inputs=visible, outputs=[output1, output2, output3])
model.compile(optimizer='adam', loss='mse')

# Fit model
model.fit(X, [y1, y2, y3], epochs=2000, verbose=0)

# Demonstrate prediction with new input
x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)

# Display output
print("\n### Multivariate Output MLP Prediction ###")
print(f"Input Data: {x_input.flatten().tolist()}")
print(f"Predicted Output 1: {yhat[0][0][0]:.2f}")
print(f"Predicted Output 2: {yhat[1][0][0]:.2f}")
print(f"Predicted Output 3: {yhat[2][0][0]:.2f}")
