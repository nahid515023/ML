import numpy as np
import pandas as pd

# Define the environment size and initialize Q-values
environment_rows = 11
environment_columns = 11
q_values = np.zeros((environment_rows, environment_columns, 4))
actions = ['up', 'right', 'down', 'left']

# Define the rewards matrix
rewards = np.full((environment_rows, environment_columns), -100.)
rewards[0, 5] = 100.

# Define aisles with valid paths
aisles = {}
aisles[1] = [i for i in range(1, 10)]
aisles[2] = [1, 7, 9]
aisles[3] = [i for i in range(1, 8)]
aisles[3].append(9)
aisles[4] = [3, 7]
aisles[5] = [i for i in range(11)]
aisles[6] = [5]
aisles[7] = [i for i in range(1, 10)]
aisles[8] = [3, 7]
aisles[9] = [i for i in range(11)]

# Set rewards for aisles
for row_index in range(1, 10):
    for column_index in aisles[row_index]:
        rewards[row_index, column_index] = -1.

# Print the reward matrix using Pandas
print("Reward Matrix:")
reward_df = pd.DataFrame(rewards)
print(reward_df)

# Define helper functions
def is_terminal_state(current_row_index, current_column_index):
    return rewards[current_row_index, current_column_index] != -1.

def get_starting_location():
    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_columns)
    while is_terminal_state(current_row_index, current_column_index):
        current_row_index = np.random.randint(environment_rows)
        current_column_index = np.random.randint(environment_columns)
    return current_row_index, current_column_index

def get_next_action(current_row_index, current_column_index, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    else:
        return np.random.randint(4)

def get_next_location(current_row_index, current_column_index, action_index):
    new_row_index = current_row_index
    new_column_index = current_column_index
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
        new_column_index += 1
    elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
    return new_row_index, new_column_index

def get_shortest_path(start_row_index, start_column_index):
    if is_terminal_state(start_row_index, start_column_index):
        return []
    else:
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = [[current_row_index, current_column_index]]
        while not is_terminal_state(current_row_index, current_column_index):
            action_index = get_next_action(current_row_index, current_column_index, 1.)
            current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
            shortest_path.append([current_row_index, current_column_index])
        return shortest_path

# Training parameters
epsilon = 0.9
discount_factor = 0.9
learning_rate = 0.9

# Q-learning algorithm
for episode in range(1000):
    row_index, column_index = get_starting_location()
    while not is_terminal_state(row_index, column_index):
        action_index = get_next_action(row_index, column_index, epsilon)
        old_row_index, old_column_index = row_index, column_index
        row_index, column_index = get_next_location(row_index, column_index, action_index)
        reward = rewards[row_index, column_index]
        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value
        q_values[old_row_index, old_column_index, action_index] = old_q_value + (learning_rate * temporal_difference)

print("\nTraining complete!")

# Display shortest paths
def display_path(path):
    return " -> ".join(f"({row}, {col})" for row, col in path)

print("\nShortest Paths:")
start_points = [(3, 9), (5, 0), (9, 5), (5, 2)]
for start_row, start_col in start_points:
    path = get_shortest_path(start_row, start_col)
    if (start_row, start_col) == (5, 2):  # Reverse path for this one
        path.reverse()
        print(f"Reversed path from ({start_row}, {start_col}): {display_path(path)}")
    else:
        print(f"Path from ({start_row}, {start_col}): {display_path(path)}")
