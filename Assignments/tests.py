#!/usr/bin/python3

import numpy as np

ENV_SIZE = 5

class GridWorld():

    def __init__(self, env_size):
        self.env_size = env_size
        # Initialize the value function and set terminal state value to 0
        self.V = np.zeros((env_size, env_size))
        self.terminal_states = [(2, 4), (1, 3)]
        self.V[self.terminal_states] = 0
        self.threshold = 0.01
        self.pi_str = np.full((env_size, env_size), None)
        # Define the transition probabilities and rewards
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
        self.action_description = ["Right", "Left", "Down", "Up"]
        self.gamma = 1.0  # Discount factor
        self.reward = -1  # Reward for non-terminal states
        self.pi_greedy = np.zeros((self.env_size, self.env_size), dtype=int)
    
    def is_done(self, val):
        return abs(self.V - val).max() <= self.threshold
    
    def is_terminal_state(self, i, j):
        return (i, j) in self.terminal_states
    
    def update_value_function(self, V):
        self.V = np.copy(V)

    def get_value_function(self):
        return self.V
    
    def update_pi_str(self, V):
        self.pi_str = np.copy(V)

    def get_pi_str(self):
        return self.pi_str

    def get_policy(self):
        return self.pi_greedy
    
    def print_policy(self):
        for row in self.action_description:
            print(row)

    def calculate_max_value(self, i, j):
        max_value = float('-inf')
        best_action = None
        best_actions_str = ""
        
        if self.is_terminal_state(i, j):
            return 0, None, None
        
        for action_index in range(len(self.actions)):
            next_i, next_j = self.step(action_index, i, j)
            
            if self.is_valid_state(next_i, next_j):
                value = self.get_value(next_i, next_j, self.reward)
                if value > max_value:
                    max_value = value
                    best_action = action_index
                    best_actions_str = self.action_description[action_index]
                
        self.pi_greedy[i, j] = best_action
        return max_value, best_action, best_actions_str
    
    def get_value(self, next_i, next_j, reward):
        return reward + self.V[next_i, next_j]

    def step(self, action_index, i, j):
        action = self.actions[action_index]
        return i + action[0], j + action[1]

    def is_valid_state(self, i, j):
        return 0 <= i < self.env_size and 0 <= j < self.env_size
    
    def update_greedy_policy(self):
        for i in range(ENV_SIZE):
            for j in range(ENV_SIZE):
                self.calculate_max_value(i, j)

gridworld = GridWorld(ENV_SIZE)

# Perform value iteration
num_iterations = 1000

for _ in range(num_iterations):
    val = gridworld.get_value_function()
    val_str = gridworld.get_pi_str()

    for row in range(ENV_SIZE):
        for col in range(ENV_SIZE):
            val[row, col], _, val_str[row, col] = gridworld.calculate_max_value(row, col)

    gridworld.update_value_function(val)
    gridworld.update_pi_str(val_str)

# Print the optimal value function
print("Optimal Value Function:")
print(gridworld.get_pi_str())
