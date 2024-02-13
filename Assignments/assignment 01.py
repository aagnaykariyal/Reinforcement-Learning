import numpy as np

ENV_SIZE = 5

class GridWorld():

    def __init__(self, env_size):
        self.env_size = env_size
        # Initialize the value function and set terminal state value to 0
        self.V = np.zeros((env_size, env_size))
        self.terminal_states = [(4,4)]
        self.grey_states = [(0,4),(1,2),(3,0)]
        self.V[self.terminal_states] = 0
        self.V[self.grey_states] = -5
        self.threshold = 0.01
        self.pi_str = np.full((env_size, env_size), None)

        # Defining the transition probabilities and rewards
        self.actions = [(0,1),(0,-1),(1,0),(-1,0)] # Right, Left, Down, Up
        self.action_description = ["Right", "left", "Down", "Up"]
        self.gamma = 1.0
        self.reward = -1
        self.pi_greedy = np.zeros((self.env_size, self.env_size), dtype=int)

    # To check if there is a change in V less than the preset threshold
    def is_done(self, val):
        return abs(self.V - val).max() <= self.threshold
    
    # Returns True if the state is a terminal state
    def is_terminal_state(self, i, j):
        return (i,j) in self.terminal_states
    
    def is_grey_state(self, i, j):
        return (i, j) in self.grey_states
    
    # Overwrites the current state-value function with a new one
    def update_value_function(self, V):
        self.V = np.copy(V)

    # Returns the full state-value function V_pi
    def get_value_function(self):
        return self.V
    
    # Overwrites the current state-value function with a new one
    def update_pi_str(self, V):
        self.pi_str = np.copy(V)

    # Returns the full state-value function which consists of strings
    def get_pi_str(self):
        return self.pi_str
    
    # Returns the stored greedy policy
    def get_policy(self):
        return self.pi_greedy
    
    # Prints the policy using the action descriptions
    def print_policy(self):
        for row in self.pi_str:
            print(row)

    # Calculate the maximum value by following a greedy policy
    def calculate_max_value(self, i:int, j:int) -> (float, tuple, str):
        # Find the maximum value for the current state using Bellman's equation
        max_value = float('-inf')
        best_action = None
        best_actions_str = ""

        # Loop over all actions
        if self.is_terminal_state(i, j):
            return 0,None,'Final'
        
        if self.is_grey_state(i, j):
            return -5, None, 'Grey'
        for action_index in range(len(self.actions)):
            # Find next state
            next_i, next_j = self.step(action_index, i, j)
            # Calculate the value function if state is valid
            # Update the max_value as required
            if self.is_valid_state(next_i, next_j):
                
                value = self.get_value(next_i, next_j, self.reward)
                if value > max_value:
                    max_value = value
                    best_action = action_index
                    best_actions_str = self.action_description[action_index]

        self.pi_greedy[i,j] = best_action
        return max_value, best_action, best_actions_str
    
    def get_value(self, next_i, next_j, reward):
        return reward + self.V[next_i, next_j]
    
    def step(self, action_index, i, j):
        # Assuming a Transition Probability Matrix
        action = self.actions[action_index]
        return i + action[0], j + action[1]
    
    def is_valid_state(self, i, j):
        return 0 <= i < self.env_size and 0 <= j < self.env_size
    
    def update_greedy_policy(self):
        for i in range(ENV_SIZE):
            for j in range(ENV_SIZE):
                self.calculate_max_value(i, j)

gridworld = GridWorld(ENV_SIZE)
num_iterations = 1000

for _ in range(num_iterations):
    val = gridworld.get_value_function()
    val_str = gridworld.get_pi_str()

    for row in range(ENV_SIZE):
        for col in range(ENV_SIZE):
            val[row, col], _, val_str[row, col] = gridworld.calculate_max_value(row, col)

gridworld.update_value_function(val)
gridworld.update_pi_str(val_str)

print("Optimal Value Function:")
print(gridworld.get_pi_str())
