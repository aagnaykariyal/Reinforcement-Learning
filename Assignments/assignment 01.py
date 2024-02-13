import numpy as np

ENV_SIZE = 5

class GridWorld():

    def __init__(self, env_size):
        self.env_size = env_size
        # Initialize the value function and set terminal state value to 0
        self.V = np.zeros((env_size, env_size))
        self.terminal_states = [(4,4)] # Creating an array of terminal states
        self.grey_states = [(0,4),(1,2),(3,0)] # Creating an array consisting of grey states
        self.pi_str = np.full((env_size, env_size), None) # Creating an array for the optimal policy string
        self.pi_greedy = np.zeros((self.env_size, self.env_size), dtype=int) 

        self.V[self.terminal_states] = 10 # Assigning values to the terminal states
        self.V[self.grey_states] = -5 # Assigning values to the grey states

        self.threshold = 0.01

        # Defining the transition probabilities
        self.actions = [(0,1),(0,-1),(1,0),(-1,0)] # Right, Left, Down, Up
        self.action_description = ["Right", "left", "Down", "Up"]
        self.gamma = 1.0 # Defining the discount factor

        # Defining a reward
        self.reward = -1


    # To check if there is a change in V less than the preset threshold
    def is_done(self, val):
        return abs(self.V - val).max() <= self.threshold
    
    
    # Returns True if the state is a terminal state
    def is_terminal_state(self, i, j):
        return (i,j) in self.terminal_states
    
    
    # Returns True if the state is at a grey state
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

        # Checks if the state is a terminal state and returns values accordingly
        if self.is_terminal_state(i, j):
            return 10, 100,'Final'
        
        # Checks if the state is a grey state and returns values accordingly
        if self.is_grey_state(i, j):
            return -5, -5, 'Grey'
        
        # Loops over all actions
        for action_index in range(len(self.actions)):
            # Find next state
            next_i, next_j = self.step(action_index, i, j)

            # Calculate the value function if state is valid
            if self.is_valid_state(next_i, next_j):
                value = self.get_value(next_i, next_j, self.reward)
                
            # Update the max_value as required
                if value > max_value:
                    max_value = value
                    best_action = action_index
                    best_actions_str = self.action_description[action_index]

        return max_value, best_action, best_actions_str
    

    # Returns the value of the current state based on the reward and the current value of present in the value function
    def get_value(self, next_i, next_j, reward):
        return reward + self.V[next_i, next_j]


    # Calculates the state transition based on the given action. 
    def step(self, action_index, i, j):
        # Assuming a Transition Probability Matrix
        action = self.actions[action_index] # Getting the action / direction to move in
        return i + action[0], j + action[1] # Adding the action to the current state to indicate the new state to move in


    # Function to check if the state is a valid state inside the environment size declared 
    def is_valid_state(self, i, j):
        return 0 <= i < self.env_size and 0 <= j < self.env_size
    

    # Updating the greedy policy to find the best optimal policy
    def update_greedy_policy(self):
        # Iterating through each state
        for i in range(ENV_SIZE):
            for j in range(ENV_SIZE):
                _, best_action, _ = self.calculate_max_value(i, j) # Getting the best action to take after calculating the max value per state
                self.pi_greedy[i, j] = best_action # Assigning the best action to take per state to the pi_greedy array


gridworld = GridWorld(ENV_SIZE) # Calling the gridworld class
num_iterations = 1000 # Specifying the number of iterations to go through till the model learns the optimal policy

# Looping through the number of iterations to figure out the optimal value function
for _ in range(num_iterations):
    val = gridworld.get_value_function() # Making a copy of the value_function
    val_str = gridworld.get_pi_str() # Making a copy of the optimal policy string

    # Iterating through all the states to figure out the maximum value per state
    for row in range(ENV_SIZE):
        for col in range(ENV_SIZE):
            val[row, col], _, val_str[row, col] = gridworld.calculate_max_value(row, col)

gridworld.update_value_function(val) # Assigning the max value obtatined to the value_function per state
gridworld.update_pi_str(val_str) # Assigning the optimal policy string obtained after calculating the max_value per state

# Updating the greedy policy 
gridworld.update_greedy_policy()

# Printing out the results
print("Optimal Value Function:")
print(gridworld.get_value_function())

print("Optimal Policy:")
print('0: Right, 1: Left, 2: Down, 3: Up')
print(gridworld.get_policy())
print(gridworld.get_pi_str())
