import random


class QValues:
    """A table of Q-values for a discrete state and action domain"""
    def __init__(self, num_states: int, num_actions: int, discount_factor: float):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = 0.1
        self.discount_factor = discount_factor
        

        # Data Structure to represent Q Values
        # Initialized them with random values
        self.q_values = [[random.uniform(-0.01, 0.01) for _ in range(num_actions)] for _ in range(num_states)]# this will be a 2d array


    def set_learning_rate(self, learning_rate: float) -> None:
        """Sets the learning rate"""
        self.learning_rate = learning_rate


    def q(self, state: int, action: int) -> float:
        """Returns the current Q(state, action) value
        
        """
        return self.q_values[state][action]
    
    def update(self, state: int, action: int, reward: float, new_state: int):
        """Update Q(state, action) based on the observed reward (from state) and new_state
        """
        # Finding which one gives out best value
        new_value = (1-self.learning_rate) * self.q(state, action) + self.learning_rate*(reward + self.discount_factor * self.q(new_state, self.best_action(new_state)))
        self.q_values[state][action] = new_value
        

        
    def best_action(self, state: int):
        """Return the best action from the given state"""
        best_action = self.num_actions - 1
        for i in range(self.num_actions):
            if self.q(state, i) > self.q(state, best_action):
                best_action = i
        return  best_action


