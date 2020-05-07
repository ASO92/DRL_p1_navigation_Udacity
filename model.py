import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, dueling=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            dueling (Bool): Boolean to activate dueling agent
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dueling = dueling
        
        # Layers 
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        # Layer for Dueling Agent
        self.state_value = nn.Linear(fc2_units,1)
        #self.state_value = nn.Linear(fc2_units,action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        if self.dueling:
            # advantage values + state value
            return self.fc3(x) + self.state_value(x)
        return self.fc3(x)
    

## Model Test
# model= QNetwork(state_size=3, action_size=4, seed=0)
# print(model)
# model= QNetwork(state_size=3, action_size=4, seed=0, dueling=True)
# print(model)

