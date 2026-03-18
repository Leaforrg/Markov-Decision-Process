import torch
import torch.nn as nn

# This network "looks" at the state and "guesses" the reward for each action
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_size, 24)
        self.layer2 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

# The Agent uses this "brain" to choose its next move in the MDP
model = DQN(state_size=4, action_size=2)
print("DQN 'Brain' initialized for complex MDP decisions.")