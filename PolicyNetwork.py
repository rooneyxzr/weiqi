import torch
import torch.nn as nn

class PolicyValueNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_layers=(256, 256)):
        super().__init__()
        
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], num_actions),
            nn.Softmax(dim=-1)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[1], 1)
        )

    def forward(self, x):
        policy_logits = self.policy_net(x)
        value = self.value_net(x).squeeze(-1)
        return policy_logits, value