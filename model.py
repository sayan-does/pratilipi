import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, num_users, num_items, factors=20):
        super().__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, factors)
        self.item_embedding = nn.Embedding(num_items, factors)

        # Sequential layers with exact naming to match saved state
        self.fc_layers = nn.Sequential(
            nn.Linear(factors * 2, 64),     # fc_layers.0
            nn.Linear(64, 32),              # fc_layers.3
            nn.Linear(32, 16),              # fc_layers.6
            nn.Linear(16, 1)                # fc_layers.8
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, user_input, item_input):
        # Get embeddings
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Concatenate embeddings
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Forward pass through layers with ReLU activation between linear layers
        # All except last layer
        for i, layer in enumerate(self.fc_layers[:-1]):
            vector = layer(vector)
            vector = torch.relu(vector)

        # Final layer with sigmoid
        vector = self.fc_layers[-1](vector)
        output = self.sigmoid(vector)

        return output.squeeze()
