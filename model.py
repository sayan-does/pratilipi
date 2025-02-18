# model.py
import torch
import torch.nn as nn


class NCF(nn.Module):
    def __init__(self, num_users, num_items, factors=50):
        super().__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, factors)
        self.item_embedding = nn.Embedding(num_items, factors)

        # MLP layers using ModuleList
        self.fc_layers = nn.ModuleList([
            nn.Linear(factors * 2, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        ])

        self.sigmoid = nn.Sigmoid()

    def forward(self, user_input, item_input):
        # Get embeddings
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Concatenate embeddings
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Forward pass through MLP
        for layer in self.fc_layers:
            vector = layer(vector)

        output = self.sigmoid(vector)
        return output.squeeze()
