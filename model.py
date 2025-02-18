import torch
import torch.nn as nn


class NCF(nn.Module):
    # Changed to 50 to match second version
    def __init__(self, num_users, num_items, factors=50):
        super().__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, factors)
        self.item_embedding = nn.Embedding(num_items, factors)

        # Sequential layers modified to match second version
        self.fc_layers = nn.Sequential(
            nn.Linear(factors * 2, 128),    # Changed to 128
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()                    # Moved sigmoid into Sequential
        )

    def forward(self, user_input, item_input):
        # Get embeddings
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Concatenate embeddings
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Single forward pass through all layers (including activation functions)
        output = self.fc_layers(vector)

        return output.squeeze()
