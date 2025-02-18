import streamlit as st
import torch
import pandas as pd
import numpy as np
from torch import nn

# Define the NCF model class (same as your original implementation)


class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super(NCF, self).__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        self.fc_layers = nn.Sequential(
            nn.Linear(2 * embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        prediction = self.fc_layers(vector)
        return prediction.squeeze()

# Load model and data


@st.cache_resource
def load_model_and_mappings():
    # Load mappings
    user_mapping = torch.load('user_mapping.pt')
    item_mapping = torch.load('item_mapping.pt')

    # Initialize and load model
    model = NCF(len(user_mapping), len(item_mapping))
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()

    return model, user_mapping, item_mapping


@st.cache_data
def load_metadata():
    return pd.read_csv('metadata.csv')


# Set page config
st.set_page_config(
    page_title="Pratilipi Story Recommender",
    page_icon="📚",
    layout="wide"
)

# Main app


def main():
    st.title("📚 Pratilipi Story Recommendation System")
    st.markdown("""
    This application provides personalized story recommendations based on user reading patterns.
    Enter a user ID to get story recommendations!
    """)

    try:
        # Load model and data
        model, user_mapping, item_mapping = load_model_and_mappings()
        metadata_df = load_metadata()

        # Sidebar for user input
        st.sidebar.header("Enter User Details")
        user_id = st.sidebar.text_input("User ID", "5506791954036110")

        if st.sidebar.button("Get Recommendations"):
            try:
                user_id = int(user_id)
                if user_id in user_mapping:
                    # Generate recommendations
                    device = torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')
                    model.to(device)

                    user_idx = user_mapping[user_id]
                    user_tensor = torch.LongTensor(
                        [user_idx] * len(item_mapping)).to(device)
                    item_tensor = torch.LongTensor(
                        list(range(len(item_mapping)))).to(device)

                    with torch.no_grad():
                        predictions = model(user_tensor, item_tensor)
                        _, indices = torch.topk(predictions, 5)

                        reverse_item_mapping = {
                            v: k for k, v in item_mapping.items()}
                        recommendations = [
                            reverse_item_mapping[idx.item()] for idx in indices]

                    # Display recommendations
                    st.subheader("Top 5 Recommended Stories")
                    cols = st.columns(5)

                    for idx, (col, pratilipi_id) in enumerate(zip(cols, recommendations)):
                        with col:
                            story_info = metadata_df[metadata_df['pratilipi_id']
                                                     == pratilipi_id].iloc[0]
                            st.markdown(f"### Story {idx+1}")
                            st.markdown(f"**ID:** {pratilipi_id}")
                            st.markdown(
                                f"**Category:** {story_info['category_name']}")
                            st.markdown(
                                f"**Reading Time:** {story_info['reading_time']} mins")
                            st.markdown(
                                f"**Published:** {pd.to_datetime(story_info['published_at']).strftime('%Y-%m-%d')}")

                    # Display model metrics
                    st.subheader("Model Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MSE", "0.0465")
                    with col2:
                        st.metric("RMSE", "0.2156")
                    with col3:
                        st.metric("MAE", "0.1136")

                else:
                    st.error("User ID not found in the database!")

            except ValueError:
                st.error("Please enter a valid User ID!")

    except Exception as e:
        st.error(f"Error loading model or data: {str(e)}")
        st.info(
            "Please ensure all required files (model, mappings, and metadata) are present.")

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit • Neural Collaborative Filtering")


if __name__ == "__main__":
    main()
