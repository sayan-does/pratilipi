# app.py
import streamlit as st
import torch
import pandas as pd
from model import NCF  # Import the NCF class from model.py

# Page config
st.set_page_config(
    page_title="Story Recommendation System",
    layout="wide"
)

# Cache the model and data loading


@st.cache_resource
def load_model_and_data():
    try:
        # Load model with correct number of users and items
        model = NCF(num_users=243606, num_items=241405, factors=50)
        model.load_state_dict(torch.load('best_model.pt'))
        model.eval()

        # Load mappings and metadata
        user_mapping = torch.load('user_mapping.pt')
        item_mapping = torch.load('item_mapping.pt')
        metadata_df = pd.read_csv('metadata.csv')

        return model, user_mapping, item_mapping, metadata_df
    except Exception as e:
        st.error(f"Error loading model and data: {str(e)}")
        return None, None, None, None


def get_recommendations(user_id, model, user_mapping, item_mapping, metadata_df, top_k=5):
    try:
        # Check if user exists in mapping
        if user_id not in user_mapping:
            return None, "User ID not found in the training data"

        user_idx = user_mapping[user_id]

        # Get predictions for all items for this user
        user_tensor = torch.tensor([user_idx] * len(item_mapping))
        item_indices = torch.tensor(list(range(len(item_mapping))))

        with torch.no_grad():
            predictions = model(user_tensor, item_indices)

        # Get top k recommendations
        top_k_items = torch.topk(predictions, k=top_k)

        recommended_items = []
        for idx in top_k_items.indices:
            item_id = list(item_mapping.keys())[list(
                item_mapping.values()).index(idx.item())]
            story_info = metadata_df[metadata_df['pratilipi_id'] == item_id]
            if not story_info.empty:
                story_dict = {
                    'story_id': story_info['pratilipi_id'].iloc[0],
                    'category': story_info['category_name'].iloc[0],
                    'reading_time': story_info['reading_time'].iloc[0],
                    'published_date': story_info['published_at'].iloc[0]
                }
                recommended_items.append(story_dict)

        if not recommended_items:
            return None, "No valid recommendations found"

        return recommended_items, None

    except Exception as e:
        st.write(f"Full error: {str(e)}")
        return None, f"Error generating recommendations: {str(e)}"


def main():
    st.title("Story Recommendation System")
    st.write("This application provides personalized story recommendations based on user reading patterns. Select a user ID to get story recommendations:")

    # Load model and data
    model, user_mapping, item_mapping, metadata_df = load_model_and_data()

    if model is None:
        st.error(
            "Failed to load model and data. Please check the error message above.")
        return

    # User selection section
    with st.container():
        st.subheader("Select User")

        # Convert user_mapping keys to a sorted list for the selectbox
        user_ids = sorted(list(user_mapping.keys()))

        # Create a selectbox with all available user IDs
        selected_user_id = st.selectbox(
            "Choose a User ID",
            options=user_ids,
            # Formats how each option is displayed
            format_func=lambda x: f"User {x}"
        )

        if st.button("Get Recommendations"):
            recommendations, error = get_recommendations(
                selected_user_id,
                model,
                user_mapping,
                item_mapping,
                metadata_df
            )

            if error:
                st.error(error)
            else:
                st.subheader("Top 5 Recommended Stories")

                # Create columns for each recommendation
                cols = st.columns(4)

                for idx, story in enumerate(recommendations):
                    with cols[idx]:
                        st.write(f"Story {idx + 1}")
                        st.write(f"ID: {story['story_id']}")
                        st.write(f"Category: {story['category']}")
                        st.write(f"Reading Time: {story['reading_time']} mins")
                        st.write(f"Published: {story['published_date']}")

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit • Neural Collaborative Filtering")


if __name__ == "__main__":
    main()
