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
        model = NCF(num_users=243606, num_items=241405,
                    factors=50)  # Updated numbers
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
            item_id = list(item_mapping.keys())[list(item_mapping.values()).index(idx.item())]
            story_info = metadata_df[metadata_df['story_id'] == item_id].iloc[0]
            recommended_items.append(story_info)
            
        return recommended_items, None
        
    except Exception as e:
        return None, f"Error generating recommendations: {str(e)}"

def main():
    st.title("Story Recommendation System")
    st.write("This application provides personalized story recommendations based on user reading patterns. Enter a user ID to get story recommendations:")

    # Load model and data
    model, user_mapping, item_mapping, metadata_df = load_model_and_data()
    
    if model is None:
        st.error("Failed to load model and data. Please check the error message above.")
        return

    # User input section
    with st.container():
        st.subheader("Enter User Details")
        user_id_input = st.text_input("User ID")

        if st.button("Get Recommendations"):
            if not user_id_input:
                st.warning("Please enter a User ID")
                return
                
            try:
                user_id = int(user_id_input)
            except ValueError:
                st.error("Please enter a valid numeric User ID")
                return

            recommendations, error = get_recommendations(
                user_id,
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