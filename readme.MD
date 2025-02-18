# Story Recommendation System

## Overview

This project aims to build a predictive model that recommends at least 5 pratilipis (stories) for each user based on their reading history.

## Table of Contents

1. Data Ingestion
2. Data Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Selection
6. Training & Evaluation
7. Generate Recommendations
8. Deployment & Documentation

## 1. Data Ingestion

* Load `User_interaction.csv`
* Load `Meta_data.csv`

## 2. Data Preprocessing

* Handle missing values
* Convert timestamps
* Normalize `read_percentage`
* Merge datasets on `pratilipi_id`

## 3. Exploratory Data Analysis (EDA)

* Analyze user reading behavior patterns
* Study story popularity distribution
* Identify genre-based trends

## 4. Feature Engineering

* Extract user preferences (favorite genres, authors)
* Compute story similarity (based on metadata)
* Identify time-based trends

## 5. Model Selection

* **Collaborative Filtering:** SVD, ALS
* **Content-Based Filtering:** TF-IDF, Cosine Similarity
* **Hybrid Approach:** Combination of both methods

## 6. Training & Evaluation

* Train model using **75% of data**
* Validate on **25% test data**
* Evaluate using:
  * Precision@5
  * Recall@5
  * Mean Average Precision (MAP)

## 7. Generate Recommendations

* Predict **top 5 pratilipis** per user
* Rank recommendations based on relevance

## 8. Deployment & Documentation

* Save trained model
* Create API or script for inference
* Write **README & project report**

## Installation & Usage

### Prerequisites

Ensure you have the following installed:

* Python 3.8+
* pip
* Virtual environment (recommended, but optional)

### Step 1: Clone the Repository

```bash
git clone <repo_url>
cd recommendation-system
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Ensure Model and Data Files are Present

Place the following required files in the project directory:

* `best_model.pt`  (Pretrained NCF model)
* `user_mapping.pt` (Mapping of user IDs)
* `item_mapping.pt` (Mapping of item IDs)
* `metadata.csv` (Metadata about the stories)

### Step 4: Run the Model Training

```bash
python train.py
```

### Step 5: Generate Recommendations

```bash
python recommend.py
```

## Sample Screenshots

*Add screenshots of the application interface here.*

## Demo Video

*Add a link or embed a demo video here.*

## Technologies Used

* **Streamlit** - Web UI framework
* **PyTorch** - Model inference
* **Pandas** - Data handling

## License

This project is licensed under the MIT License.

## Contact

For any queries, contact [your-email@example.com](mailto:your-email@example.com)
