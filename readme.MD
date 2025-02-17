# Recommendation System Project

## Overview

This project aims to build a predictive model that recommends at least 5 pratilipis (stories) for each user based on their reading history.

## Table of Contents

1. [Data Ingestion](https://chatgpt.com/c/67b33ea1-92c8-800c-beaa-353f55e1827c#data-ingestion)
2. [Data Preprocessing](https://chatgpt.com/c/67b33ea1-92c8-800c-beaa-353f55e1827c#data-preprocessing)
3. [Exploratory Data Analysis](https://chatgpt.com/c/67b33ea1-92c8-800c-beaa-353f55e1827c#exploratory-data-analysis)
4. [Feature Engineering](https://chatgpt.com/c/67b33ea1-92c8-800c-beaa-353f55e1827c#feature-engineering)
5. [Model Selection](https://chatgpt.com/c/67b33ea1-92c8-800c-beaa-353f55e1827c#model-selection)
6. [Training &amp; Evaluation](https://chatgpt.com/c/67b33ea1-92c8-800c-beaa-353f55e1827c#training-and-evaluation)
7. [Generate Recommendations](https://chatgpt.com/c/67b33ea1-92c8-800c-beaa-353f55e1827c#generate-recommendations)
8. [Deployment &amp; Documentation](https://chatgpt.com/c/67b33ea1-92c8-800c-beaa-353f55e1827c#deployment-and-documentation)

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

1. Clone the repository:
   ```sh
   git clone <repo_url>
   cd recommendation-system
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the model training:
   ```sh
   python train.py
   ```
4. Generate recommendations:
   ```sh
   python recommend.py
   ```

## License

This project is licensed under the MIT License.
