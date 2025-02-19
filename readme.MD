# Story Recommendation System

## Overview

This project develops a predictive model to recommend at least five Pratilipi stories to each user based on their reading history. The system leverages user-item interactions to generate personalized suggestions.

## Table of Contents

1. [Key Features](#key-features)
2. [Data Processing Steps](#data-processing-steps)
3. [Experimental Setup](#experimental-setup)
4. [Results](#results)
5. [Repository Structure](#repository-structure)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

## Key Features

* **Embedding Dimension** : Utilizes 50 factors for both user and item embeddings.
* **Network Architecture** :
* Input Layer: 100 neurons (concatenated 50-dimensional user and item embeddings).
* Hidden Layers: 128 → 64 → 32 neurons.
* Output Layer: 1 neuron with sigmoid activation.
* **Regularization** : Incorporates dropout layers with a rate of 0.2 to prevent overfitting.

## Data Processing Steps

1. **Timestamp Conversion** :

* Converted 'updated_at' and 'published_at' fields to datetime format.
* Sorted interactions chronologically.

1. **Data Organization** :

* Created user and item mappings for model input.
* Normalized reading percentages.
* Addressed missing values appropriately.

1. **Data Preparation** :

* Split data into training and testing sets.
* Constructed user-item interaction matrices.
* Generated negative samples for training.

## Experimental Setup

* **Model Configuration** :
* Embedding Size: 50
* Learning Rate: Determined through experimentation.
* Batch Size: Optimized for efficient training.
* Dropout Rate: 0.2 for regularization.
* **Training Process** :
* Employed binary cross-entropy loss.
* Implemented mini-batch training.
* Monitored training and validation metrics.
* Saved the best-performing model based on validation performance.
* **Evaluation Metrics** :
* Top-K recommendation accuracy.
* Hit rate.
* Model convergence monitoring.

## Results

* Successfully generated personalized top-5 recommendations for users.
* The model demonstrated good convergence during training.
* Effectively handled sparse user-item interactions.
* Provided a scalable solution suitable for large user bases.

## Repository Structure

* `main.py`: Main script to run the recommendation system.
* `model.py`: Contains the neural network model definition.
* `reccomendation_experiment.ipynb`: Jupyter notebook detailing the experimentation process.
* `requirements.txt`: Lists the dependencies required to run the project.
* `best_model.pt`: Saved state of the best-performing model.
* `user_mapping.pt` & `item_mapping.pt`: Serialized user and item mappings.
* `content/`: Directory containing data files.


## Plots
