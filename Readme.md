# Twitter Disaster Analysis
## Overview
- This project aims to analyze and classify tweets to determine if they are disaster or not. 
- It employs a variety of machine learning, deep learning models and ensemble methods to assess which models perform best in natural language processing and sentiment classification tasks.

## Code Structure

### 1. Installation and Imports
- **Dependencies**: Uses libraries such as **nltk, pandas, numpy, scikit-learn, seaborn, matplotlib, tensorflow.keras, imblearn, and xgboost.**
- **Installation Commands**: Necessary library installation commands are provided, commented out for optional execution as required.
### 2. Data Preprocessing
- **Loading Data**: Tweets are loaded from a CSV file tweets.csv.
- **Cleaning Data**: Removes empty rows and incorrect data types. Columns needed for analysis are converted to appropriate data types.
- **Text Preprocessing**: Includes tokenization, removal of stopwords, and lemmatization of tweet texts.
### 3. Feature Engineering
- **Vectorization**: Text data is transformed into a numerical format using TF-IDF vectorization.

### 4. Model Training and Evaluation

#### Models Trained and Evaluated:

- **Logistic Regression**: Serves as a baseline for binary classification.

- **Decision Tree Classifier**: Provides insight into feature importance and decision process.

- **Random Forest Classifier**: Used for its robustness and effectiveness in handling overfitting.

- **Gradient Boosting Classifier**: A strong learner utilizing weak learners sequentially for improved prediction accuracy.

- **Support Vector Machine (SVM)**: Effective in high-dimensional spaces typical of text data.

- **XGBoost (Extreme Gradient Boosting)**: An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.

- **Deep Learning Model (CNN)**: A convolutional neural network that captures spatial hierarchy in text data for classification.

- **Voting Classifier**: Combines predictions from the various models to improve accuracy through a majority voting system.

**AdaBoost Classifier**: Boosts the performance of decision trees by focusing on incorrectly classified instances in successive iterations.


#### Evaluation: Each model is evaluated using metrics such as accuracy, precision, recall, and F1-score.

### 5. Visualization
- **Performance Metrics Visualization**: Bar charts are used to visually compare model performance across various metrics.
### 6. Model Serialization
- **Saving Models**: Trained models are saved to disk for future use without the need for retraining.


## Input and Output
- **Input**: The input is a CSV file containing tweet text and a binary target indicating whether the tweet is related to a real disaster.

- **Output**: Outputs include performance metrics for each model, visualizations of these metrics, and serialized models.

## Achievements

- Trained and evaluated a wide array of machine learning and deep learning models, including ensemble methods and a deep learning approach.

- Visual comparison of model efficacy using precision, recall, and F1-scores.

## How to Run
- Install necessary dependencies. 

- The ipnyb and csv file should be in the same folder

- Run each cell in the notebook sequentially from data loading, preprocessing, to model evaluation.

- Examine the output metrics and visualizations directly in the notebook or from the generated files.



## Files Included
- **SMM_final_project.ipynb**: The main project notebook.
- **Dataset** : Data used.
