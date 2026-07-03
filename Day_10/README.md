# README

## What I learned about Data Preprocessing
I learned that data preprocessing is an important step before training a machine learning model.
It includes loading the dataset, encoding categorical data, creating new features, selecting the required features and target,
splitting the data, and scaling the features. Good preprocessing helps improve the model's performance.

## Why Train-Test Splitting is Important
Train-test splitting helps evaluate how well a model performs on new, unseen data.
The model is trained on 80% of the data and tested on the remaining 20%,
which helps avoid overfitting and gives a fair measure of the model's accuracy.

## Evaluation Metrics Used
The model was evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

## Model Performance and Observations

The Linear Regression model performed well in predicting students' average scores.
It achieved a Mean Absolute Error (MAE) of 2.53, a Mean Squared Error (MSE) of 13.46,
and an R² Score of 0.80. The predicted scores were close to the actual scores,
indicating that the model learned the relationship between the selected features and the target effectively.
Overall, the model explained about 80% of the variation in students' average scores.