# Iris Flower Classification System

## What is Classification?

Classification is a Machine Learning technique used to predict categories or labels.
Instead of predicting a number, it predicts which class an object belongs to.
In this project, the model predicts the species of an Iris flower based on its features.

## Difference between Regression and Classification

| Regression                       | Classification                          |
| -------------------------------- | --------------------------------------- |
| Predicts numerical values        | Predicts categories or classes          |
| Example: Predicting house prices | Example: Predicting Iris flower species |
| Output is a number               | Output is a label                       |

## Evaluation Metrics Used

* Accuracy: Measures the overall percentage of correct predictions.
* Precision: Measures how many predicted classes are actually correct.
* Recall: Measures how many actual classes are correctly identified.
* F1-Score: Combines Precision and Recall into a single performance score.
* Confusion Matrix: Shows the number of correct and incorrect predictions for each class.

## Model Performance and Observations

The Logistic Regression model performed very well on the Iris dataset.
It achieved an Accuracy, Precision, Recall, and F1-Score of 1.0 (100%),
which means it correctly classified all the flowers in the test dataset.
The Confusion Matrix also showed no incorrect predictions.
This indicates that the Iris dataset is well-structured and that
Logistic Regression is an effective model for this classification task.
