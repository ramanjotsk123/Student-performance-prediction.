# Student Performance Prediction

## Project Overview

This project uses machine learning to explore student
performance and predict final mathematics grades from
selected study-related and personal features.

## Objective

Develop a regression model that estimates final grades
using information available before the final grade is issued.

## Technologies

- Python
- Pandas
- scikit-learn
- Random Forest Regression

## Dataset

The dataset comes from the UCI Machine Learning Repository.

Source:
https://archive.ics.uci.edu/dataset/320/student-performance

It contains 649 student records from two Portuguese schools.

The target variable is G3, the final mathematics grade.
G1 and G2 are excluded from the input features.

## Project Structure

- `data/`: Dataset and attribution information
- `src/train_model.py`: Model training and evaluation
- `requirements.txt`: Required Python libraries

## Evaluation

The model reports:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
- Baseline MAE for comparison

Actual results will be documented after the model is run.

## Limitations

The dataset represents two Portuguese schools and is
relatively small. Results may not generalize to other
schools or student populations.

Predictions should not be used to make decisions about
individual students.

## Dataset Attribution

Cortez, P. (2008). Student Performance.
UCI Machine Learning Repository.
https://doi.org/10.24432/C5TG7T

Dataset licence: CC BY 4.0.
