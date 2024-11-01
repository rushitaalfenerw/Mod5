Breast Cancer Analysis with PCA and Logistic Regression

This Python script performs an exploratory analysis of the breast cancer dataset using Principal Component Analysis (PCA) and Logistic Regression.
PCA is a dimensionality reduction technique that transforms a high-dimensional dataset into a lower-dimensional space while preserving most of the variance.
Functionality:

Data Loading:

Loads the breast cancer dataset from scikit-learn.
Converts the data to a pandas DataFrame with informative column names.
Data Preprocessing:

Standardizes the features using a standard scaler for better PCA performance.
Dimensionality Reduction:

Applies PCA to reduce the data to two principal components, capturing the most significant variations.
Visualization:

Creates a scatter plot of the data in the reduced PCA space, colored by class labels (benign vs. malignant).
Logistic Regression (Optional):

Splits the data into training and testing sets.
Trains a logistic regression model to predict tumor type based on the reduced features (PCA components).
Evaluates the model's accuracy on the testing set.

How to Use:

Run this script from your terminal: cancercenter.py
The script will generate a PCA plot and  print the accuracy of the logistic regression model.

Libraries:

pandas: data manipulation
numpy: numerical operations
sklearn.datasets: load datasets
sklearn.preprocessing: data preprocessing
sklearn.decomposition: dimensionality reduction with PCA
sklearn.model_selection: train-test split
sklearn.linear_model: logistic regression
sklearn.metrics: model evaluation

