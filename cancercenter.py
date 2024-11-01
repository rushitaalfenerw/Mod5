import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Load the breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Create a PCA object with 2 components
pca = PCA(n_components=2)

# Fit the PCA model to the scaled data
pca.fit(X_scaled)

# Transform the data to the new PCA space
X_pca = pca.transform(X_scaled)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('First Principal Component(PC1)')
plt.ylabel('Second Principal Component(PC2)')
plt.title('PCA of Breast Cancer Dataset')
plt.colorbar()
plt.show()
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)