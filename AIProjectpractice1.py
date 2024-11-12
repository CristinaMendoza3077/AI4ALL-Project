import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

random.seed(42)

# Get the file
data = pd.read_csv('data.csv')
data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})

# Use the 30 original variables
X = data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
          'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 
          'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 
          'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 
          'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
          'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']]

Y = data['diagnosis']

# Divide the data in training and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Standarize the characteristics
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# # Create and train the logistic regression model
model = LogisticRegression(solver='liblinear', C=0.1)
model.fit(X_train_scaled, y_train)

pca = PCA(n_components=10)  # Choose the number of components
X_train_pca = pca.fit_transform(X_train_scaled)  # Transform the training data
X_test_pca = pca.transform(X_test_scaled)

# Start predicting
y_pred = model.predict(X_test_scaled)

# Prediction example
new_example_data = {
    'radius_mean': 0,
    'texture_mean': 0,
    'perimeter_mean': 0,
    'area_mean': 0,
    'smoothness_mean': 0,
    'compactness_mean': 0,
    'concavity_mean': 0,
    'concave_points_mean': 0,
    'symmetry_mean': 0,
    'fractal_dimension_mean': 0,
    'radius_se': 0,
    'texture_se': 0,
    'perimeter_se': 0,
    'area_se': 0,
    'smoothness_se': 0,
    'compactness_se': 0,
    'concavity_se': 0,
    'concave_points_se': 0,
    'symmetry_se': 0,
    'fractal_dimension_se': 0,
    'radius_worst': 0,
    'texture_worst': 0,
    'perimeter_worst': 0,
    'area_worst': 0,
    'smoothness_worst': 0,
    'compactness_worst': 0.45,
    'concavity_worst': 0.47,
    'concave_points_worst': 0.1708,
    'symmetry_worst': 0.33,
    'fractal_dimension_worst': 0.1015
}

# Transfomr the new case in a DataFrame
new_example = pd.DataFrame([new_example_data])

new_example_scaled = scaler.transform(new_example)

prediction = model.predict(new_example_scaled)

# Define the result
result_phrase = "Malign" if prediction[0] == 1 else "Benign"
print("Prediction for the new case: the tumor is ", result_phrase)
