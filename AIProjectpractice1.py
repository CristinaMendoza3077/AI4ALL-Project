# import csv
# import random

# # Open the CSV file
# with open('data.csv', mode='r') as file:
#     reader = csv.reader('data.csv')
    
#     # Read the header (column names)
#     header = next(reader)
    
#     # Choose 100 random column indices from the header
#     num_columns = len(header)
#     random_columns = random.sample(range(num_columns), 100)
    
#     # Create a list to store the filtered rows
#     selected_rows = []
    
#     # Add selected columns to each row
#     for row in reader:
#         selected_row = [row[i] for i in random_columns]
#         selected_rows.append(selected_row)

# # Print the header with only the selected columns
# selected_header = [header[i] for i in random_columns]
# print(selected_header)
# for row in selected_rows:
#     print(row)



#-------------------------------------------------------------------
# import csv

# # Open the CSV file
# with open('data.csv', mode='r') as file:
#     reader = csv.reader(file)
    
#     # Read the header (column names)
#     header = next(reader)
    
#     # Get the number of columns
#     num_columns = len(header)
#     print(f"Number of columns: {num_columns}")





#------------------------------------------------------------
# import csv
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

random.seed(42)

# Cargar el archivo CSV en un DataFrame
data = pd.read_csv('data.csv')
data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})

# Usamos las 30 características originales
X = data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
          'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 
          'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 
          'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 
          'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
          'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']]

Y = data['diagnosis']

# Seleccionamos 10 filas aleatorias
# random_rows = data.sample(n=10, random_state=42)

# print(random_rows)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Estandarizamos las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creamos y entrenamos el modelo de regresión logística
model = LogisticRegression(solver='liblinear', C=0.1)
model.fit(X_train_scaled, y_train)

pca = PCA(n_components=10)  # Elige el número de componentes que deseas (en este caso, 10)
X_train_pca = pca.fit_transform(X_train_scaled)  # Transformamos los datos de entrenamiento
X_test_pca = pca.transform(X_test_scaled)

# Ahora puedes predecir los valores para los datos de prueba
y_pred = model.predict(X_test_scaled)

# Predicción para un nuevo caso clínico
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

# Convierte el nuevo caso a un DataFrame
new_example = pd.DataFrame([new_example_data])

# Escala el nuevo ejemplo para que coincida con la escala del modelo
new_example_scaled = scaler.transform(new_example)

# Usa el modelo para hacer una predicción
prediction = model.predict(new_example_scaled)

# Define una frase de resultado personalizada basada en la predicción
result_phrase = "Maligno" if prediction[0] == 1 else "Benigno"
print("Predicción para el nuevo caso clínico: El tumor es", result_phrase)
