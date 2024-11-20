import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def get_float_input(prompt, min_val=0, max_val=float('inf')):
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"Please enter a value between {min_val} and {max_val}")
        except ValueError:
            print("Please enter a valid number")

def collect_patient_data():
    print("\n=== Breast Cancer Detection Data Input ===")
    print("Please enter the following measurements:")
    
    parameters = {
        'radius_mean': ('Mean radius (between 0-100): ', 0, 100),
        'texture_mean': ('Mean texture (between 0-50): ', 0, 50),
        'perimeter_mean': ('Mean perimeter (between 0-500): ', 0, 500),
        'area_mean': ('Mean area (between 0-5000): ', 0, 5000),
        'smoothness_mean': ('Mean smoothness (between 0-1): ', 0, 1),
        'compactness_mean': ('Mean compactness (between 0-1): ', 0, 1),
        'concavity_mean': ('Mean concavity (between 0-1): ', 0, 1),
        'concave_points_mean': ('Mean concave points (between 0-1): ', 0, 1),
        'symmetry_mean': ('Mean symmetry (between 0-1): ', 0, 1),
        'fractal_dimension_mean': ('Mean fractal dimension (between 0-1): ', 0, 1),
        'radius_se': ('Radius SE (between 0-10): ', 0, 10),
        'texture_se': ('Texture SE (between 0-10): ', 0, 10),
        'perimeter_se': ('Perimeter SE (between 0-50): ', 0, 50),
        'area_se': ('Area SE (between 0-1000): ', 0, 1000),
        'smoothness_se': ('Smoothness SE (between 0-0.1): ', 0, 0.1),
        'compactness_se': ('Compactness SE (between 0-0.1): ', 0, 0.1),
        'concavity_se': ('Concavity SE (between 0-0.1): ', 0, 0.1),
        'concave_points_se': ('Concave points SE (between 0-0.1): ', 0, 0.1),
        'symmetry_se': ('Symmetry SE (between 0-0.1): ', 0, 0.1),
        'fractal_dimension_se': ('Fractal dimension SE (between 0-0.1): ', 0, 0.1),
        'radius_worst': ('Worst radius (between 0-100): ', 0, 100),
        'texture_worst': ('Worst texture (between 0-50): ', 0, 50),
        'perimeter_worst': ('Worst perimeter (between 0-500): ', 0, 500),
        'area_worst': ('Worst area (between 0-5000): ', 0, 5000),
        'smoothness_worst': ('Smoothness worst (between 0-1): ', 0, 1),
        'compactness_worst': ('Compactness worst (between 0-1): ', 0, 1),
        'concavity_worst': ('Concavity worst (between 0-1): ', 0, 1),
        'concave_points_worst': ('Concave points worst (between 0-1): ', 0, 1),
        'symmetry_worst': ('Symmetry worst (between 0-1): ', 0, 1),
        'fractal_dimension_worst': ('Fractal dimension worst (between 0-1): ', 0, 1)
    }
    
    data = {}
    for param, (prompt, min_val, max_val) in parameters.items():
        data[param] = get_float_input(prompt, min_val, max_val)
    
    return data

def main():
    # Initialize the model (you would typically load a pre-trained model here)
    model = LogisticRegression(solver='liblinear', C=0.1)
    scaler = StandardScaler()
    
    # Get patient data
    print("Please enter the patient's measurements:")
    patient_data = collect_patient_data()
    
    # Convert to DataFrame
    new_example = pd.DataFrame([patient_data])
    
    # Scale the input (you would typically use the same scaler used during training)
    new_example_scaled = scaler.fit_transform(new_example)
    
    # Make prediction
    prediction = model.predict(new_example_scaled)
    result_phrase = "Malignant" if prediction[0] == 1 else "Benign"
    
    print(f"\nPrediction for the patient: The tumor is {result_phrase}")
    print("\nNote: This is a demonstration model and should not be used for actual medical diagnosis.")

if __name__ == "__main__":
    main()