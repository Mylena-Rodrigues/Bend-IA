import numpy as np
import joblib

# Carregar o modelo
model = joblib.load('breast_cancer_model.h5')

def predict_sample(sample_code, thickness, size_uniformity, shape_uniformity, adhesion, epithelial_size,
                   bare_nuclei, chromatin, nucleoli, mitoses):
    sample = np.array([[sample_code, thickness, size_uniformity, shape_uniformity, adhesion, epithelial_size,
                        bare_nuclei, chromatin, nucleoli, mitoses]])
    predicted_class = model.predict(sample)[0]

    if predicted_class == 2:
        result = 'benign'
    else:
        result = 'malignant'

    return result

sample_code = 101702
thickness = 8
size_uniformity = 6
shape_uniformity = 8
adhesion = 9
epithelial_size = 9
bare_nuclei = 5
chromatin = 1
nucleoli = 5
mitoses = 3

result = predict_sample(sample_code, thickness, size_uniformity, shape_uniformity, adhesion,
                        epithelial_size, bare_nuclei, chromatin, nucleoli, mitoses)

print(f'Result: {result}')

