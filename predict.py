import joblib
from sklearn.datasets import load_breast_cancer

model = joblib.load("breast_model.pkl")
scaler = joblib.load("scaler.pkl")

data = load_breast_cancer()
X = data.data

sample = X[0].reshape(1, -1)
sample_scaled = scaler.transform(sample)

pred = model.predict(sample_scaled)

if pred[0] == 0:
    print("Prediction: Malignant (Cancer)")
else:
    print("Prediction: Benign (Not Cancer)")
