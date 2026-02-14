import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
y = data.target

malignant = (y == 0).sum()
benign = (y == 1).sum()

plt.bar(["Malignant", "Benign"], [malignant, benign])
plt.title("Breast Cancer Dataset - Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
