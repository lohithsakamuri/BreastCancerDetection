# Breast Cancer Detection (Machine Learning)
## Live Demo (Streamlit)
https://breastcancerdetection-ycm9udnpji5x7r8scgywys.streamlit.app



This project predicts whether a breast tumor is **Malignant (Cancer)** or **Benign (Not Cancer)** using Machine Learning.

It uses the **Breast Cancer Wisconsin (Diagnostic) dataset** from Scikit-learn and trains a **Logistic Regression** model.

---

## 📌 Dataset
- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Source: `sklearn.datasets.load_breast_cancer()`
- Total samples: 569
- Features: 30 (tumor cell measurements)
- Target:
  - `0` = Malignant (Cancer)
  - `1` = Benign (Not Cancer)

---

## ⚙️ Algorithms Used
- Logistic Regression (Main model)
- Decision Tree (Comparison)
- Random Forest (Comparison)

---

## 📊 Model Evaluation
The project includes:
- Accuracy score
- Classification report
- Confusion matrix visualization

---

## 🖥️ Project Files
- `model.py` → trains the ML model and saves it
- `predict.py` → loads the saved model and predicts output
- `compare_models.py` → compares Logistic Regression, Decision Tree, Random Forest
- `confusion_plot.py` → shows confusion matrix graph
- `view_dataset.py` → prints dataset preview
- `visualize_dataset.py` → shows class distribution graph
- `app.py` → Streamlit Web App for prediction

---

## 🚀 How to Run

### 1️⃣ Install Required Libraries
```bash
pip install -r requirements.txt
