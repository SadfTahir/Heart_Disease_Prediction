# 🫀 Heart Disease Prediction Using Machine Learning

This project focuses on predicting the presence of heart disease using various supervised machine learning algorithms. The goal is to analyze patient health data and predict the likelihood of heart disease to assist in early diagnosis and medical decision-making.

---

## 🧠 Features

- **Exploratory Data Analysis (EDA)**
- **Feature Engineering**
- **Model Comparison across 6 algorithms**
- **Evaluation metrics (Accuracy, MSE, R², F1-Score)**
- **Logistic Regression implemented from scratch**
- **Artificial Neural Network (ANN)** performance included

---

## 🧪 Models Used

| Algorithm                        | Metric                                  |
|----------------------------------|------------------------------------------|
| Logistic Regression (from scratch) | Accuracy: 71.05%                      |
| Random Forest                    | **Accuracy: 75.71%** (Highest Accuracy)  |
| K-Nearest Neighbors (KNN)        | Accuracy: 72.02%                         |
| Linear Regression                | MSE: 0.1728, R²: 0.30                    |
| Support Vector Machine (SVM + PCA) | Accuracy: 72.55%                      |
| Artificial Neural Network (ANN) | Accuracy: 75.66%                         |

---

## 📊 ANN Classification Report
          precision    recall  f1-score   support

       0       0.78      0.76      0.77      6347
       1       0.73      0.75      0.74      5467

accuracy                           0.76     11814

macro avg 0.76 0.76 0.76 11814
weighted avg 0.76 0.76 0.76 11814

---

## 📁 Project Structure
Heart_Disease_Prediction/
│
├── heart_disease_prediction.ipynb # Main Jupyter Notebook
├── README.md # Project overview
└── models/ # (Optional) Saved models directory


---

## 📚 Libraries Used

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` (Random Forest, KNN, SVM, PCA)
- `tensorflow` / `keras` for ANN
- Custom implementation for Logistic Regression

---

## ✅ Conclusion

- **Random Forest** delivered the highest overall accuracy (75.71%)
- **ANN** also performed competitively with strong balance across classes
- Feature analysis and model evaluation confirmed the predictive power of ensemble and deep learning approaches

---

## ✨ Future Improvements

- Hyperparameter tuning (GridSearchCV)
- Feature selection via Lasso or Mutual Info
- Deployment via Flask or Streamlit
- Model interpretability using SHAP or LIME

---

## 🤝 Let's Connect

If you found this project helpful, feel free to ⭐ the repo and connect with me:

- **LinkedIn:** [Sadaf Tahir](https://www.linkedin.com/in/sadaf-tahir-884879347/)
- **GitHub:** [@SadfTahir](https://github.com/SadfTahir)

---






