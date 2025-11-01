# 🌸 Iris Flower Classification with Multiple Models

This project uses the classic Iris dataset to compare multiple classification algorithms. It evaluates each model using accuracy, precision, recall, and F1-score, and visualizes performance with bar charts and confusion matrix heatmaps.

---

## 📂 Dataset

- Source: `sklearn.datasets.load_iris()`
- Classes: `Setosa`, `Versicolor`, `Virginica`
- Features:
  - Sepal length & width
  - Petal length & width

---

## 🧪 Workflow Overview

1. **Data Preparation**
   - Loaded Iris dataset from scikit-learn
   - Split into training and test sets (80/20)

2. **Models Compared**
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)

3. **Evaluation Metrics**
   - Accuracy
   - Precision (macro)
   - Recall (macro)
   - F1 Score (macro)

4. **Visualizations**
   - Confusion matrix heatmaps for each model
   - Bar chart comparing all metrics across models

---

## 📈 Visuals

All plots are saved in the `VISUALS/` folder:
VISUALS/ 
    ├── Logistic_Regression_Confusion_Matrix.png 
    ├── Decision_Tree_Confusion_Matrix.png 
    ├── Random_Forest_Confusion_Matrix.png 
    ├── SVM_Confusion_Matrix.png 
    ├── KNN_Confusion_Matrix.png 
    └── Classification_Metrics_Comparison.png


---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/umarbasha-ai/IRIS-FLOWER-CLASSIFICATION-PROJECT.git

# Navigate to project folder
cd IRIS-FLOWER-CLASSIFICATION-PROJECT

# Run the script
python src/iris_classification.py

Folder Structure

iris-classification/
│
├── src/
│   └── iris_classification.py
│
├── VISUALS/
│   └── [All saved plots]
│
└── README.md
