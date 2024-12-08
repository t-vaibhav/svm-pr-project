
---

# Breast Cancer Classification using SVM

This project demonstrates the classification of breast cancer data using a Support Vector Machine (SVM) model. The dataset used is from the Breast Cancer Wisconsin dataset, which contains features derived from digitized images of fine needle aspirates (FNA) of breast mass. The goal is to predict whether a tumor is malignant or benign.

## Project Overview

The model uses the Support Vector Classifier (SVC) with a linear kernel to classify the tumor based on features such as radius and texture. The dataset is pre-processed, and the following steps are taken:
1. Mapping the diagnosis column ('M' for Malignant and 'B' for Benign) to numerical values.
2. Visualizing the data points based on two features: `radius_mean` and `texture_mean`.
3. Splitting the dataset into training and testing sets.
4. Training an SVM classifier.
5. Evaluating the model using a confusion matrix and classification report (precision, recall, and F1-score).

## Requirements

Before running the code, ensure that you have the following libraries installed:

- `pandas`
- `sklearn`
- `matplotlib`

You can install the necessary libraries using pip:

```bash
pip install pandas scikit-learn matplotlib
```

## Dataset

The dataset is retrieved from a public GitHub repository:

- **Dataset URL**: [https://raw.githubusercontent.com/melwinlobo18/K-Nearest-Neighbors/master/Dataset/data.csv](https://raw.githubusercontent.com/melwinlobo18/K-Nearest-Neighbors/master/Dataset/data.csv)

The dataset contains the following columns:
- `id`: Unique identifier for each record.
- `Unnamed: 32`: An extra column that is dropped as it contains missing values.
- `diagnosis`: The target variable (Malignant = M, Benign = B).
- Various features describing the size, shape, and texture of the tumor.

## Steps to Run

1. **Data Loading and Preprocessing**:
   - The dataset is loaded from a CSV file using `pandas`.
   - The 'diagnosis' column is mapped to numerical values (`1` for Malignant, `2` for Benign).
   - Unnecessary columns like `id`, `Unnamed: 32`, and the original `diagnosis` column are dropped.
   
2. **Data Visualization**:
   - A scatter plot is created to visualize the distribution of malignant and benign tumors based on `radius_mean` and `texture_mean`.

3. **Model Training and Testing**:
   - The dataset is split into features (`X`) and target (`y`), and further divided into training and testing sets.
   - An SVM model with a linear kernel is trained on the training set and tested on the test set.
   - Predictions are made, and the model's accuracy is calculated.

4. **Model Evaluation**:
   - The performance of the model is evaluated using a confusion matrix and a classification report, which provides metrics such as precision, recall, and F1-score.

## Output

The script outputs the following:
1. Predictions for the test set.
2. Accuracy of the model.
3. Confusion matrix showing the true vs predicted classifications.
4. A classification report containing precision, recall, and F1-score metrics.

## Example Output

```bash
Training set size: 398
Test set size: 171
Predictions on Test Set:
[2 2 2 ... 2 1 1]
Accuracy: 97.08%
Confusion Matrix:
[[ 54   3]
 [  2 112]]
Classification Report:
              precision    recall  f1-score   support

           1       0.96      0.95      0.95        57
           2       0.98      0.98      0.98       114

    accuracy                           0.97       171
   macro avg       0.97      0.97      0.97       171
weighted avg       0.97      0.97      0.97       171
```

## Conclusion

The SVM model achieves a high accuracy of around 97%, which indicates its effectiveness in classifying malignant and benign tumors. This model can serve as a foundation for further exploration and improvement in cancer diagnosis systems.

## License

This project is open-source and available under the [MIT License](LICENSE).

---

Let me know if you need further modifications or additions!
