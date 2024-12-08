import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
# %matplotlib inline

url = 'https://raw.githubusercontent.com/melwinlobo18/K-Nearest-Neighbors/master/Dataset/data.csv'
df = pd.read_csv(url)  # Dataset - Breast Cancer Wisconsin Data

# Map diagnosis values to numerical values
df['diagnosis'] = df['diagnosis'].map({
    'M': 1,  # Malignant
    'B': 2   # Benign
})

# Copy 'diagnosis' to 'Class' and drop unnecessary columns
df['Class'] = df['diagnosis']
df = df.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1)

# Display the first five rows of the dataset
print(df.head())

# Visualize data points based on 'radius_mean' and 'texture_mean'
df1 = df[df.Class == 1]
df2 = df[df.Class == 2]

plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.scatter(df1['radius_mean'], df1['texture_mean'], color='green', marker='+', label='Malignant')
plt.scatter(df2['radius_mean'], df2['texture_mean'], color='blue', marker='.', label='Benign')
plt.legend()
plt.title('Breast Cancer Data: Malignant vs Benign')
plt.show()

# Prepare features (X) and target (y)
X = df.drop(['Class'], axis='columns')
y = df['Class']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Initialize the SVM model and fit to training data
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print the predictions
print("Predictions on Test Set:")
print(predictions)

# Model accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report (Precision, Recall, F1-Score)
print("Classification Report:")
print(classification_report(y_test, predictions))
