
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Replace 'your_file.csv' with the actual name of your CSV file
df = pd.read_csv('data/titanic_train.csv')
df.columns = df.columns.str.strip().str.lower()
#df.columns = df.columns.str.lower()
print(df.columns)

# Print the first 5 rows of the dataframe
print(df.head())
#We retrieve its summary info via '.info()'
df.info()

# Here we 'print' the first 10 rows via '.head(10)'
print(df.head(10))

# Here we use '.isnull()' to retrieve missing values
# We follow by retrieving summary count via 'sum()'
df.isnull().sum()

# We use '.describe()' to perform summary statistic analysis
print(df.describe())

# Here '.corr(numeric_only=True)' retrieves numerical correlations
print(df.corr(numeric_only=True))

# We assign our attributes to the scatter_matrix and figure sizes.
attributes = ['age', 'fare', 'pclass']
scatter_matrix(df[attributes], figsize=(10, 10))

# We use our attributes to create scatterplots.
plt.scatter(df['age'], df['fare'], c=df['sex'].apply(lambda x: 0 if x == 'male' else 1))
plt.xlabel('age')
plt.ylabel('fare')
plt.title('age vs fare by Gender')
plt.show()

#We create a histogram of age.
sns.histplot(df['age'], kde=True)
plt.title('age Distribution')
plt.show()

#We create a count plot for class and survival.
sns.countplot(x='pclass', hue='survived', data=df)
plt.title('Class Distribution by Survival')
plt.show()

# age was missing values. 
# We can impute missing values for age using the median:
df.loc[:, 'age'] = df['age'].fillna(df['age'].median())
 
# embarked was missing values.
# We can drop missing values for embarked (or fill with mode):
df.loc[:, 'embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# We create a new feature: Family size
df['family_size'] = df['sibsp'] + df['parch'] + 1

# We convert categorical data to numeric.
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# We create a binary feature for 'Alone'.
# Create 'Alone' column: 1 if no siblings/spouse or parents/children aboard, else 0
df['Alone'] = ((df['sibsp'] + df['parch']) == 0).astype(int)
#df['Alone'] = df['Alone'].astype(int)

# Here we assign inputs to "x" and 
# target variables to "y".
X = df[['age', 'fare', 'pclass', 'sex']]
y = df['survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Here we split the data for training and testing
# 'X_train' and 'X_test'.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print('Train size:', len(X_train))
print('Test size:', len(X_test))

# Here we assign indices to stratify our test & train data.
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)

for train_indices, test_indices in splitter.split(X, y):
    train_set = X.iloc[train_indices]
    test_set = X.iloc[test_indices]

print('Train size:', len(train_set))
print('Test size:', len(test_set))

# Here we print our findings to compare results.
print("Original Class Distribution:\n", y.value_counts(normalize=True))
print("")
# Basic Train/Test:
print("Basic Train/Test Method")
print("Train Set Class Distribution:\n", X_train['pclass'].value_counts(normalize=True))
print("Test Set Class Distribution:\n", X_test['pclass'].value_counts(normalize=True))
print("")
# Stratified Train/Test
print("Stratified Train/Test Method")
print("Train Set Class Distribution:\n", train_set['pclass'].value_counts(normalize=True))
print("Test Set Class Distribution:\n", test_set['pclass'].value_counts(normalize=True))

# Case 2: age only
#x = titanic[['age']]
#y = titanic['survived']

# Split the data into training and testing sets
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
for train_indices, test_indices in splitter.split(X, y):
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

print('Train size: ', len(X_train), 'Test size: ', len(X_test))

# Create and train a Decision Tree model
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

# Evaluate the model on training data
y_pred = tree_model.predict(X_train)
print("Results for Decision Tree on training data:")
print(classification_report(y_train, y_pred))

# Evaluate the model on test data
y_test_pred = tree_model.predict(X_test)
print("Results for Decision Tree on test data:")
print(classification_report(y_test, y_test_pred))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot the decision tree
fig = plt.figure(figsize=(25,10))
plot_tree(tree_model, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True)
plt.show()
fig.savefig("decision_tree_titanic.png")

#This function calculates the R^2 of the trained data.
#r2 = r2_score(y_test, y_pred)
#print(f'R²: {r2:.2f}')
#This function calculates the MAE for the trained data.
#mae = mean_absolute_error(y_test, y_pred)
#print(f'MAE: {mae:.2f}')
#This function calculates the RMSE of the trained data.
#rmse = root_mean_squared_error(y_test, y_pred)
#print(f'RMSE: {rmse:.2f}')

from sklearn.metrics import accuracy_score

# Predict on the test set
y_test_pred = tree_model.predict(X_test)

# Use classification metrics
print("Classification Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_test_pred):.2f}")
print(f"RMSE: {mean_squared_error(y_test, y_test_pred, squared=False):.2f}")
print(f"R²: {r2_score(y_test, y_test_pred):.2f}")

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(kernel='rbf', probability=True),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"--- {name} ---")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Finally we build a Confusion Matrix on the results.
ConfusionMatrixDisplay.from_estimator(models['Decision Tree'], X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix: Decision Tree')
plt.show()
ConfusionMatrixDisplay.from_estimator(models['SVM'], X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix: SVM')
plt.show()
ConfusionMatrixDisplay.from_estimator(models['MLP'], X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix: MLP')
plt.show()
ConfusionMatrixDisplay.from_estimator(models['Random Forest'], X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix: Random Forest')
plt.show()
ConfusionMatrixDisplay.from_estimator(models['KNN'], X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix: KNN')
plt.show()


for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: Mean accuracy = {scores.mean():.2f}")

# ROC Curve of 'Decision Tree'
y_pred_prob = models['Decision Tree'].predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# ROC Curve of 'SVM'
y_pred_prob = models['SVM'].predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# ROC Curve of 'MLP'
y_pred_prob = models['MLP'].predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# ROC Curve of 'Random Forest'
y_pred_prob = models['Random Forest'].predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# ROC Curve of 'KNN'
y_pred_prob = models['KNN'].predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
