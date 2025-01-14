import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
from sklearn.model_selection import GridSearchCV

# Load data
file_path = 'cleaned_job_posting_2.csv'
df = pd.read_csv(file_path, encoding="utf-8")

# Split dataset into features and target variable
feature_cols = df.columns.tolist()[1:-1]
target_col = df.columns.tolist()[-1]
X = df[feature_cols]  # Features
y = df[target_col]  # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=55)

# Create a DecisionTreeClassifier with hyperparameters for tuning
clf = DecisionTreeClassifier(random_state=55)

# Hyperparameter tuning with GridSearchCV (for max_depth, min_samples_split, min_samples_leaf)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best parameters found by GridSearchCV
print("Best Parameters: ", grid_search.best_params_)

# Use the best model
best_clf = grid_search.best_estimator_

# Predict the response for the test dataset
y_pred = best_clf.predict(X_test)

# Model Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
precision=metrics.precision_score(y_test, y_pred)
recall=metrics.recall_score(y_test, y_pred)
F1=metrics.f1_score(y_test, y_pred)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", F1)

# Cross-validation score to assess model robustness
cv_scores = cross_val_score(best_clf, X, y, cv=5)
print("Cross-validation accuracy: ", cv_scores.mean())

# Graph the decision tree
dot_data = StringIO()
export_graphviz(best_clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('optimized_decision_tree.png')

# Optional: Display the image
# Image(graph.create_png())