# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus


file_path = 'cleaned_job_posting_2.csv'
df = pd.read_csv("cleaned_job_posting_2.csv", encoding="utf-8")


#split dataset in features and target variable
feature_cols = df.columns.tolist()[1:-1]
target_col = df.columns.tolist()[-1]
X = df[feature_cols] # Features
y = df[target_col] # Target variable


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=55) # 75% training and 25% test

# Create Decision Tree classifer object with 5 attributes with most information gain
# 5 attributes is good enough
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Graph
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('decision_tree.png')

# Image(graph.create_png())