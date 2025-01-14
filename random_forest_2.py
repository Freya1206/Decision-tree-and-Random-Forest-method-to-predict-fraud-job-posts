import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

file_path = 'cleaned_file_path_4.csv'
df = pd.read_csv(file_path)

# define text_columns and numeric_columns
text_columns = ['title', 'location', 'company_profile', 'description', 'requirements', 'benefits', 'employment_type', 'required_experience', 'required_education', 'industry', 'function']
numeric_columns = ['telecommuting', 'has_company_logo', 'has_questions']

# make sure the text_columns only contains str
for col in text_columns:
    df[col] = df[col].astype(str)

# make sure the numeric_columns only contains int
for col in numeric_columns:

    df[col] = pd.to_numeric(df[col], errors='coerce')

# fill the Null value
imputer = SimpleImputer(strategy='most_frequent')
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# check the result
print(df.head())

#import the tools and libraries need to use
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

text_columns = ['title', 'location', 'company_profile', 'description', 'requirements', 'benefits', 'employment_type',
                'required_experience', 'required_education', 'industry', 'function']
numeric_columns = ['telecommuting', 'has_company_logo', 'has_questions']

# Combine text columns into one
df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)

# Initialize and apply TfidfVectorizer only to text columns
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
text_features = tfidf.fit_transform(df['combined_text'])

# Check TF-IDF matrix shape
print("TF-IDF matrix shape:", text_features.shape)

numeric_features = df[numeric_columns]

# Fill missing values in numeric columns
imputer = SimpleImputer(strategy='most_frequent')  # Use most frequent value to fill missing data
numeric_features_filled = imputer.fit_transform(numeric_features)

# Standardize numeric featuresteatur
scaler = StandardScaler()
numeric_features_scaled = scaler.fit_transform(numeric_features_filled)

# Combine TF-IDF features and scaled numeric features
all_features = hstack([text_features, numeric_features_scaled])

# Check the combined feature matrix shape (optional)
print("Combined feature matrix shape:", all_features.shape)

# Assuming 'fraudulent' is your target column
# Features
X = all_features  # All features are now combined into one matrix
y = df['fraudulent']  # Target column

# Verify the shapes
print(X.shape)  # X should have the shape (17837, n_features)
print(y.shape)  # y should have the shape (17837,)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train a Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision", precision_score(y_test, y_pred))
print("Recall", recall_score(y_test, y_pred))
print ("F1",f1_score(y_test, y_pred))


import matplotlib.pyplot as plt
import numpy as np

# get feature importance
importances = clf.feature_importances_

# get the feature names after combined columns
text_feature_names = tfidf.get_feature_names_out()
numeric_feature_names = numeric_columns

# combine feature names
feature_names = list(text_feature_names) + list(numeric_feature_names)

# rank the importance and make the top 10
indices = np.argsort(importances)[::-1][:10]

# draw the pic
plt.figure(figsize=(10, 6))
plt.title("Top 10 Feature Importance")
plt.barh(range(10), importances[indices], align="center")
plt.yticks(range(10), np.array(feature_names)[indices])
plt.gca().invert_yaxis()
plt.xlabel("Relative Importance")
plt.show()

