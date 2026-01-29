import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

print("Current working directory:", os.getcwd())
data_path = '../data/titanic/train.csv'
print("Looking for data at:", os.path.abspath(data_path))

if not os.path.exists(data_path):
    print("ERROR: Data file not found!")
    exit(1)

try:
    df = pd.read_csv(data_path)
    print(f"Data loaded. Shape: {df.shape}")
except Exception as e:
    print(f"ERROR loading csv: {e}")
    exit(1)

# Preprocessing
try:
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Mapping
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    df['Embarked'] = df['Embarked'].map(embarked_map)
    
    # Features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features]
    y = df['Survived']
    
    # Check for NaNs
    if X.isnull().sum().sum() > 0:
        print("Warning: NaNs remain in X")
        print(X.isnull().sum())
        X = X.fillna(0) # Fallback
        
except Exception as e:
    print(f"ERROR in preprocessing: {e}")
    exit(1)

# Modeling
try:
    models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42)),
        ('knn', KNeighborsClassifier())
    ]
    
    voting_clf = VotingClassifier(estimators=models, voting='soft')
    
    print("Training VotingClassifier...")
    voting_clf.fit(X, y)
    print(f"Score: {voting_clf.score(X, y)}")
    
    joblib.dump(voting_clf, 'titanic_model.pkl')
    print("SUCCESS: Model saved to titanic_model.pkl")

except Exception as e:
    print(f"ERROR in modeling: {e}")
    exit(1)
