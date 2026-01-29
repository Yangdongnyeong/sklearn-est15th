import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

# 1. Load Data
try:
    df = pd.read_csv('../data/titanic/train.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: File not found. Make sure '../data/titanic/train.csv' exists.")
    exit(1)

# 2. Preprocessing
def manual_preprocess(df):
    df = df.copy()
    # Fill missing
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Encode
    # Sex: male=0, female=1
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # Embarked: S=0, C=1, Q=2 
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    df['Embarked'] = df['Embarked'].map(embarked_map)
    
    return df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']], df['Survived']

X_final, y_final = manual_preprocess(df)

# 3. Modeling
models = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier())
]

voting_clf = VotingClassifier(estimators=models, voting='soft')

# Train
print("Training models...")
voting_clf.fit(X_final, y_final)

# Evaluate
score = voting_clf.score(X_final, y_final)
print(f"Training Accuracy: {score:.4f}")

# 4. Save Model
joblib.dump(voting_clf, 'titanic_model.pkl')
print("Model saved as titanic_model.pkl")
