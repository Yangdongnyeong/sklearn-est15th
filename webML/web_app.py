import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Load the trained model
try:
    model = joblib.load('titanic_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file 'titanic_model.pkl' not found. Please run myModel.ipynb first.")
    # For demo purposes, we might want to handle this gracefully or exit
    # exit(1)

def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    # Create DataFrame for prediction
    # We pass the raw values directly because the Pipeline (with ColumnTransformer) 
    # handles the imputation, scaling, and one-hot encoding automatically!
    data = pd.DataFrame([{
        'Pclass': int(pclass),
        'Sex': sex,         # 'male' or 'female'
        'Age': float(age),
        'SibSp': int(sibsp),
        'Parch': int(parch),
        'Fare': float(fare),
        'Embarked': embarked # 'S', 'C', or 'Q'
    }])
    
    # Predict
    # Ensure the model is loaded
    try:
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]
        
        result = "Survived" if prediction == 1 else "Did Not Survive"
        return f"{result} (Probability: {probability:.2%})"
    except Exception as e:
        return f"Prediction Error: {str(e)}"

# Define Gradio Interface
iface = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Radio([1, 2, 3], label="Pclass (Ticket Class)", value=3),
        gr.Radio(["male", "female"], label="Sex", value="male"),
        gr.Slider(0, 100, value=30, label="Age"),
        gr.Number(value=0, label="SibSp (Siblings/Spouses aboard)"),
        gr.Number(value=0, label="Parch (Parents/Children aboard)"),
        gr.Number(value=32.2, label="Fare"),
        gr.Radio(["S", "C", "Q"], label="Embarked (Port)", value="S")
    ],
    outputs="text",
    title="Titanic Survival Predictor",
    description="Enter passenger details to predict survival probability."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0")
