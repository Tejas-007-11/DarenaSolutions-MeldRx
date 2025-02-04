import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('stroke_prediction_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    gender = request.form['gender']
    age = float(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = 1 if request.form['ever_married'] == 'Yes' else 0
    work_type = request.form['work_type']
    residence_type = request.form['residence_type']
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = request.form['smoking_status']

    # Encode categorical data
    gender = 1 if gender == 'Male' else 0
    work_type = {'children': 0, 'Govt_job': 1, 'Never_worked': 2, 'Private': 3, 'Self-employed': 4}[work_type]
    residence_type = 1 if residence_type == 'Urban' else 0
    smoking_status = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}[smoking_status]

    # Create input array for model
    input_data = np.array([[gender, age, hypertension, heart_disease, ever_married, work_type,
                            residence_type, avg_glucose_level, bmi, smoking_status]])

    # Make prediction
    print("Input Data:", input_data)
    prediction = model.predict(input_data)
    print("Prediction:", prediction)
    print("Model Raw Prediction:", prediction)


    # Return result
    if prediction == 1:
        return render_template('index.html', prediction_text="The patient has a high risk of stroke.")
    else:
        return render_template('index.html', prediction_text="The patient has a low risk of stroke.")

if __name__ == "__main__":
    app.run(debug=True)


