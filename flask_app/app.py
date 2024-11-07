from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('../models/random_forest_model.pkl')


@app.route('/')
def index():
    return render_template('index.html', prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    days_between_next_last = request.form['days_between_next_last']
    days_between_last_self = request.form['days_between_last_self']
    months_of_prescription = request.form['months_of_prescription']
    ahd_client = request.form['ahd_client']
    age = request.form['age']
    bmi = request.form['bmi']
    systolic_bp = request.form['systolic_bp']
    diastolic_bp = request.form['diastolic_bp']
    medical_cover = request.form['medical_cover']
    sex = request.form['sex']

    # Prepare input for the model
    input_data = pd.DataFrame([[int(age), float(bmi), int(systolic_bp), int(diastolic_bp),
                                 int(days_between_next_last), int(days_between_last_self),
                                 int(months_of_prescription), int(ahd_client), int(medical_cover), int(sex)]],
                               columns=['Age',
                                        'BMI',
                                        'Systolic_BP',
                                        'Diastolic_BP',
                                        'Days Between Last and Next Visit',
                                        'Days Between Last and Self Visit',
                                        'Months Of Prescription',
                                        'AHD Client',
                                        'Medical Cover',
                                        'Sex'])

    # Check input data before prediction
    print("Input data for prediction:", input_data)

    try:
        prediction = model.predict(input_data)[0]
    except ValueError as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction="Error: Check input features!")

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
