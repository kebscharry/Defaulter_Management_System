import joblib
import pandas as pd
from sklearn.metrics import classification_report
from data_preprocessing import preprocess_data


def evaluate_model(file_path):
    df = preprocess_data(file_path)
    X = df[['Age', 'BMI', 'Sex', 'Systolic_BP', 'Diastolic_BP', 'Months Of Prescription', 'AHD Client', 'Medical Cover']]
    y = df['defaulter']

    model = joblib.load('models/random_forest_model.pkl')
    y_pred = model.predict(X)

    report = classification_report(y, y_pred)
    print(report)

if __name__ == '__main__':
    evaluate_model('activeOnART.xls')
