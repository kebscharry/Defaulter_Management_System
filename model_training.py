import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_preprocessing import preprocess_data, split_data

def train_random_forest(file_path):
    df = preprocess_data(file_path)
    X_train, X_test, y_train, y_test = split_data(df)

    # Define the features
    features = ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP',
                'Days Between Last and Next Visit', 'Days Between Last and Self Visit',
                'Months Of Prescription', 'AHD Client', 'Medical Cover', 'Sex']

    print("Features used for training:", features)  # Moved inside function before training

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions and Evaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save the model
    joblib.dump(clf, 'models/random_forest_model.pkl')

if __name__ == '__main__':
    train_random_forest('activeOnART.xls')
