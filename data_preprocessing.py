
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

def calculate_age(dob, report_date):
    dob = datetime.strptime(dob, '%d/%m/%Y')
    today = datetime.strptime(report_date, '%d/%m/%Y')
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

def calculate_bmi(weight, height):
    if height > 0:
        return weight / (height / 100) ** 2
    return None

def preprocess_data(file_path):
    df = pd.read_excel(file_path, header=4)  # Make sure this header is correct

    print("Columns in DataFrame:", df.columns)
    print(df.head())

    df.columns = df.columns.str.strip()  # Strip whitespace from column names

    df.ffill(inplace=True)  # Fill missing values

    if 'DOB' not in df.columns:
        raise KeyError("Column 'DOB' not found in the data.")

    # Convert relevant columns to datetime
    df['Next Appointment Date'] = pd.to_datetime(df['Next Appointment Date'], errors='coerce')
    df['Last Visit Date'] = pd.to_datetime(df['Last Visit Date'], errors='coerce')
    df['Self Visit Date'] = pd.to_datetime(df['Self Visit Date'], errors='coerce')

    # Debugging: Print the date columns to check for NaT
    print("Last Visit Dates After Conversion:", df['Last Visit Date'])
    print("Self Visit Dates After Conversion:", df['Self Visit Date'])

    # Calculate the number of days between visits
    df['Days Between Last and Next Visit'] = (df['Next Appointment Date'] - df['Last Visit Date']).dt.days
    df['Days Between Last and Self Visit'] = (df['Self Visit Date'] - df['Last Visit Date']).dt.days

    # Calculate Age and BMI
    df['Age'] = df.apply(lambda row: calculate_age(row['DOB'], '07/10/2024'), axis=1)
    df['BMI'] = df.apply(lambda row: calculate_bmi(row['Weight'], row['Height']), axis=1)

    # Create target variable based on converted date
    df['defaulter'] = (df['Next Appointment Date'] < pd.Timestamp.now()).astype(int)

    # Encode categorical features
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['AHD Client'] = label_encoder.fit_transform(df['AHD Client'].fillna('No'))
    df['Medical Cover'] = label_encoder.fit_transform(df['Medical Cover'].fillna('No'))

    # Handle Blood Pressure
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)

    return df



def split_data(df):
    from sklearn.model_selection import train_test_split

    # Add the missing features to the list
    features = ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP',
                'Days Between Last and Next Visit', 'Days Between Last and Self Visit',
                'Months Of Prescription', 'AHD Client', 'Medical Cover', 'Sex']
    target = 'defaulter'

    X = df[features]
    y = df[target]

    return train_test_split(X, y, test_size=0.2, random_state=42)

