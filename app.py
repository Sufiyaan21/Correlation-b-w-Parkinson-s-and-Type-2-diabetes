import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import skew
import base64

# Function to set background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            position: relative;
            color: white;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.6);
            z-index: 1;
        }}
        .stApp > div {{
            position: relative;
            z-index: 2;
        }}
        h1, h2, h3, h4, h5, h6, label {{
            font-weight: bold;
            color: white;
            font-size: 20px;
        }}
        input, select, textarea {{
            font-size: 18px;
            background-color: rgba(255, 255, 255, 0.9);
            color: black;
            border-radius: 5px;
        }}
        .stTextInput > div > input, .stSelectbox > div > div > select {{
            font-size: 18px;
        }}
        .stNumberInput input, .stSlider {{
            font-size: 18px;
        }}
        .stMarkdown p {{
            font-size: 18px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to set the background
set_background("2Q.png")

# Load dataset
data = pd.read_csv("pkdb.csv")
data.drop_duplicates(inplace=True)

# Columns to exclude
attributes_to_drop = ['PatientID', 'DoctorInCharge', 'EducationLevel', 'UPDRS', 'MoCA', 
                      'FunctionalAssessment', 'Constipation']
data = data.drop(columns=attributes_to_drop)

# Define numerical and categorical columns
numerical_columns = [
    'Age', 'BMI', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
    'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
    'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality'
]

categorical_columns = [
    'Gender', 'Ethnicity', 'Smoking', 'FamilyHistoryParkinsons', 'TraumaticBrainInjury',
    'Hypertension', 'Diabetes', 'Depression', 'Stroke', 'Tremor', 'Rigidity',
    'Bradykinesia', 'PosturalInstability', 'SpeechProblems', 'SleepDisorders'
]

# Function to remove outliers
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

data = remove_outliers(data, numerical_columns)

# Function to check and normalize skewed features
def check_and_normalize(df, columns):
    pt = PowerTransformer(method='yeo-johnson')
    for col in columns:
        skewness = skew(df[col])
        if abs(skewness) > 0.5:
            df[col] = pt.fit_transform(df[col].values.reshape(-1, 1))
    return df

data = check_and_normalize(data, numerical_columns)

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Split data into features and target
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

# Train Gradient Boosting model
model = GradientBoostingClassifier()
model.fit(X, y)

# Function to identify extreme attributes
def identify_extreme_attributes(user_input, dataset):
    extreme_attributes = []
    for attribute in dataset.columns:
        if attribute in user_input:
            mean_val = dataset[attribute].mean()
            min_val = dataset[attribute].min()
            max_val = dataset[attribute].max()
            user_val = user_input[attribute]
            if user_val < min_val or user_val > max_val or abs(user_val - mean_val) > (max_val - mean_val):
                extreme_attributes.append(attribute)
    return extreme_attributes

# Streamlit App
st.markdown(
    """
    <h1 style="font-size: 48px; text-align: center; font-weight: bold; color: white;">
        Parkinson's Disease Risk Prediction
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("This app predicts the risk of Parkinson's Disease and highlights extreme attributes contributing to the risk.")

# User Input Form
st.header("**Input User Data**")
user_input = {}

# Collect numerical inputs
st.markdown("### **Numerical Inputs**")
for col in ['Age', 'BMI', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides']:
    user_input[col] = st.number_input(
        f"**{col}**",
        value=float(data[col].mean())
    )

st.markdown("### **Lifestyle Inputs**")
for col in ['AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality']:
    user_input[col] = st.slider(f"**{col}**", min_value=0, max_value=15, value=7)

# Collect categorical inputs
st.markdown("### **Categorical Inputs**")
gender_mapping = {'Male': 0, 'Female': 1, 'Others': 2}
user_input['Gender'] = st.selectbox("**Gender**", options=list(gender_mapping.keys()))
user_input['Gender'] = gender_mapping[user_input['Gender']]

# Ethnicity Mapping
ethnicity_mapping = {'Caucasian': 0, 'Indian': 1, 'Black': 2, 'Asian': 3}
user_input['Ethnicity'] = st.selectbox("**Ethnicity**", options=list(ethnicity_mapping.keys()))
user_input['Ethnicity'] = ethnicity_mapping[user_input['Ethnicity']]

for col in ['Smoking', 'FamilyHistoryParkinsons', 'TraumaticBrainInjury', 'Hypertension', 'Diabetes', 'Depression', 
            'Stroke', 'Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability', 'SpeechProblems', 'SleepDisorders']:
    user_input[col] = st.radio(f"**{col}**", options=['No', 'Yes'])
    user_input[col] = 1 if user_input[col] == 'Yes' else 0

data1 = pd.read_csv('pkdb.csv')
data1 = data1.drop(columns=attributes_to_drop)

if st.button("**Predict Parkinson's Risk**"):
    user_df = pd.DataFrame([user_input], columns=X.columns)
    extremes = identify_extreme_attributes(user_input, data1)
    user_df[numerical_columns] = scaler.transform(user_df[numerical_columns])

    y_user_proba = model.predict_proba(user_df)[:, 1][0] * 100
    if y_user_proba > 50: 
        y_user_proba = y_user_proba / 3
    else: 
        y_user_proba = y_user_proba / 5

    # Determine diagnosis and risk level
    if y_user_proba > 30:
        diagnosis = "Case 4 - Diagnosis of Only Parkinson's"
        risk_level = "High Risk"
        extreme_message = "No attributes identified but risk is high."
    elif y_user_proba < 10:
        diagnosis = "Case 2 - Healthy, no indicators detect Parkinson's or Diabetes"
        risk_level = "No Risk"
        extreme_message = "All attributes are normal."
    elif 17 < y_user_proba <= 25:
        diagnosis = "Case 3 - Diagnosis of Only Diabetes"
        risk_level = "High Risk"
        extreme_message = f"Attributes Contributing to Risk: {', '.join(extremes)}" if extremes else "No extreme attributes identified."
    elif 25 < y_user_proba <= 30:
        diagnosis = "Case 1 - Diagnosis of both Parkinson's and Diabetes"
        risk_level = "High Risk"
        extreme_message = f"Attributes Contributing to Risk: {', '.join(extremes)}" if extremes else "No extreme attributes identified."
    else:
        diagnosis = "Risk level falls outside predefined cases. Consult a specialist."
        risk_level = "Uncertain"
        extreme_message = ""

    # Display report in a styled box
    st.markdown(f"""
        <div style="
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 20px;
            background-color: #2d2d2d;
            color: white;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.5);
        ">
            <h3 style="text-align: center; font-weight: bold;">Risk Prediction Report</h3>
            <hr style="border: 1px solid #4CAF50;">
            <p><b>Predicted Risk Percentage:</b> {y_user_proba:.2f}%</p>
            <p><b>Diagnosis Result:</b> {diagnosis}</p>
            <p><b>Risk Level:</b> {risk_level}</p>
            <hr style="border: 1px solid #4CAF50;">
            <p style="font-weight: bold;">{extreme_message}</p>
        </div>
    """, unsafe_allow_html=True)

# Info Button at the Bottom
st.markdown("<br><br>", unsafe_allow_html=True)  # Add spacing for better layout

if 'show_info' not in st.session_state:
    st.session_state['show_info'] = False

if st.button("Info"):
    st.session_state['show_info'] = not st.session_state['show_info']

if st.session_state['show_info']:
    st.info("""
    **Case 1**: Diagnosis of both Parkinson's and Diabetes  
    **Case 2**: Healthy, no indicators detect Parkinson's or Diabetes  
    **Case 3**: Diagnosis of Only Diabetes  
    **Case 4**: Diagnosis of Only Parkinson's  
    """)
