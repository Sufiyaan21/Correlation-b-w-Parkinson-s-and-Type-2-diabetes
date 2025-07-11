import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import skew
import base64

# Function to set background image
def set_background(image_path):
    """
    Set the background image with a semi-transparent overlay for better readability,
    and remove unnecessary white backgrounds from components.
    """
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
            background-color: rgba(0, 0, 0, 0.6); /* Semi-transparent black overlay */
            z-index: 1;
        }}
        .stApp > div {{
            position: relative;
            z-index: 2; /* Make sure content is above the overlay */
        }}
        h1, h2, h3, h4, h5, h6, label {{
            font-weight: bold;
            color: white; /* Text color for contrast */
        }}
        input, select, textarea {{
            background-color: rgba(255, 255, 255, 0.9); /* Light background for text inputs */
            color: black; /* Dark text color for inputs */
            border-radius: 5px;
        }}
        .stSlider {{
            background: none !important; /* Remove background from sliders */
            border: none !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to set the background
set_background("2Q.png")

# Load dataset
data = pd.read_csv("pkdb.csv")

# Remove duplicates
data.drop_duplicates(inplace=True)

# Columns to exclude
attributes_to_drop = ['PatientID', 'DoctorInCharge', 'EducationLevel', 'UPDRS', 'MoCA', 'FunctionalAssessment', 'Constipation']
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
st.title("**Parkinson's Disease Risk Prediction**")
st.write("This app predicts the risk of Parkinson's Disease and highlights extreme attributes contributing to the risk.")

# User Input Form
st.header("**Input User Data**")
user_input = {}

# Collect numerical inputs
st.markdown("### **Numerical Inputs**")
for col in ['Age', 'BMI', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides']:
    user_input[col] = st.number_input(
        f"**{col}**",
        value=float(data[col].mean())  # Default value with no constraints
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
ethnicity_mapping = {
    'Caucasian': 0,
    'Indian': 1,
    'Black': 2,
    'Asian': 3
}
user_input['Ethnicity'] = st.selectbox(
    "**Ethnicity**",
    options=list(ethnicity_mapping.keys())  # Display human-readable labels
)
user_input['Ethnicity'] = ethnicity_mapping[user_input['Ethnicity']]  # Map to numeric values

for col in ['Smoking', 'FamilyHistoryParkinsons', 'TraumaticBrainInjury', 'Hypertension', 'Diabetes', 'Depression', 'Stroke', 'Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability', 'SpeechProblems', 'SleepDisorders']:
    user_input[col] = st.radio(f"**{col}**", options=['No', 'Yes'])
    user_input[col] = 1 if user_input[col] == 'Yes' else 0

data1 = pd.read_csv('pkdb.csv')
data1 = data1.drop(columns=attributes_to_drop)

# Predict and Display Results
if st.button("**Predict Parkinson's Risk**"):
    # Convert user input into DataFrame
    user_df = pd.DataFrame([user_input], columns=X.columns)
    extremes = identify_extreme_attributes(user_input, data1)
    # Apply scaling to numerical inputs
    user_df[numerical_columns] = scaler.transform(user_df[numerical_columns])

    # Predict the probability for user input
    y_user_proba = model.predict_proba(user_df)[:, 1][0] * 100  # Convert to percentage
    if y_user_proba > 50: y_user_proba = y_user_proba/3
    else: y_user_proba = y_user_proba/5
    # Identify extreme attributes
    

    # Display results
    st.subheader("**Prediction Results**")
    st.write(f"**Predicted Risk of Parkinson's Disease: {y_user_proba:.2f}%**")

    if y_user_proba > 10:
        if extremes:
            st.write("**Probable Attributes Contributing to Risk:** [" + ", ".join([f"**{attr}**" for attr in extremes]) + "]")
        else:
            st.write("**No extreme attributes identified, but the risk is high.**")
    else:
        st.write("**Values are Normal. Maintain this moving forth!**")
