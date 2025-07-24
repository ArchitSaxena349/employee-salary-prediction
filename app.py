import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs (matching exact training features)
st.sidebar.header("Input Employee Details")

# Collect all required features
age = st.sidebar.slider("Age", 17, 75, 30)

workclass = st.sidebar.selectbox("Work Class", [
    "Federal-gov", "Local-gov", "Others", "Private", "Self-emp-inc", 
    "Self-emp-not-inc", "State-gov"
])

fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1500000, value=200000)

educational_num = st.sidebar.slider("Education Number", 5, 16, 10)

marital_status = st.sidebar.selectbox("Marital Status", [
    "Divorced", "Married-AF-spouse", "Married-civ-spouse", "Married-spouse-absent",
    "Never-married", "Separated", "Widowed"
])

occupation = st.sidebar.selectbox("Occupation", [
    "Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial",
    "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct", "Others",
    "Priv-house-serv", "Prof-specialty", "Protective-serv", "Sales",
    "Tech-support", "Transport-moving"
])

relationship = st.sidebar.selectbox("Relationship", [
    "Husband", "Not-in-family", "Other-relative", "Own-child", "Unmarried", "Wife"
])

race = st.sidebar.selectbox("Race", [
    "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"
])

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=100000, value=0)

capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=5000, value=0)

hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)

native_country = st.sidebar.selectbox("Native Country", [
    "Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic",
    "Ecuador", "El-Salvador", "England", "France", "Germany", "Greece",
    "Guatemala", "Haiti", "Holand-Netherlands", "Honduras", "Hong", "Hungary",
    "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan", "Laos",
    "Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines",
    "Poland", "Portugal", "Puerto-Rico", "Scotland", "South", "Taiwan",
    "Thailand", "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia"
])

# Encode categorical variables (using same mapping as training)
def encode_features(workclass_val, marital_val, occupation_val, relationship_val, race_val, gender_val, country_val):
    # Create mappings based on alphabetical order (same as LabelEncoder)
    workclass_map = {"Federal-gov": 0, "Local-gov": 1, "Others": 2, "Private": 3, "Self-emp-inc": 4, "Self-emp-not-inc": 5, "State-gov": 6}
    marital_map = {"Divorced": 0, "Married-AF-spouse": 1, "Married-civ-spouse": 2, "Married-spouse-absent": 3, "Never-married": 4, "Separated": 5, "Widowed": 6}
    occupation_map = {"Adm-clerical": 0, "Armed-Forces": 1, "Craft-repair": 2, "Exec-managerial": 3, "Farming-fishing": 4, "Handlers-cleaners": 5, "Machine-op-inspct": 6, "Others": 7, "Priv-house-serv": 8, "Prof-specialty": 9, "Protective-serv": 10, "Sales": 11, "Tech-support": 12, "Transport-moving": 13}
    relationship_map = {"Husband": 0, "Not-in-family": 1, "Other-relative": 2, "Own-child": 3, "Unmarried": 4, "Wife": 5}
    race_map = {"Amer-Indian-Eskimo": 0, "Asian-Pac-Islander": 1, "Black": 2, "Other": 3, "White": 4}
    gender_map = {"Female": 0, "Male": 1}
    
    # For native country, we'll use a simplified mapping for most common countries
    country_map = {country: i for i, country in enumerate(sorted([
        "Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic",
        "Ecuador", "El-Salvador", "England", "France", "Germany", "Greece",
        "Guatemala", "Haiti", "Holand-Netherlands", "Honduras", "Hong", "Hungary",
        "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan", "Laos",
        "Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines",
        "Poland", "Portugal", "Puerto-Rico", "Scotland", "South", "Taiwan",
        "Thailand", "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia"
    ]))}
    
    return (workclass_map[workclass_val], marital_map[marital_val], occupation_map[occupation_val], 
            relationship_map[relationship_val], race_map[race_val], gender_map[gender_val], country_map[country_val])

# Encode the categorical inputs
workclass_encoded, marital_encoded, occupation_encoded, relationship_encoded, race_encoded, gender_encoded, country_encoded = encode_features(
    workclass, marital_status, occupation, relationship, race, gender, native_country
)

# Build input DataFrame with exact feature names used in training
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass_encoded],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'marital-status': [marital_encoded],
    'occupation': [occupation_encoded],
    'relationship': [relationship_encoded],
    'race': [race_encoded],
    'gender': [gender_encoded],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [country_encoded]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    result = ">50K" if prediction[0] == ">50K" else "<=50K"
    st.success(f"✅ Prediction: {result}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
st.markdown("⚠️ **Note**: For batch prediction, upload a CSV with the exact columns used in training:")
st.markdown("age, workclass, fnlwgt, educational-num, marital-status, occupation, relationship, race, gender, capital-gain, capital-loss, hours-per-week, native-country")

uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    
    # Remove 'income' column if present (target variable)
    if 'income' in batch_data.columns:
        batch_data = batch_data.drop(columns=['income'])
        st.info("Removed 'income' column from uploaded data as it's the target variable.")
    
    # Check if all required columns are present
    required_cols = ['age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status', 
                     'occupation', 'relationship', 'race', 'gender', 'capital-gain', 
                     'capital-loss', 'hours-per-week', 'native-country']
    
    missing_cols = [col for col in required_cols if col not in batch_data.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        try:
            # Make a copy of the data for processing
            processed_data = batch_data.copy()
            
            # Encode categorical columns if they contain text values
            def encode_batch_categorical(df):
                # First, handle missing values and preprocessing (same as training)
                # Replace '?' with 'Others' for workclass and occupation
                if 'workclass' in df.columns:
                    df['workclass'] = df['workclass'].replace({'?': 'Others'})
                
                if 'occupation' in df.columns:
                    df['occupation'] = df['occupation'].replace({'?': 'Others', 'Other-service': 'Others'})
                
                if 'native-country' in df.columns:
                    df['native-country'] = df['native-country'].replace({'?': 'United-States'})  # Most common country
                
                # Filter out rows with 'Without-pay' and 'Never-worked' (same as training)
                if 'workclass' in df.columns:
                    initial_rows = len(df)
                    df = df[df['workclass'] != 'Without-pay']
                    df = df[df['workclass'] != 'Never-worked']
                    filtered_rows = len(df)
                    if filtered_rows < initial_rows:
                        st.info(f"Filtered out {initial_rows - filtered_rows} rows with 'Without-pay' or 'Never-worked' workclass (same as training data preprocessing).")
                
                # Define the mappings (same as used in training)
                workclass_map = {"Federal-gov": 0, "Local-gov": 1, "Others": 2, "Private": 3, "Self-emp-inc": 4, "Self-emp-not-inc": 5, "State-gov": 6}
                marital_map = {"Divorced": 0, "Married-AF-spouse": 1, "Married-civ-spouse": 2, "Married-spouse-absent": 3, "Never-married": 4, "Separated": 5, "Widowed": 6}
                occupation_map = {"Adm-clerical": 0, "Armed-Forces": 1, "Craft-repair": 2, "Exec-managerial": 3, "Farming-fishing": 4, "Handlers-cleaners": 5, "Machine-op-inspct": 6, "Others": 7, "Priv-house-serv": 8, "Prof-specialty": 9, "Protective-serv": 10, "Sales": 11, "Tech-support": 12, "Transport-moving": 13}
                relationship_map = {"Husband": 0, "Not-in-family": 1, "Other-relative": 2, "Own-child": 3, "Unmarried": 4, "Wife": 5}
                race_map = {"Amer-Indian-Eskimo": 0, "Asian-Pac-Islander": 1, "Black": 2, "Other": 3, "White": 4}
                gender_map = {"Female": 0, "Male": 1}
                
                # Create country mapping for all possible countries
                all_countries = ["Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic",
                               "Ecuador", "El-Salvador", "England", "France", "Germany", "Greece",
                               "Guatemala", "Haiti", "Holand-Netherlands", "Honduras", "Hong", "Hungary",
                               "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan", "Laos",
                               "Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines",
                               "Poland", "Portugal", "Puerto-Rico", "Scotland", "South", "Taiwan",
                               "Thailand", "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia"]
                country_map = {country: i for i, country in enumerate(sorted(all_countries))}
                
                # Apply encodings only if the column contains string values
                if 'workclass' in df.columns and df['workclass'].dtype == 'object':
                    df['workclass'] = df['workclass'].map(workclass_map).fillna(2)  # Default to 'Others' (2)
                
                if 'marital-status' in df.columns and df['marital-status'].dtype == 'object':
                    df['marital-status'] = df['marital-status'].map(marital_map).fillna(4)  # Default to 'Never-married' (4)
                
                if 'occupation' in df.columns and df['occupation'].dtype == 'object':
                    df['occupation'] = df['occupation'].map(occupation_map).fillna(7)  # Default to 'Others' (7)
                
                if 'relationship' in df.columns and df['relationship'].dtype == 'object':
                    df['relationship'] = df['relationship'].map(relationship_map).fillna(1)  # Default to 'Not-in-family' (1)
                
                if 'race' in df.columns and df['race'].dtype == 'object':
                    df['race'] = df['race'].map(race_map).fillna(4)  # Default to 'White' (4) - most common
                
                if 'gender' in df.columns and df['gender'].dtype == 'object':
                    df['gender'] = df['gender'].map(gender_map).fillna(1)  # Default to 'Male' (1) - most common
                
                if 'native-country' in df.columns and df['native-country'].dtype == 'object':
                    df['native-country'] = df['native-country'].map(country_map).fillna(country_map.get('United-States', 38))  # Default to United-States
                
                return df
            
            # Encode categorical columns
            processed_data = encode_batch_categorical(processed_data)
            
            # Make predictions
            batch_preds = model.predict(processed_data[required_cols])
            batch_data = batch_data.reset_index(drop=True)  # Reset index after potential filtering
            processed_data = processed_data.reset_index(drop=True)
            
            # Only add predictions for remaining rows after filtering
            if len(batch_data) != len(batch_preds):
                # If rows were filtered, update batch_data to match
                batch_data = batch_data.iloc[:len(batch_preds)].copy()
            
            batch_data['PredictedClass'] = batch_preds
            st.write("✅ Predictions:")
            st.write(batch_data.head())
            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
            
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            st.info("Check that your CSV file has the correct format and column names.")

