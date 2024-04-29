import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



# Load the dataset for Symptoms Checker page
@st.cache_data
def load_symptoms_data():
    data = pd.read_csv("CSV\SymptomsMedicine.csv")
    return data



# Load the dataset for Drug Verifier page
@st.cache_data
def load_drug_data():
    data = pd.read_csv("CSV\MedicineDetails.csv")
    return data



# Load the dataset for Home Remedies page
@st.cache_data
def load_remedies_data():
    data = pd.read_csv("CSV\HomeRemedies.csv")
    return data



# Train the Random Forest Regressor
def train_model(data):
    X = data.drop(columns=["Health Issue", "Home Remedy", "Effectiveness"])  # Features
    y = data["Effectiveness"]  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Splitting the dataset
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # Initializing the model
    model.fit(X_train, y_train)  # Training the model
    y_pred = model.predict(X_test)  # Making predictions on the test set
    mse = mean_squared_error(y_test, y_pred)  # Calculating Mean Squared Error
    st.write("Mean Squared Error:", mse)  # Displaying MSE
    return model



# Function to predict effectiveness based on symptoms
def predict_effectiveness(model, symptoms_features):
    # Predicting effectiveness using the trained model
    effectiveness = model.predict([symptoms_features])
    return effectiveness[0]



# Function to display Symptoms Verifier page
def symptoms_verifier(data):
    st.title("Symptoms Checker")
    symptoms = st.text_input("Enter your symptoms separated by commas (e.g., fever, cough, headache):")
    if st.button("Check Symptoms"):
        if symptoms.strip():  # Check if the search query is not empty
            detect_illness(data, symptoms)
        else:
            st.write("Please enter one or more symptoms before checking.")



# Function to detect illness and medicine
def detect_illness(data, symptoms):
    detected_illness = []
    detected_medicine = []
    for symptom in symptoms.split(","):
        matches = data[data["Symptoms"].str.contains(symptom.strip(), case=False)]
        if not matches.empty:
            detected_illness.extend(matches["Disease"].tolist())
            detected_medicine.extend(matches["Medicine"].tolist())
    
    detected_illness = list(set(detected_illness))  # Remove duplicates
    detected_medicine = list(set(detected_medicine))  # Remove duplicates
    
    st.subheader("Detected Illness:")
    if detected_illness:
        st.write(detected_illness)
    else:
        st.write("No matching illness found for the given symptoms.")
    
    st.subheader("Recommended Medicine:")
    if detected_medicine:
        st.write(detected_medicine)
    else:
        st.write("No recommended medicine found.")



# Function to display Drug Verifier page
def drug_checker(data):
    st.title("Drug Verifier")
    
    drug_name = st.text_input("Enter the name of the drug:")
    drug_name = drug_name.lower()
    
    if drug_name:
        matches = data[data["Medicine Name"].str.lower().str.contains(drug_name)]
        if not matches.empty:
            st.subheader("Medicine Name:")
            st.write(matches["Medicine Name"].iloc[0])
            
            st.subheader("Uses:")
            st.write(matches["Uses"].iloc[0])
            
            st.subheader("Side Effects:")
            st.write(matches["Side_effects"].iloc[0])
            
            st.subheader("Manufacturer:")
            st.write(matches["Manufacturer"].iloc[0])
        else:
            st.write("Drug not found in the dataset.")



# Function to display Home Remedies page
def home_remedies(data):
    st.title("Home Remedies")
    illness = st.text_input("Enter the common illness:")
    if st.button("Find Home Remedy"):
        display_home_remedy(data, illness)



# Function to display home remedies for the illness
def display_home_remedy(data, health_issue):
    # Check if the column names match
    if set(["Health Issue", "Home Remedy"]).issubset(set(data.columns)):
        # Convert health issue to lowercase for case-insensitive matching
        health_issue_lower = health_issue.lower()
        # Find matching health issue
        matches = data[data["Health Issue"].str.lower() == health_issue_lower]
        if not matches.empty:
            st.subheader("Home Remedies:")
            st.write(matches["Home Remedy"].iloc[0])
        else:
            st.write("Health issue not found in the dataset.")
    else:
        st.write("Invalid dataset format: Required columns not found.")



# Home page with navigation buttons
def main():
    st.title("Welcome to Health Assistant")

    page = st.sidebar.selectbox("Select a page", ["Symptoms Checker", "Drug Verifier", "Home Remedies"])

    if page == "Symptoms Checker":
        data = load_symptoms_data()
        symptoms_verifier(data)
    elif page == "Drug Verifier":
        data = load_drug_data()
        drug_checker(data)
    elif page == "Home Remedies":
        data = load_remedies_data()
        home_remedies(data)

if __name__ == "__main__":
    main()
