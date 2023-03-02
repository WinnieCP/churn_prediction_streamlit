import streamlit as st
import pandas as pd
import joblib

def preprocessing(df):
    df['TotalCharges'] = df['TotalCharges'].replace(' ','0') 
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    # drop customer ID: not a feature for training 
    df = df.drop('customerID', axis=1)
    return df

def generate_predictions(customer_data):
    pipe = joblib.load("pipeline.pkl")
    customer_data = pipe.transform(customer_data)
    model = joblib.load("churn_prediction_model.pkl")
    pred_y = model.predict(customer_data)
    return pred_y

if __name__ == '__main__':

    st.title("Customer Churn Prediction")

    gender = st.selectbox('What is the gender of the customer?',('Female','Male'))
    senior_citizen = st.selectbox('Is the customer a senior citizen',('Yes','No'))
    partner = st.selectbox('Does the customer have a partner?', ('Yes', 'No'))
    dependents = st.selectbox('Does the customer have a dependents?', ('Yes', 'No'))
    tenure = st.slider("How many months has the customer been with the company?", min_value=0, max_value=72, value=20)
    phone_service = st.selectbox('Does the customer have phone service?',('Yes','No'))
    multiple_lines = st.selectbox('Does the customer have multiple lines?',('No', 'Yes', 'No phone service'))
    internet_service = st.selectbox('What type of internet service does the customer have?',('DSL', 'No', 'Fiber optic'))
    online_security = st.selectbox('Does the customer have online security?',('No', 'No internet service', 'Yes'))
    online_backup = st.selectbox('Does the customer have an online backup?', ('No', 'No internet service', 'Yes'))
    device_protection = st.selectbox('Does the customer have device protection?',('Yes', 'No', 'No internet service'))
    tech_support = st.selectbox('Does the customer use tech support?',('Yes', 'No', 'No internet service'))
    streaming_tv = st.selectbox('Does the customer stream TV?',('Yes', 'No', 'No internet service'))
    streaming_movies = st.selectbox('Does the customer stream movies?',('Yes', 'No internet service', 'No'))
    contract = st.selectbox('What type of contract does the customer have?',('Month-to-month', 'One year', 'Two year'))
    paperless_billing = st.selectbox('Does the customer use paperless billing?',('No', 'Yes'))
    payment_method = st.selectbox("What is the customer's payment method?",('Bank transfer (automatic)', 'Electronic check', 'Mailed check',
        'Credit card (automatic)'))

    monthly_charges = st.slider("What is the customer's monthly charge? :", min_value=0, max_value=118, value=50)
    total_charges = st.slider('What is the total charge of the customer? :', min_value=0, max_value=8600, value=2000)
    input_dict = {'gender': gender,
                    'SeniorCitizen': senior_citizen,
                    'Partner': partner,
                    'Dependents': dependents,
                    'tenure': tenure,
                    'PhoneService': phone_service,
                    'MultipleLines': multiple_lines,
                    'InternetService': internet_service,
                    'OnlineSecurity': online_security,
                    'OnlineBackup': online_backup,
                    'DeviceProtection': device_protection,
                    'TechSupport': tech_support,
                    'StreamingTV': streaming_tv,
                    'StreamingMovies': streaming_movies,
                    'Contract': contract,
                    'PaperlessBilling': paperless_billing,
                    'PaymentMethod': payment_method,
                    'MonthlyCharges': monthly_charges,
                    'TotalCharges': total_charges,
                    'customerID': 1
                    }
    input_data = pd.DataFrame([input_dict])

    # generate the prediction for the customer
    if st.button("Predict Churn"):
        pred = generate_predictions(input_data)
        if bool(pred):
            st.error("Customer will churn!")
        else:
            st.success("Customer not predicted to churn")
