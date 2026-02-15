import streamlit as st
import pickle
import pandas as pd

MODEL_PATHS = {
    'XGBoost': "Model/model_xgb.pkl",
    'SVM': "Model/model_svm.pkl",
    'Logistic Regression': "Model/model_lr.pkl",
    'Random Forest': "Model/model_rf.pkl"
}

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model(model_name):
    with open(MODEL_PATHS[model_name], 'rb') as file:
        model = pickle.load(file)
    return model


# -------------------------------
# Preprocess Data
# -------------------------------
def preprocess_data(data):
    # Manual encoding (must match training encoding)
    # Assuming during training:
    # Yes = 1, No = 0

    data['International plan'] = data['International plan'].map({'Yes': 1, 'No': 0})
    data['Voice mail plan'] = data['Voice mail plan'].map({'Yes': 1, 'No': 0})

    # If State was label encoded during training,
    # you MUST load the same encoder.
    # For now we assume states were already numeric.
    # If not, remove State or properly encode it.

    return data


# -------------------------------
# Prediction Function
# -------------------------------
def predict_churn(data, model):
    data = preprocess_data(data)
    prediction = model.predict(data)
    return prediction


# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.title('Telecom Churn Prediction App')
    st.markdown("---")

    model_name = st.sidebar.selectbox('Select Model', list(MODEL_PATHS.keys()))
    model = load_model(model_name)

    st.sidebar.header('Input Features')

    state = st.sidebar.text_input('State', '')
    account_length = st.sidebar.number_input('Account Length', min_value=1, step=1)
    area_code = st.sidebar.number_input('Area Code', min_value=100, max_value=999, step=1)
    international_plan = st.sidebar.selectbox('International Plan', ['Yes', 'No'])
    voice_mail_plan = st.sidebar.selectbox('Voice Mail Plan', ['Yes', 'No'])
    number_vmail_messages = st.sidebar.number_input('Number of Voicemail Messages', min_value=0)

    st.sidebar.header('Day Usage')
    total_day_minutes = st.sidebar.number_input('Total Day Minutes', min_value=0.0)
    total_day_calls = st.sidebar.number_input('Total Day Calls', min_value=0)
    total_day_charge = st.sidebar.number_input('Total Day Charge', min_value=0.0)

    st.sidebar.header('Evening Usage')
    total_eve_minutes = st.sidebar.number_input('Total Eve Minutes', min_value=0.0)
    total_eve_calls = st.sidebar.number_input('Total Eve Calls', min_value=0)
    total_eve_charge = st.sidebar.number_input('Total Eve Charge', min_value=0.0)

    st.sidebar.header('Night Usage')
    total_night_minutes = st.sidebar.number_input('Total Night Minutes', min_value=0.0)
    total_night_calls = st.sidebar.number_input('Total Night Calls', min_value=0)
    total_night_charge = st.sidebar.number_input('Total Night Charge', min_value=0.0)

    st.sidebar.header('International Usage')
    total_intl_minutes = st.sidebar.number_input('Total Intl Minutes', min_value=0.0)
    total_intl_calls = st.sidebar.number_input('Total Intl Calls', min_value=0)
    total_intl_charge = st.sidebar.number_input('Total Intl Charge', min_value=0.0)

    st.sidebar.header('Customer Service')
    customer_service_calls = st.sidebar.number_input('Customer Service Calls', min_value=0)

    # Create DataFrame
    user_data = pd.DataFrame({
        'State': [state],
        'Account length': [account_length],
        'Area code': [area_code],
        'International plan': [international_plan],
        'Voice mail plan': [voice_mail_plan],
        'Number vmail messages': [number_vmail_messages],
        'Total day minutes': [total_day_minutes],
        'Total day calls': [total_day_calls],
        'Total day charge': [total_day_charge],
        'Total eve minutes': [total_eve_minutes],
        'Total eve calls': [total_eve_calls],
        'Total eve charge': [total_eve_charge],
        'Total night minutes': [total_night_minutes],
        'Total night calls': [total_night_calls],
        'Total night charge': [total_night_charge],
        'Total intl minutes': [total_intl_minutes],
        'Total intl calls': [total_intl_calls],
        'Total intl charge': [total_intl_charge],
        'Customer service calls': [customer_service_calls]
    })

    if st.button('Predict'):
        try:
            prediction = predict_churn(user_data, model)

            if prediction[0] == 0:
                st.success('This customer is predicted to stay.')
            else:
                st.error('This customer is predicted to churn.')

        except Exception as e:
            st.error("Prediction failed. Check model features & encoding.")
            st.write(e)


if __name__ == '__main__':
    main()
