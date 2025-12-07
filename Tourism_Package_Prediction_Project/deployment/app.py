import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os


st.set_page_config(page_title='Tourism Package Prediction', layout='wide')


@st.cache_resource
def load_model():
  # Either load from local file or HF model repo
  model_local = os.getenv('LOCAL_MODEL_PATH', 'Tourism_Package_Prediction_Project/model_building/tourism_pkg_predition_model_v1.joblib')
  if os.path.exists(model_local):
     return joblib.load(model_local)


  # Try HuggingFace repo
  try:
      model_file = hf_hub_download(
          repo_id="greatlearninglokeshrajoria/tourism-package-prediction",
          filename="tourism_model_xgb_v1.joblib",
          repo_type="model"
      )
      return joblib.load(model_file)
  except:
      return None


model = load_model()


st.title('Tourism Package Purchase Predictor')


with st.form('input_form'):
   col1, col2 = st.columns(2)
   with col1:
      Age = st.number_input('Age', min_value=18, max_value=100, value=33)
      TypeofContact = st.selectbox('TypeofContact', ['Self', 'Company'])
      Occupation = st.selectbox('Occupation', ['Salaried', 'Business', 'Retired', 'Student'])
      ProductPitched = st.selectbox('ProductPitched', ['Basic', 'Standard', 'Deluxe', 'King', 'Premium'])
      MaritalStatus = st.selectbox('MaritalStatus', ['Single', 'Married', 'Divorced', 'Widowed'])
   with col2:
      Designation = st.selectbox('Designation', ['Executive', 'Manager', 'Senior Manager', 'Director', 'Others'])
      Gender = st.selectbox('Gender', ['Male', 'Female'])
      MonthlyIncome = st.number_input('MonthlyIncome', min_value=0, value=25000)
      PitchSatisfactionScore = st.slider('PitchSatisfactionScore', 0, 10, 7)


   submitted = st.form_submit_button('Predict')


if submitted:
   if model is None:
      st.error('Model not loaded')
   else:
      df = pd.DataFrame([{ # keep keys matching training columns
           'Age': Age,
           'TypeofContact': TypeofContact,
           'Occupation': Occupation,
           'ProductPitched': ProductPitched,
           'MaritalStatus': MaritalStatus,
           'Designation': Designation,
           'Gender': Gender,
           'MonthlyIncome': MonthlyIncome,
           'PitchSatisfactionScore': PitchSatisfactionScore,
            # Required missing features → fill default values
           'CustomerID': 0,
           'NumberOfPersonVisiting': 1,
           'NumberOfTrips': 1,
           'OwnCar': 0,
           'DurationOfPitch': 0,
           'NumberOfChildrenVisiting': 0,
           'NumberOfFollowups': 0,
           'CityTier': 1,
           'PreferredPropertyStar': 3,
           'Passport': 0,
           'Unnamed: 0': 0
       }])


      try:
          proba = model.predict_proba(df)[0, 1]
          pred = model.predict(df)[0]
          st.metric('Purchase probability', f'{proba*100:.2f}%')
          st.success('Will buy package' if pred == 1 else 'Will NOT buy package')
      except Exception as e:
          st.error(f'Prediction error: {e}')
