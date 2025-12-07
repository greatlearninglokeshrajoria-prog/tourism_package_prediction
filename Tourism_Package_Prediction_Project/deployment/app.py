import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="greatlearninglokeshrajoria/tourism-package-prediction",
            filename="tourism_pkg_predition_model_v1.joblib"
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Streamlit UI
st.title("Tourism Package Prediction")
st.markdown("""
This application predicts whether a customer will purchase a **tourism package**
from a leading travel company based on their profile and interaction details.
Enter the customer details below to get a purchase prediction.
""")

# Sidebar for inputs
st.sidebar.header("Customer Profile")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=33)
    city_tier = st.selectbox("City Tier", ["1", "2", "3"], index=0)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)

with col2:
    num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=2)
    num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
    num_trips = st.number_input("Number of Previous Trips", min_value=0, max_value=20, value=2)

# Contact and Occupation
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business"])

# Demographics
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
own_car = st.selectbox("Owns Car", ["No", "Yes"])
passport = st.selectbox("Has Passport", ["No", "Yes"])

# Product and Preferences
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "King", "Super Deluxe"])
preferred_property_star = st.selectbox("Preferred Property Star Rating", [1, 2, 3, 4, 5], index=3)
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=3)
num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=1)

# Professional Details
designation = st.selectbox("Designation", [
    "Executive", "Manager", "Senior Manager", "AVP", "VP"
])
monthly_income = st.number_input("Monthly Income", min_value=10000, max_value=500000, value=25000)

# Assemble input data matching training features
input_data = pd.DataFrame([{
    'Age': int(age),
    'TypeofContact': type_of_contact,
    'CityTier': int(city_tier),
    'DurationOfPitch': int(duration_of_pitch),
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': int(num_person_visiting),
    'NumberOfFollowups': int(num_followups),
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': int(preferred_property_star),
    'MaritalStatus': marital_status,
    'NumberOfTrips': int(num_trips),
    'Passport': int(1 if passport == "Yes" else 0),
    'PitchSatisfactionScore': int(pitch_satisfaction_score),
    'OwnCar': int(1 if own_car == "Yes" else 0),
    'NumberOfChildrenVisiting': int(num_children_visiting),
    'Designation': designation,
    'MonthlyIncome': int(monthly_income)
}])

# use row index as ID
input_data["CustomerID"] = range(1, len(input_data) + 1)

# Display input summary
with st.expander("📋 Input Summary"):
    st.dataframe(input_data, use_container_width=True)

# Predict button
if st.button("🔮 Predict Package Purchase", type="primary", use_container_width=True):
    if model is not None:
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

            st.subheader("🎯 Prediction Result")
            if prediction == 1:
                st.success("✅ **Customer WILL purchase the tourism package!**")
            else:
                st.warning("❌ **Customer is unlikely to purchase**")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Purchase Probability", f"{probability[1]*100:.1f}%")
            with col2:
                st.metric("Prediction", "Yes" if prediction == 1 else "No")
            with col3:
                st.metric("Confidence", f"{max(probability)*100:.1f}%")

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error("Model could not be loaded. Please check the repository.")

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. Fill all customer details accurately
    2. **CityTier**: 1=Metro, 2=City, 3=Tier-3
    3. **ProductPitched**: Package type offered
    4. Click **Predict** to get results
    5. Higher purchase probability (>60%) indicates strong conversion potential
    """)

st.markdown("---")
st.markdown("*Built with tourism.csv dataset for package purchase prediction* [file:37]")
