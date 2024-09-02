import streamlit as st
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def web_app():
    st.set_page_config(page_title='Risk of VAP in IS Patients')
    # rf = joblib.load('./strokevapweb/rf7.pkl')
    rf = joblib.load('./rf7.pkl')

    class Subject:
        def __init__(self, SBP, DBP, INR, LOS_before_MV, Antibiotic_counts, Suctioning_counts, Dysphagia):
            self.SBP = SBP
            self.DBP = DBP
            self.INR = INR
            self.LOS_before_MV = LOS_before_MV
            self.Antibiotic_counts = Antibiotic_counts
            self.Suctioning_counts = Suctioning_counts
            self.Dysphagia = Dysphagia

        def make_predict(self):
            subject_data = {
                "SBP": [self.SBP],
                "DBP": [self.DBP],
                "INR": [self.INR],
                "LOS_before_MV": [self.LOS_before_MV],
                "Antibiotic_counts": [self.Antibiotic_counts],
                "Suctioning_counts": [self.Suctioning_counts],
                "Dysphagia": [self.Dysphagia],
            }

            df_subject = pd.DataFrame(subject_data)
            prediction = rf.predict_proba(df_subject)[:, 1]
            cutoff = 0.3325556
            if prediction >= cutoff:
                adjusted_prediction = (prediction - cutoff) * (0.5 / (1 - cutoff)) + 0.5
                adjusted_prediction = np.clip(adjusted_prediction, 0.5, 1)
            else:
                adjusted_prediction = prediction * (0.5 / cutoff)
                adjusted_prediction = np.clip(adjusted_prediction, 0, 0.5)

            adjusted_prediction = np.round(adjusted_prediction * 100, 2)
            st.write(f"""
                <div class='all'>
                    <p style='text-align: center; font-size: 20px;'>
                        <b>Based on the information provided, the model predicts the risk of ventilator-associated pneumonia is {adjusted_prediction} %</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)

            explainer = shap.Explainer(rf)
            shap_values = explainer.shap_values(df_subject)
            shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], df_subject.iloc[0, :], matplotlib=True)
            st.pyplot(plt.gcf())

    st.markdown(f"""
                <div class='all'>
                    <h1 style='text-align: center;'>Web App - Predicting Ventilator-Associated Pneumonia Risk in Ischemic Stroke Patients</h1>
                    <p class='intro'></p>
                </div>
                """, unsafe_allow_html=True)
    SBP = st.number_input("Systolic Blood Pressure (mmHg)", min_value=60, max_value=200, value=100)
    DBP = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30, max_value=120, value=60)
    INR = st.number_input("International Normalized Ratio (INR)", min_value=0.5, max_value=7.0, value=1.2, format="%.1f")
    LOS_before_MV = st.slider("Length of Stay Before Mechanical Ventilation (days)", 0, 31, 6)
    Antibiotic_counts = st.slider("Number of Antibiotic Uses", 0, 10, 4)
    Suctioning_counts = st.slider("Number of Suctioning Procedures", 0, 15, 10)
    Dysphagia = st.slider("Dysphagia (0: No, 1: Yes)", min_value=0, max_value=1, value=1)
    # dysphagia = st.selectbox("Dysphagia", options=[0, 1], index=0)

    if st.button(label="Submit"):
        user = Subject(SBP, DBP, INR, LOS_before_MV, Antibiotic_counts, Suctioning_counts, Dysphagia)
        user.make_predict()


web_app()
