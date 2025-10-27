# # # ui_app.py
# # import streamlit as st
# # import requests
# # import os
# # from typing import Dict

# # # Temporary environment setup (for Hugging Face / restricted hosts)
# # os.environ["STREAMLIT_HOME"] = "/tmp"
# # os.environ["STREAMLIT_RUNTIME_DIR"] = "/tmp"

# # # -------------------------
# # # 1) Basic page setup
# # # -------------------------
# # st.set_page_config(
# #     page_title="Absenteeism Risk Dashboard",
# #     page_icon="📊",
# #     layout="wide",
# #     initial_sidebar_state="expanded",
# # )

# # # <-- Change this to your backend deployment if different -->
# # BACKEND_URL = "http://127.0.0.1:8000"

# # st.title("📉 Absenteeism Risk Assessment Dashboard")
# # st.markdown(
# #     "A simple tool to help HR/team leader to spot employees who **may** be at higher risk of frequent absence. "
# #     "Use the prediction as a **supporting signal** — always combine with human judgement."
# # )
# # st.divider()

# # # -------------------------
# # # 2) Sidebar: human-friendly inputs
# # # -------------------------
# # st.sidebar.header("🧾 Employee information (enter details below)")
# # st.sidebar.caption("Values are translated for the model automatically.")

# # reason_map = {
# #     1: "Infectious diseases",
# #     2: "Injury / poisoning",
# #     3: "Respiratory diseases",
# #     4: "Digestive diseases",
# #     5: "Pregnancy-related",
# #     23: "Medical consultation",
# #     28: "Other reasons",
# # }
# # month_map = {i: name for i, name in enumerate(
# #     ["January", "February", "March", "April", "May", "June",
# #      "July", "August", "September", "October", "November", "December"], start=1)}
# # season_map = {1: "Summer", 2: "Autumn", 3: "Winter", 4: "Spring"}
# # education_map = {1: "High school", 2: "Graduate", 3: "Postgraduate", 4: "PhD"}

# # reason = st.sidebar.selectbox("Reason for absence", list(reason_map.keys()),
# #                               format_func=lambda k: reason_map.get(k, str(k)))
# # month = st.sidebar.selectbox("Month of absence", list(month_map.keys()),
# #                              format_func=lambda k: month_map.get(k, str(k)))
# # dow = st.sidebar.selectbox("Day of the week", [2, 3, 4, 5, 6],
# #                            format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri"][x - 2])
# # season = st.sidebar.selectbox("Season", list(season_map.keys()),
# #                               format_func=lambda k: season_map.get(k, str(k)))
# # edu = st.sidebar.selectbox("Education level", list(education_map.keys()),
# #                            format_func=lambda k: education_map.get(k, str(k)))
# # discipline = st.sidebar.radio("Disciplinary failure?", [0, 1],
# #                               format_func=lambda x: "Yes" if x else "No")
# # drinker = st.sidebar.radio("Social drinker?", [0, 1], format_func=lambda x: "Yes" if x else "No")
# # smoker = st.sidebar.radio("Social smoker?", [0, 1], format_func=lambda x: "Yes" if x else "No")
# # service_time = st.sidebar.slider("Service time (years)", 0, 40, 8)
# # age = st.sidebar.slider("Age", 18, 70, 35)
# # bmi = st.sidebar.slider("Body Mass Index (BMI)", 15.0, 40.0, 24.5)


# # # -------------------------
# # # 3) Build features dict (include both name variations the model may expect)
# # # -------------------------
# # # Some saved pipelines expect slightly different column names; include both safe keys.
# # features: Dict[str, object] = {
# #     "Reason for absence": reason,
# #     "Month of absence": month,
# #     "Day of the week": dow,      # many pipelines use 'Day of the week'
# #     "Day of week": dow,          # some pipelines use 'Day of week'
# #     "Seasons": season,
# #     "Education": edu,
# #     "Disciplinary failure": discipline,
# #     "Social drinker": drinker,
# #     "Social smoker": smoker,
# #     "Service time": service_time,
# #     "Age": age,
# #     "BMI": bmi,
# #     "Body mass index": bmi,  # alternative naming
# #     # Add safe defaults for numeric features that the model may expect
# #     "Transportation expense": 200,
# #     "Distance from Residence to Work": 10,
# #     "Work load Average/day": 250,
# #     "Hit target": 95,
# #     "Son": 1,
# #     "Pet": 0,
# #     "Weight": 70,
# #     "Height": 170,
# # }

# # # -------------------------
# # # 4) Prediction area (big clear clickable button)
# # # -------------------------
# # st.subheader("🎯 Predict absenteeism risk")

# # left, right = st.columns([1, 1])

# # # Make the predict button visually easy to spot (CSS)
# # st.markdown(
# #     """
# #     <style>
# #     div.stButton > button:first-child {
# #         background-color: #0078D4;
# #         color: white;
# #         border-radius: 8px;
# #         height: 44px;
# #         width: 100%;
# #         font-size: 16px;
# #         font-weight: 600;
# #     }
# #     div.stButton > button:hover {
# #         transform: translateY(-1px);
# #     }
# #     </style>
# #     """,
# #     unsafe_allow_html=True,
# # )

# # with left:
# #     if st.button("🔍 Click here to predict absenteeism risk"):
# #         # show spinner while calling backend
# #         with st.spinner("Checking model and generating prediction…"):
# #             try:
# #                 resp = requests.post(f"{BACKEND_URL}/predict", json={"features": features}, timeout=10)
# #             except Exception as e:
# #                 st.error(f"⚠️ Could not contact backend at {BACKEND_URL}: {e}")
# #                 resp = None

# #         if resp is None:
# #             st.stop()

# #         # handle response
# #         if resp.status_code == 200:
# #             data = resp.json()
# #             proba = float(data.get("proba", 0.0))
# #             label = int(data.get("label", 0))
# #             threshold = float(data.get("threshold", 0.5))
# #             explanations = data.get("explanations", []) or []

# #             # Friendly headline
# #             if label == 1:
# #                 st.error(f"🚨 High risk — {proba:.1%} probability")
# #                 st.write("This prediction indicates the employee is **more likely** to be frequently absent. "
# #                          "Consider supportive outreach (check-in, workload review, wellbeing resources).")
# #             else:
# #                 st.success(f"✅ Low risk — {proba:.1%} probability")
# #                 st.write("Attendance looks stable. No immediate action suggested from the model.")

# #             # Threshold display (human friendly)
# #             st.metric("Decision cutoff (model threshold)", f"{threshold * 100:.0f}%")
# #             st.caption("Predictions with probability above the cutoff are labelled **High risk**.")

# #             st.caption("This prediction is based on multiple attendance-related factors. "
# #            "Feature-by-feature explanations are not shown in this version.")

# #             st.markdown("---")
# #             st.caption("⚖️ Note: This is a predictive signal, not a decision. Combine with HR context and review before action.")
# #         else:
# #             # Show backend error message (friendly)
# #             try:
# #                 err = resp.json()
# #                 # some errors arrive as {"detail": "..."}; show politely
# #                 detail = err.get("detail", str(err))
# #             except Exception:
# #                 detail = resp.text
# #             st.error(f"Prediction failed: {detail}")

# # with right:
# #     st.info(
# #         """
# #         **How to read the result**
# #         - The percentage is the model's estimate of the *chance* the employee will be frequently absent.
# #         - **High risk (red)** → consider follow-up and support.
# #         - **Low risk (green)** → no action required from model alone.
# #         - Always verify with human context (manager notes, leave policies).
# #         """
# #     )

# # st.divider()

# # # # -------------------------
# # # 5) About the model & Performance FIRST (click to expand)
# # # -------------------------
# # st.subheader("📘 Model overview & performance")

# # # Fetch model-info + metrics safely
# # model_info, metrics_resp = {}, {}
# # try:
# #     model_info = requests.get(f"{BACKEND_URL}/model-info", timeout=5).json()
# #     metrics_resp = requests.get(f"{BACKEND_URL}/metrics", timeout=5).json()
# # except Exception:
# #     pass

# # # ----- EXPANDER 1: HOW WELL MODEL PERFORMS -----
# # with st.expander("📈 How well does the model perform?", expanded=False):
# #     before = metrics_resp.get("overall_before", {})
# #     after = metrics_resp.get("overall_after", {})
# #     if before and after:
# #         acc = after.get("acc", 0)
# #         f1 = after.get("f1", 0)
# #         auc = after.get("auc", 0)

# #         st.success(f"✅ The model predicts correctly about **{acc*100:.0f}%** of the time.")
# #         st.info(f"⚖️ It balances errors with an overall stability (F1) of **{f1*100:.0f}%**.")
# #         st.caption(f"✨ It distinguishes high-risk vs low-risk employees correctly about **{auc*100:.0f}%** of the time.")
# #     else:
# #         st.info("Performance information is not available for this deployment.")

# # # ----- EXPANDER 2: MODEL DETAILS -----
# # with st.expander("🔍 Model overview", expanded=False):
# #     st.markdown(f"**Purpose:** {model_info.get('purpose', '')}")
# #     st.markdown(f"**Intended user:** {model_info.get('intended_user', '')}")
# #     st.markdown(f"**Decision supported:** {model_info.get('decision_supported', '')}")
# #     st.caption("⚠️ This tool should not be used for hiring, firing, or disciplinary action.")

# # # ----- EXPANDER 3: FAIRNESS -----
# # with st.expander("⚖️ Fairness & transparency", expanded=False):
# #     fair = model_info.get("fairness", {})
# #     fair_after = metrics_resp.get("fairness_after", {})

# #     st.markdown(f"**Protected attribute:** {fair.get('protected_attribute','Age (≥40)')}")
# #     st.markdown("**What we did to reduce bias:**")
# #     mapping = {
# #         "drop_age_feature": "Removed age from inputs",
# #         "reweight_by_AxY": "Balanced training data",
# #         "probability_calibration": "Calibrated predictions for fairness"
# #     }
# #     for m in fair.get("mitigations", []):
# #         st.markdown(f"- {mapping.get(m,m)}")

# #     if fair_after:
# #         st.markdown("**Fairness after mitigation:**")
# #         name_map = {"SPD":"Parity gap","EOD":"TPR gap","FPR_diff":"False alert gap"}
# #         for k,v in fair_after.items():
# #             st.metric(label=name_map.get(k,k), value=f"{v:.3f}")
# #         st.caption("Closer to 0 → fairer between groups.")
# #     else:
# #         st.info("Fairness metrics not available.")

# # st.divider()



# # # -------------------------
# # # 6) Footer / small help
# # # -------------------------
# # st.markdown(
# #     """
# #     <small>
# #     Built for course assignment — uses a predictive model. Predictions are probabilistic signals, not final decisions..
# #     </small>
# #     """,
# #     unsafe_allow_html=True,
# # )


# import streamlit as st
# import requests
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# import json

# # Page configuration
# st.set_page_config(
#     page_title="Absenteeism Risk Dashboard",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .risk-high {
#         background-color: #ffcccc;
#         padding: 10px;
#         border-radius: 5px;
#         border-left: 5px solid #ff4444;
#     }
#     .risk-low {
#         background-color: #ccffcc;
#         padding: 10px;
#         border-radius: 5px;
#         border-left: 5px solid #44ff44;
#     }
#     .explanation-box {
#         background-color: #f0f2f6;
#         padding: 15px;
#         border-radius: 5px;
#         margin: 10px 0;
#     }
#     .metric-box {
#         background-color: white;
#         padding: 15px;
#         border-radius: 5px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         margin: 5px 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Backend URL (update with your Render URL)
# BACKEND_URL = "https://absenteeism-fairness-a3-1.onrender.com"

# def main():
#     st.markdown('<div class="main-header">🏢 Absenteeism Risk Prediction Dashboard</div>', unsafe_allow_html=True)
    
#     # Sidebar for global explanations
#     with st.sidebar:
#         st.header("📈 Model Insights")
        
#         if st.button("View Global Model Analysis"):
#             try:
#                 response = requests.get(f"{BACKEND_URL}/model_info")
#                 if response.status_code == 200:
#                     model_info = response.json()
#                     display_global_explanations(model_info)
#                 else:
#                     st.error("Could not fetch model information")
#             except Exception as e:
#                 st.error(f"Error fetching model info: {e}")
        
#         if st.button("View Fairness Report"):
#             try:
#                 response = requests.get(f"{BACKEND_URL}/fairness")
#                 if response.status_code == 200:
#                     fairness_info = response.json()
#                     display_fairness_analysis(fairness_info)
#                 else:
#                     st.error("Could not fetch fairness information")
#             except Exception as e:
#                 st.error(f"Error fetching fairness info: {e}")
    
#     # Main content area
#     tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Explanation", "ℹ️ About"])
    
#     with tab1:
#         display_prediction_interface()
    
#     with tab2:
#         display_explanation_dashboard()
    
#     with tab3:
#         display_about_section()

# def display_prediction_interface():
#     st.header("Employee Information Input")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         reason_for_absence = st.selectbox(
#             "Reason for Absence",
#             ["Routine checkup", "Consultation", "Illness", "Injury", "Medical procedure"]
#         )
        
#         education = st.selectbox(
#             "Education Level",
#             ["High school", "Graduate", "Postgraduate", "PhD"]
#         )
        
#         season = st.selectbox(
#             "Season",
#             ["Summer", "Autumn", "Winter", "Spring"]
#         )
    
#     with col2:
#         social_drinker = st.radio("Social Drinker", ["Yes", "No"])
#         social_smoker = st.radio("Social Smoker", ["Yes", "No"])
#         age = st.slider("Age", min_value=18, max_value=65, value=35)
#         service_time = st.slider("Service Time (years)", min_value=1, max_value=30, value=5)
    
#     if st.button("Predict Absenteeism Risk", type="primary"):
#         with st.spinner("Analyzing employee data..."):
#             try:
#                 # Prepare data for API call
#                 employee_data = {
#                     "reason_for_absence": reason_for_absence,
#                     "education": education,
#                     "season": season,
#                     "social_drinker": social_drinker,
#                     "social_smoker": social_smoker,
#                     "age": age,
#                     "service_time": service_time
#                 }
                
#                 # Make API call
#                 response = requests.post(f"{BACKEND_URL}/predict", json=employee_data)
                
#                 if response.status_code == 200:
#                     result = response.json()
#                     display_prediction_result(result)
                    
#                     # Store result in session state for explanation tab
#                     st.session_state.prediction_result = result
#                     st.session_state.employee_data = employee_data
                    
#                 else:
#                     st.error("Error making prediction. Please try again.")
                    
#             except Exception as e:
#                 st.error(f"Connection error: {e}")
#                 st.info("Please ensure the backend server is running.")

# def display_prediction_result(result):
#     st.markdown("---")
#     st.header("Prediction Result")
    
#     probability = result['probability']
#     risk_level = result['risk_level']
    
#     # Display risk indicator
#     if risk_level == "High":
#         st.markdown(f"""
#         <div class="risk-high">
#             <h3>🚨 High Absenteeism Risk</h3>
#             <p><strong>Probability:</strong> {probability:.1%}</p>
#             <p>This employee shows characteristics associated with higher absenteeism risk.</p>
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown(f"""
#         <div class="risk-low">
#             <h3>✅ Low Absenteeism Risk</h3>
#             <p><strong>Probability:</strong> {probability:.1%}</p>
#             <p>This employee shows characteristics associated with stable attendance patterns.</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Confidence indicator
#     confidence = result['explanation']['confidence']
#     st.progress(confidence, text=f"Model Confidence: {confidence:.1%}")
    
#     # Quick summary of top factors
#     st.subheader("Key Influencing Factors")
#     top_factors = result['top_features'][:3]
    
#     for factor in top_factors:
#         direction = "↑ Increases risk" if factor['impact'] > 0 else "↓ Decreases risk"
#         st.write(f"• **{factor['feature']}**: {direction}")

# def display_explanation_dashboard():
#     st.header("Detailed Explanation Dashboard")
    
#     if 'prediction_result' not in st.session_state:
#         st.info("Please make a prediction first to see detailed explanations.")
#         return
    
#     result = st.session_state.prediction_result
#     employee_data = st.session_state.employee_data
    
#     # Create tabs for different explanation types
#     exp_tab1, exp_tab2, exp_tab3, exp_tab4 = st.tabs([
#         "🔍 Local Explanation", 
#         "🔄 What-If Analysis", 
#         "📈 Feature Impact",
#         "🎯 Model Confidence"
#     ])
    
#     with exp_tab1:
#         display_local_explanation(result)
    
#     with exp_tab2:
#         display_what_if_analysis(result, employee_data)
    
#     with exp_tab3:
#         display_feature_impact(result)
    
#     with exp_tab4:
#         display_confidence_analysis(result)

# def display_local_explanation(result):
#     st.subheader("Why this prediction?")
    
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         # SHAP values visualization
#         features = [feat['feature'] for feat in result['top_features']]
#         impacts = [feat['impact'] for feat in result['top_features']]
        
#         colors = ['red' if impact > 0 else 'green' for impact in impacts]
        
#         fig = go.Figure(go.Bar(
#             x=impacts,
#             y=features,
#             orientation='h',
#             marker_color=colors
#         ))
        
#         fig.update_layout(
#             title="Top Feature Contributions to Prediction",
#             xaxis_title="Impact on Prediction",
#             yaxis_title="Features",
#             showlegend=False
#         )
        
#         st.plotly_chart(fig, use_container_width=True)
    
#     with col2:
#         st.subheader("Explanation Summary")
#         st.metric("Base Probability", f"{result['explanation']['decision_boundary']:.0%}")
#         st.metric("Confidence Level", f"{result['explanation']['confidence']:.1%}")
        
#         st.info("""
#         **How to read:**
#         - Red bars increase absenteeism risk
#         - Green bars decrease absenteeism risk
#         - Longer bars = stronger influence
#         """)

# def display_what_if_analysis(result, employee_data):
#     st.subheader("What-If Scenario Analysis")
    
#     st.write("Explore how changing factors would affect the prediction:")
    
#     # Display counterfactuals
#     if result['counterfactuals']:
#         for i, cf in enumerate(result['counterfactuals']):
#             with st.expander(f"Scenario {i+1}: {cf['scenario']}"):
#                 st.write(f"**Expected Impact:** {cf['expected_impact']}")
#                 st.write(f"**Reason:** {cf['reason']}")
    
#     # Interactive what-if tool
#     st.subheader("Interactive What-If Tool")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         new_service_time = st.slider(
#             "Adjust Service Time", 
#             min_value=1, 
#             max_value=30, 
#             value=employee_data['service_time'],
#             key="what_if_service"
#         )
        
#         new_drinker = st.radio(
#             "Adjust Drinking Habit",
#             ["Yes", "No"],
#             index=0 if employee_data['social_drinker'] == "Yes" else 1,
#             key="what_if_drinker"
#         )
    
#     with col2:
#         if st.button("Simulate New Scenario", type="secondary"):
#             # Create modified scenario
#             modified_data = employee_data.copy()
#             modified_data['service_time'] = new_service_time
#             modified_data['social_drinker'] = new_drinker
            
#             # In a real implementation, you would call the backend with modified data
#             st.success("Scenario updated! (Note: This is a simulation)")
#             st.write(f"Original risk: {result['risk_level']}")
#             st.write("Modified factors would typically change the risk assessment.")

# def display_feature_impact(result):
#     st.subheader("Feature Impact Analysis")
    
#     # Create detailed feature impact table
#     feature_data = []
#     for feature in result['top_features']:
#         feature_data.append({
#             'Feature': feature['feature'],
#             'Impact Score': abs(feature['impact']),
#             'Direction': 'Increases Risk' if feature['impact'] > 0 else 'Decreases Risk',
#             'Magnitude': 'High' if abs(feature['impact']) > 0.1 else 'Medium' if abs(feature['impact']) > 0.05 else 'Low'
#         })
    
#     df = pd.DataFrame(feature_data)
#     st.dataframe(df, use_container_width=True)
    
#     # Impact distribution
#     st.subheader("Impact Distribution")
#     fig = px.pie(
#         df, 
#         names='Direction', 
#         values='Impact Score',
#         title="Overall Impact Distribution"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# def display_confidence_analysis(result):
#     st.subheader("Model Confidence Analysis")
    
#     confidence = result['explanation']['confidence']
    
#     # Confidence gauge
#     fig = go.Figure(go.Indicator(
#         mode = "gauge+number+delta",
#         value = confidence * 100,
#         domain = {'x': [0, 1], 'y': [0, 1]},
#         title = {'text': "Prediction Confidence"},
#         gauge = {
#             'axis': {'range': [None, 100]},
#             'bar': {'color': "darkblue"},
#             'steps': [
#                 {'range': [0, 50], 'color': "lightgray"},
#                 {'range': [50, 80], 'color': "yellow"},
#                 {'range': [80, 100], 'color': "lightgreen"}
#             ],
#             'threshold': {
#                 'line': {'color': "red", 'width': 4},
#                 'thickness': 0.75,
#                 'value': 90
#             }
#         }
#     ))
    
#     fig.update_layout(height=300)
#     st.plotly_chart(fig, use_container_width=True)
    
#     # Confidence interpretation
#     if confidence > 0.8:
#         st.success("High confidence: The model is very certain about this prediction.")
#     elif confidence > 0.6:
#         st.warning("Medium confidence: The model is reasonably certain but some uncertainty remains.")
#     else:
#         st.error("Low confidence: The model is uncertain about this prediction. Consider additional factors.")

# def display_global_explanations(model_info):
#     st.header("Global Model Explanations")
    
#     st.subheader("Model Performance")
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric("Accuracy", f"{model_info['accuracy']:.1%}")
#     with col2:
#         st.metric("Precision", f"{model_info['precision']:.1%}")
#     with col3:
#         st.metric("Recall", f"{model_info['recall']:.1%}")
#     with col4:
#         st.metric("F1-Score", f"{model_info['f1_score']:.1%}")
    
#     st.subheader("Top Global Feature Importance")
    
#     # Feature importance chart
#     features = [feat['feature'] for feat in model_info['feature_importance']]
#     importance = [feat['importance'] for feat in model_info['feature_importance']]
#     directions = [feat['direction'] for feat in model_info['feature_importance']]
    
#     fig = px.bar(
#         x=importance,
#         y=features,
#         orientation='h',
#         title="Global Feature Importance",
#         labels={'x': 'Importance', 'y': 'Features'},
#         color=directions,
#         color_discrete_map={'positive': 'red', 'negative': 'green'}
#     )
    
#     st.plotly_chart(fig, use_container_width=True)

# def display_fairness_analysis(fairness_info):
#     st.header("Fairness Analysis")
    
#     st.write("**Protected Attribute:** Age (≥40 years)")
    
#     st.subheader("Mitigation Techniques Applied")
#     for technique in fairness_info['mitigation_techniques']:
#         st.write(f"• {technique}")
    
#     st.subheader("Fairness Metrics")
#     metrics = fairness_info['metrics']
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.metric("Statistical Parity Difference", f"{metrics['statistical_parity_difference']:.3f}")
#         st.metric("Equal Opportunity Difference", f"{metrics['equal_opportunity_difference']:.3f}")
    
#     with col2:
#         st.metric("False Positive Rate Difference", f"{metrics['false_positive_rate_difference']:.3f}")
#         st.metric("Disparate Impact", f"{metrics['disparate_impact']:.3f}")
    
#     st.info(fairness_info['interpretation'])

# def display_about_section():
#     st.header("About This Dashboard")
    
#     st.markdown("""
#     ### 🎯 Purpose
#     This dashboard provides HR professionals with AI-powered insights into employee absenteeism risk, 
#     emphasizing **transparency**, **fairness**, and **explainability**.
    
#     ### 🔍 Explanation Features
    
#     **Local Explanations** (For individual predictions):
#     - SHAP values showing feature contributions
#     - Top influencing factors with direction
#     - Confidence intervals and uncertainty measures
    
#     **Global Explanations** (For model understanding):
#     - Overall feature importance
#     - Model performance metrics
#     - Fairness analysis across demographics
    
#     **What-If Analysis**:
#     - Counterfactual explanations
#     - Interactive scenario testing
#     - Impact simulation
    
#     ### 🛡️ Fairness & Ethics
#     - Protected attribute: Age (≥40 years)
#     - Mitigation: Feature removal, reweighting, threshold calibration
#     - Continuous monitoring of fairness metrics
    
#     ### 🚀 Technology Stack
#     - **Backend**: FastAPI with scikit-learn
#     - **Frontend**: Streamlit
#     - **Explainability**: SHAP, counterfactual analysis
#     - **Hosting**: Render (backend), Hugging Face (frontend)
#     """)
    
#     st.success("""
#     **Note**: This tool is designed for **supportive interventions** not punitive actions. 
#     Always combine AI insights with human judgment and contextual understanding.
#     """)

# if __name__ == "__main__":
#     # main()


