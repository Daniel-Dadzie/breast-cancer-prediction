import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --------------------------
# Page Configuration
# --------------------------
st.set_page_config(
    page_title="Breast Cancer AI Predictor",
    page_icon="🎗️",
    layout="wide"
)

# --------------------------
# Initialize Session State
# --------------------------
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# --------------------------
# Load and Apply Custom CSS
# --------------------------
@st.cache_data
def load_css(dark_mode):
    css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    with open(css_path, 'r') as f:
        css = f.read()

    if dark_mode:
        return f"<style>{css}</style>"
    else:
        # Light mode :root
        light_root = """
:root {
    --bg-primary: #ffffff;
    --bg-secondary: #f8f9fa;
    --bg-card: #ffffff;
    --text-primary: #1a1a1a;
    --text-secondary: #666666;
    --accent-primary: #667eea;
    --accent-secondary: #764ba2;
    --gradient-1: #667eea;
    --gradient-2: #764ba2;
    --shadow: rgba(0, 0, 0, 0.15);
}
.main-header p {
    color: #333333;
}
"""
        # Find and replace the dark :root
        root_start = css.find(':root {')
        root_end = css.find('}', root_start) + 1
        css = css[:root_start] + light_root + css[root_end:]
        return f"<style>{css}</style>"

# Apply theme CSS
st.markdown(load_css(st.session_state.dark_mode), unsafe_allow_html=True)

# --------------------------
# Load Model, Scaler & Metadata
# --------------------------
@st.cache_resource
def load_models():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    try:
        model_path = os.path.join(project_root, 'models', 'breast_cancer_model.pkl')
        scaler_path = os.path.join(project_root, 'models', 'scaler.pkl')
        metadata_path = os.path.join(project_root, 'models', 'model_metadata.pkl')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        metadata = joblib.load(metadata_path)
        return model, scaler, metadata
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please train the model first using the notebook.")
        st.stop()

model, scaler, metadata = load_models()

# --------------------------
# Horizontal Navbar
# --------------------------
st.markdown("""
<div class='navbar'>
    <div class='nav-container'>
        <div class='nav-links'>
""", unsafe_allow_html=True)

nav_cols = st.columns([1, 1, 1, 1, 1], gap="small")
pages = ['Home', 'How to Use','About' , 'Contact']

for i, page in enumerate(pages):
    with nav_cols[i]:
        if st.button(page, key=f"nav_{page}"):
            st.session_state.current_page = page
            st.rerun()

with nav_cols[4]:
    theme_icon = "🌙" if st.session_state.dark_mode else "☀️"
    if st.button(theme_icon, key="theme_toggle", help="Toggle Dark/Light Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown("""
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------
# Page Content
# --------------------------
if st.session_state.current_page == 'Home':
    # Main Header
    st.markdown("""
    <div class='main-header'>
        <h1>🎗️ Breast Cancer AI Predictor</h1>
        <p>Advanced machine learning analysis for early detection and risk assessment</p>
    </div>
    """, unsafe_allow_html=True)

    # Info Banner
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Analysis Type</div>
            <div class='metric-value'>30 Features</div>
            <div style='color: #666; font-size: 0.85rem;'>Comprehensive cellular analysis</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Prediction Speed</div>
            <div class='metric-value'>Instant</div>
            <div style='color: #666; font-size: 0.85rem;'>Real-time AI processing</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Model Accuracy</div>
            <div class='metric-value'>{:.0%}</div>
            <div style='color: #666; font-size: 0.85rem;'>Validated performance</div>
        </div>
        """.format(metadata['accuracy']), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model metrics in cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{metadata['accuracy']:.1%}</div>
            <div class='metric-label'>ACCURACY</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>30</div>
            <div class='metric-label'>FEATURES</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Features
    st.markdown("### 📊 Input Cellular Features")
    st.markdown("Enter the measurements from the diagnostic report. Default values represent dataset averages.")

    feature_names = metadata['features']

    # Original dataset means for default values
    original_means = [
        14.127, 19.290, 91.969, 654.889, 0.096, 0.104, 0.089, 0.049, 0.181, 0.063,
        0.405, 1.217, 2.866, 40.337, 0.007, 0.025, 0.032, 0.012, 0.021, 0.004,
        16.269, 25.677, 107.261, 880.583, 0.132, 0.254, 0.272, 0.115, 0.290, 0.084
    ]

    # Tabs for organization
    tab1, tab2, tab3 = st.tabs(["📈 Mean Features", "📉 Standard Error", "⚠️ Worst Case"])
    inputs = {}

    with tab1:
        st.markdown("##### Mean values of features (measurements 1-10)")
        col1, col2 = st.columns(2)
        for i in range(10):
            if i < 5:
                with col1:
                    inputs[feature_names[i]] = st.number_input(
                        f"{feature_names[i]}",
                        value=float(original_means[i]),
                        step=0.01,
                        key=f"home_{feature_names[i]}",
                        help=f"Mean value for {feature_names[i]}"
                    )
            else:
                with col2:
                    inputs[feature_names[i]] = st.number_input(
                        f"{feature_names[i]}",
                        value=float(original_means[i]),
                        step=0.01,
                        key=f"home_{feature_names[i]}",
                        help=f"Mean value for {feature_names[i]}"
                    )

    with tab2:
        st.markdown("##### Standard error of features (measurements 11-20)")
        for i in range(10, 20):
            inputs[feature_names[i]] = st.number_input(
                f"{feature_names[i]}",
                value=float(original_means[i]),
                step=0.001,
                key=f"home_{feature_names[i]}",
                help=f"Standard error for {feature_names[i]}"
            )

    with tab3:
        st.markdown("##### Worst case values of features (measurements 21-30)")
        col1, col2 = st.columns(2)
        for i in range(20, 30):
            if i < 25:
                with col1:
                    inputs[feature_names[i]] = st.number_input(
                        f"{feature_names[i]}",
                        value=float(original_means[i]),
                        step=0.01,
                        key=f"home_{feature_names[i]}",
                        help=f"Worst case value for {feature_names[i]}"
                    )
            else:
                with col2:
                    inputs[feature_names[i]] = st.number_input(
                        f"{feature_names[i]}",
                        value=float(original_means[i]),
                        step=0.01,
                        key=f"home_{feature_names[i]}",
                        help=f"Worst case value for {feature_names[i]}"
                    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction and Reset
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        reset_button = st.button("🔄 Reset Inputs", use_container_width=True)
    with col2:
        predict_button = st.button("🔮 Analyze & Predict", use_container_width=True, type="primary")

    if reset_button:
        st.rerun()

    if predict_button:
        with st.spinner("🔄 Analyzing cellular features..."):
            input_values = list(inputs.values())

            # Validation
            if len(inputs) != len(feature_names):
                st.error(f"❌ Expected {len(feature_names)} features, got {len(inputs)}")
                st.stop()

            if all(v == 0.0 for v in input_values):
                st.warning("⚠️ All inputs are zero. Enter realistic medical values for accurate prediction.")
                st.stop()

            if any(v < 0 for v in input_values):
                st.error("❌ Negative values are not allowed in medical measurements.")
                st.stop()

            # Check for unreasonably large values (basic sanity check)
            max_reasonable = [50, 50, 200, 2000, 1, 1, 1, 1, 1, 1, 1, 5, 5, 100, 1, 1, 1, 1, 1, 1, 50, 50, 200, 3000, 1, 1, 1, 1, 1, 1]
            if any(v > max_reasonable[i] for i, v in enumerate(input_values)):
                st.error("❌ Some input values seem unreasonably large. Please check your measurements.")
                st.stop()

            # Make prediction with error handling
            try:
                input_df = pd.DataFrame([input_values], columns=feature_names)
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)[0]
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.stop()

            st.markdown("<br>", unsafe_allow_html=True)

            # Display results
            if prediction[0] == 1:
                st.markdown(f"""
                <div class='result-card-benign'>
                    <div class='result-title'>✅ Low Risk — Benign Classification</div>
                    <p style='font-size: 1.1rem; margin-bottom: 1.5rem;'>
                        The AI analysis indicates a benign mass with high confidence.
                    </p>
                    <div class='probability-bar'>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                            <span style='font-weight: 600;'>Benign Probability</span>
                            <span style='font-weight: 700; font-size: 1.2rem;'>{prediction_proba[1]:.1%}</span>
                        </div>
                        <div style='background: rgba(255,255,255,0.5); height: 20px; border-radius: 10px; overflow: hidden;'>
                            <div style='background: #065f46; height: 100%; width: {prediction_proba[1]*100}%; transition: width 1s ease;'></div>
                        </div>
                    </div>
                    <div class='probability-bar'>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                            <span style='font-weight: 600;'>Malignant Probability</span>
                            <span style='font-weight: 700; font-size: 1.2rem;'>{prediction_proba[0]:.1%}</span>
                        </div>
                        <div style='background: rgba(255,255,255,0.5); height: 20px; border-radius: 10px; overflow: hidden;'>
                            <div style='background: #7f1d1d; height: 100%; width: {prediction_proba[0]*100}%; transition: width 1s ease;'></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.success("### 🌟 Recommended Actions")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **Lifestyle Maintenance:**
                    - ✓ Continue regular exercise routine
                    - ✓ Maintain balanced, nutritious diet
                    - ✓ Ensure adequate sleep (7-9 hours)
                    - ✓ Manage stress effectively
                    """)
                with col2:
                    st.markdown("""
                    **Medical Follow-up:**
                    - ✓ Schedule routine check-ups
                    - ✓ Perform monthly self-examinations
                    - ✓ Report any tissue changes immediately
                    - ✓ Keep medical records updated
                    """)
            else:
                st.markdown(f"""
                <div class='result-card-malignant'>
                    <div class='result-title'>⚠️ High Risk — Malignant Classification</div>
                    <p style='font-size: 1.1rem; margin-bottom: 1.5rem;'>
                        The AI analysis indicates potential malignancy. Immediate medical consultation is strongly advised.
                    </p>
                    <div class='probability-bar'>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                            <span style='font-weight: 600;'>Malignant Probability</span>
                            <span style='font-weight: 700; font-size: 1.2rem;'>{prediction_proba[0]:.1%}</span>
                        </div>
                        <div style='background: rgba(255,255,255,0.5); height: 20px; border-radius: 10px; overflow: hidden;'>
                            <div style='background: #7f1d1d; height: 100%; width: {prediction_proba[0]*100}%; transition: width 1s ease;'></div>
                        </div>
                    </div>
                    <div class='probability-bar'>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                            <span style='font-weight: 600;'>Benign Probability</span>
                            <span style='font-weight: 700; font-size: 1.2rem;'>{prediction_proba[1]:.1%}</span>
                        </div>
                        <div style='background: rgba(255,255,255,0.5); height: 20px; border-radius: 10px; overflow: hidden;'>
                            <div style='background: #065f46; height: 100%; width: {prediction_proba[1]*100}%; transition: width 1s ease;'></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.error("### 🏥 Immediate Action Required")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **Medical Priority:**
                    - 🚨 Schedule oncology consultation ASAP
                    - 🚨 Discuss biopsy and imaging options
                    - 🚨 Prepare complete medical history
                    - 🚨 Consider second opinion if needed
                    """)
                with col2:
                    st.markdown("""
                    **Support & Lifestyle:**
                    - ✓ Seek emotional support resources
                    - ✓ Optimize nutrition and rest
                    - ✓ Avoid tobacco and limit alcohol
                    - ✓ Stay informed about treatment options
                    """)

            st.markdown("<br>", unsafe_allow_html=True)
            st.info("💡 **Remember**: This is an AI-assisted screening tool. Always consult with qualified healthcare professionals for diagnosis and treatment decisions.")

elif st.session_state.current_page == 'About':
    st.markdown("""
    <div class='main-header'>
        <h1>🎗️ About This Tool</h1>
        <p>Learn more about our AI-powered breast cancer prediction system</p>
    </div>
    """, unsafe_allow_html=True)

    model_name = metadata.get('model_name', 'Unknown Model')

    st.markdown(f"""
    ## 🎯 Mission
    This AI-powered diagnostic tool analyzes 30 cellular features to predict breast cancer risk with high accuracy, helping in early detection and risk assessment.

    ## 🔬 Technology
    - **Algorithm**: {model_name}
    - **Training Data**: Wisconsin Breast Cancer Dataset
    - **Features**: 30 cellular measurements
    - **Classes**: Benign (0) / Malignant (1)
    - **Accuracy**: {metadata['accuracy']:.1%}

    ## 📊 Dataset Source
    [Wisconsin Breast Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
 
    ## 🤖 Model Development 
    The model was developed using Python's scikit-learn library, employing advanced preprocessing and hyperparameter tuning to achieve optimal performance. 
    ## 🛠️ Development Team  
    **GROUP TWO**
    """)

elif st.session_state.current_page == 'How to Use':
    st.markdown("""
    <div class='main-header'>
        <h1>📖 How to Use</h1>
        <p>Step-by-step guide to using the breast cancer prediction tool</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 🚀 Getting Started

    ### 1. **Input Features**
    Enter cellular measurements from your diagnostic report. Default values represent dataset averages.

    ### 2. **Feature Categories**
    - **📈 Mean Features**: Average measurements (1-10)
    - **📉 Standard Error**: Measurement variability (11-20)
    - **⚠️ Worst Case**: Maximum recorded values (21-30)

    ### 3. **Validation**
    Ensure all values are realistic and positive. The system performs automatic validation.

    ### 4. **Prediction**
    Click the "🔮 Analyze & Predict" button to get AI-powered analysis.

    ### 5. **Review Results**
    Examine the prediction results, probabilities, and recommended actions.

    ## 💡 Tips
    - Use realistic medical values for accurate predictions
    - All measurements should be positive
    - Consult healthcare professionals for any medical decisions
    """)

elif st.session_state.current_page == 'Contact':
    st.markdown("""
    <div class='main-header'>
        <h1>📞 Contact Us</h1>
        <p>Get in touch with the development team</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 👥 Development Team
    **GROUP TWO**

    ## 📧 Contact Information
    - **Email**: group2@gmail.com
    - **Project**: Breast Cancer AI Predictor
    - **Version**: 1.0.0

    ## 🔗 Links
    - [GitHub Repository](https://github.com/group2/breast-cancer-predictor)
    - [Documentation](https://docs.example.com)
    - [Support](https://support.example.com)

    ## 📝 Feedback
    We welcome your feedback and suggestions for improving this tool.

    ---
    *Built with Streamlit & Machine Learning*
    """)

# --------------------------
# Footer
# --------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p style='font-size: 0.9rem;'>
        🎗️ Early detection saves lives | Built with Streamlit & Machine Learning<br>
         <strong>Developed by:</strong> GROUP TWO
    </p>
</div>
""", unsafe_allow_html=True)
