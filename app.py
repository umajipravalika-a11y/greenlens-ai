import os
import streamlit as st
from PIL import Image
import google.generativeai as genai

# Page Configuration
st.set_page_config(
    page_title="GreenLens AI - Smart E-Waste Assistant",
    page_icon="🌿",
    layout="wide",
)

# App Header Styling & Branding
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #059669 0%, #047857 100%); padding: 24px; border-radius: 12px; color: white; margin-bottom: 24px;">
        <h1 style="margin: 0; font-size: 2.2rem;">🌿 GreenLens AI</h1>
        <p style="margin: 4px 0 0 0; font-size: 1.05rem; opacity: 0.9;">AI-Powered Urban E-Waste & Recyclable Sorting Assistant</p>
        <p style="margin: 2px 0 0 0; font-size: 0.85rem; opacity: 0.75;">Vignan Institute of Technology & Science • Department of CSE (Data Science)</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Sidebar Configuration for API Key
st.sidebar.markdown("### 🔑 API & Engine Settings")
api_key_input = st.sidebar.text_input(
    "Enter Google Gemini API Key", type="password", help="Get a free key from Google AI Studio."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This version uses Google's multimodal Gemini Vision AI to instantly analyze"
    " e-waste, assess toxicity hazards, and provide localized disposal"
    " protocols."
)

# Main Layout
col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.markdown("#### 📷 Input Source")
    input_source = st.radio("Choose input mode:", ["Upload Image File", "Live Camera Feed"])

    uploaded_image = None
    if input_source == "Upload Image File":
        uploaded_image = st.file_uploader(
            "Upload e-waste item...", type=["jpg", "jpeg", "png", "webp"]
        )
    else:
        uploaded_image = st.camera_input("Capture item using camera")

with col2:
    st.markdown("#### 🔍 AI Analysis & Safety Protocol")
    output_placeholder = st.empty()

# Processing Logic
if uploaded_image is not None:
    image = Image.open(uploaded_image)

    with col1:
        st.image(image, caption="Captured / Uploaded Item", use_column_width=True)

    if not api_key_input:
        with col2:
            st.warning("⚠️ Please enter your Google Gemini API key in the sidebar to run the AI analysis.")
    else:
        try:
            # Configure Gemini API
            genai.configure(api_key=api_key_input)
            
            # Use Gemini Flash (fast & free tier friendly)
            model = genai.GenerativeModel("gemini-1.5-flash")

            with col2:
                with st.spinner("Analyzing item toxicity and sorting protocols..."):
                    prompt = (
                        "You are GreenLens AI, an expert municipal e-waste and recycling assistant. "
                        "Analyze this image. Identify if the item is an e-waste object (like a battery, circuit board, phone, cable) "
                        "or a recyclable item (like a plastic bottle, cardboard). "
                        "Provide your response strictly in this format:\n\n"
                        "**Item Name:** [Name of item]\n"
                        "**Category:** [E-Waste / Recyclable / General Waste]\n"
                        "**Hazard Assessment:** [Describe environmental/health risk like fire hazard or heavy metal leaching]\n"
                        "**Disposal Protocol:** [Clear step-by-step instructions on how to safely dispose of or recycle it]"
                    )

                    response = model.generate_content([prompt, image])
                    analysis_text = response.text

            # Display formatted result card
            with col2:
                st.markdown(
                    f"""
                    <div style="background-color: #1e293b; padding: 22px; border-radius: 10px; border-left: 6px solid #10b981; color: white;">
                        <h3 style="margin-top: 0; color: #38bdf8;">🧠 Gemini Vision Intelligence</h3>
                        <div style="font-size: 0.98rem; line-height: 1.6;">
                            {analysis_text.replace(chr(10), '<br>')}
                        </div>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

        except Exception as e:
            with col2:
                st.error(f"Error connecting to Gemini API: {e}")

else:
    with col2:
        st.markdown(
            """
            <div style="border: 2px dashed #475569; padding: 35px; border-radius: 10px; text-align: center; color: #94a3b8;">
                <h3>Waiting for image input...</h3>
                <p>Upload a photo or capture an item via camera to generate AI sorting instructions.</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

# Footer Metrics
st.markdown("---")
m1, m2, m3 = st.columns(3)
m1.metric("Vision Model", "Gemini 1.5 Flash", "Multimodal Intelligence")
m2.metric("Operating Cost", "₹0 / Free Tier", "Google AI Studio")
m3.metric("SDG Impact", "Goals 11, 12 & 13", "Sustainable Cities")
