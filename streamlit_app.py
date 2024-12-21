import streamlit as st
import base64
import io
from rvc_python.infer import RVCInference
import soundfile as sf
import numpy as np
import os

st.set_page_config(page_title="RVC Voice Converter", page_icon="🎤")

st.title("RVC Voice Converter")
st.markdown("Upload an audio file and a model to convert voices using RVC (Retrieval-based Voice Conversion).")

# Initialize session state
if 'rvc' not in st.session_state:
    st.session_state.rvc = RVCInference(device="cpu")  # Default to CPU for cloud deployment
    st.session_state.current_model = None

# File upload section
uploaded_audio = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
uploaded_model = st.file_uploader("Choose a model file (.pth)", type=['pth'])

# Model parameters
with st.expander("Advanced Settings"):
    f0_up_key = st.slider("Pitch Adjustment (semitones)", -12, 12, 0)
    protect = st.slider("Protection for Voiceless Consonants", 0.0, 1.0, 0.33)
    index_rate = st.slider("Feature Search Ratio", 0.0, 1.0, 0.5)
    filter_radius = st.slider("Median Filtering Radius", 0, 7, 3)
    rms_mix_rate = st.slider("Volume Envelope Mix Rate", 0.0, 1.0, 0.25)
    f0_method = st.selectbox("Pitch Extraction Method", ["harvest", "crepe", "rmvpe", "pm"])

if uploaded_model and uploaded_model != st.session_state.current_model:
    # Save the uploaded model temporarily
    model_path = os.path.join("/tmp", "temp_model.pth")
    with open(model_path, "wb") as f:
        f.write(uploaded_model.getvalue())

    try:
        st.session_state.rvc.load_model(model_path)
        st.session_state.current_model = uploaded_model
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

if st.button("Convert") and uploaded_audio and uploaded_model:
    try:
        # Read the audio file
        audio_bytes = uploaded_audio.getvalue()

        # Save temporary input file
        input_path = os.path.join("/tmp", "input_audio.wav")
        output_path = os.path.join("/tmp", "output_audio.wav")

        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        # Set conversion parameters
        st.session_state.rvc.set_params(
            f0up_key=f0_up_key,
            protect=protect,
            index_rate=index_rate,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            f0method=f0_method
        )

        # Convert the audio
        with st.spinner("Converting audio..."):
            st.session_state.rvc.infer_file(input_path, output_path)

        # Read the output file
        with open(output_path, "rb") as f:
            output_bytes = f.read()

        # Create a download button for the converted audio
        st.audio(output_bytes, format="audio/wav")
        st.download_button(
            label="Download Converted Audio",
            data=output_bytes,
            file_name="converted_audio.wav",
            mime="audio/wav"
        )

    except Exception as e:
        st.error(f"Error during conversion: {str(e)}")

# Add requirements.txt for Streamlit deployment
