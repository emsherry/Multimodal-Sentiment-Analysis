# app_ui.py
import streamlit as st
import requests

st.title("Multimodal Sentiment Analyzer")

text_input = st.text_area("Enter text:")
audio_file = st.file_uploader("Upload WAV audio:", type=["wav"])

if st.button("Analyze"):
    files = {"audio": audio_file} if audio_file else {}
    data = {"text": text_input}
    
    response = requests.post("http://localhost:5000/analyze", data=data, files=files)
    if response.ok:
        result = response.json()
        st.write("**Text Sentiment:**", result['text_result'])
        st.write("**Audio Sentiment:**", result['audio_result'])
        st.write("**Fused Sentiment:**", result['fusion_result'])
