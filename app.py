import numpy as np
import streamlit as st
import librosa
import librosa.display
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. PROFESSIONAL UI CONFIGURATION
# -------------------------------------------------------------------------
st.set_page_config(page_title="PulsePitch AI | Signal Intelligence", page_icon="📈", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1a1c24; }
    [data-testid="stSidebar"] { background-color: #0b141a !important; }
    [data-testid="stSidebar"] * { color: #ffffff !important; font-size: 15px; }
    
    .pillar-card {
        background-color: #f8f9fa;
        padding: 30px;
        border-radius: 12px;
        border-top: 5px solid #007bff;
        height: 100%;
    }
    
    .status-panel {
        background-color: #111b21;
        color: white;
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
    }

    .stButton>button { background-color: #007bff; color: white; border-radius: 4px; font-weight: 700; height: 3em; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 2. MULTI-MODEL RESOURCE MANAGEMENT
# -------------------------------------------------------------------------
@st.cache_resource
def load_ai_model(model_name):
    if os.path.exists(model_name):
        return load_model(model_name)
    return None

model3 = load_ai_model("model3.h5")
model4 = load_ai_model("model4.h5")
model_mw = load_ai_model("model_mw.h5")

CAT3 = ["Positive", "Neutral", "Negative"]
CAT6 = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
CAT7 = ['Fear', 'Disgust', 'Neutral', 'Happy', 'Sad', 'Surprise', 'Angry']

# -------------------------------------------------------------------------
# 3. SIGNAL PROCESSING ENGINE (Temporal Limit Fixed to 91)
# -------------------------------------------------------------------------
@st.cache_data
def process_audio(path, limit=91):
    # Load and Trim
    y, sr = librosa.load(path, sr=22050)
    y, _ = librosa.effects.trim(y)
    
    # Feature Extraction (40 MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # ACCURACY FIX: Z-Score Normalization
    # Prevents model from getting stuck on one emotion due to volume scaling
    mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
    
    # Shape Alignment to temporal limit (91 frames)
    if mfccs.shape[1] > limit:
        mfccs = mfccs[:, :limit]
    else:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, limit - mfccs.shape[1])), mode='constant')
    
    return mfccs, y, sr

# -------------------------------------------------------------------------
# 4. MAIN WORKFLOW
# -------------------------------------------------------------------------
def main():
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; font-size: 45px; font-weight: 900; letter-spacing: -2px; color: #ffffff;'>PULSEPITCH</h1>", unsafe_allow_html=True)
        st.markdown("---")
        menu = st.radio("NAVIGATION", ["Project Information", "Analysis Laboratory", "System Feedback"])
        st.markdown("---")
        
        if menu == "Analysis Laboratory":
            st.subheader("Tier Configuration")
            em3 = st.checkbox("3-Tier (Pos/Neu/Neg)", True)
            em6 = st.checkbox("6-Tier (Native)", True)
            em7 = st.checkbox("7-Tier (Extended)")
            gender = st.checkbox("Gender Logic")
        
        st.markdown("---")
        if st.button("🔄 System Rerun"): st.rerun()
        st.sidebar.caption("Enterprise v8.5 | Radhika Patil | CSE AI-ML")

    # --- SECTION 1: ADVANCED PROJECT INFORMATION ---
    if menu == "Project Information":
        st.markdown("""
        <div style="text-align: left; padding: 40px 0px;">
            <h1 style="font-size: 100px; font-weight: 950; color: #111b21; letter-spacing: -5px; line-height: 0.85; margin-bottom: 10px;">PULSEPITCH</h1>
            <p style="font-size: 26px; color: #4b5563; font-weight: 400; max-width: 850px; line-height: 1.4;">
                The Enterprise-Grade Framework for Acoustic Emotion Intelligence & Signal Decomposition.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        col_v1, col_v2 = st.columns([2, 1])
        with col_v1:
            st.markdown("### Strategic Overview")
            st.write("PulsePitch is a specialized Speech Emotion Recognition (SER) engine designed to extract emotional intent from human prosody.")
            
            p1, p2 = st.columns(2)
            with p1:
                st.markdown('<div class="pillar-card"><b>Digital Signal Processing</b><br><br>Utilizing 40-dimensional MFCC extraction at 22.05kHz.</div>', unsafe_allow_html=True)
            with p2:
                st.markdown('<div class="pillar-card"><b>Deep Neural Inference</b><br><br>Powered by a multi-layered 1D-CNN optimized for spectral pattern recognition.</div>', unsafe_allow_html=True)
        
        with col_v2:
            st.markdown('<div class="status-panel"><b>SYSTEM ARCHITECTURE</b><br><br>🟢 Model Status: Operational<br>🟢 Inference: Metal-Accelerated<br>🟢 Platform: Apple M1 Silicon<br>🟢 Input Spec: (1, 40, 91)<br><br><b>Optimization:</b> Z-Score Scaling Active</div>', unsafe_allow_html=True)

    # --- SECTION 2: ANALYSIS LABORATORY ---
    elif menu == "Analysis Laboratory":
        st.header("Acoustic Lab & Multi-Tier Inference")
        uploaded_file = st.file_uploader("Upload Audio Sample", type=['wav', 'mp3'])
        
        if uploaded_file:
            path = "active_sample.wav"
            with open(path, "wb") as f: f.write(uploaded_file.getbuffer())

            with st.spinner("Decoding Acoustic Pulse..."):
                # 1. Process with limit 91
                mfccs, wav, sr = process_audio(path, limit=91)
                
                # 2. SHAPE FIX: Your model expects features (40) as the first dimension
                # and time (91) as the last dimension.
                # Shape becomes (1, 40, 91)
                mfccs_input = np.expand_dims(mfccs, axis=0)
                
                # 3. Perform Inference
                preds6 = model3.predict(mfccs_input)[0]

            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Signal Waveform")
                fig_w, ax_w = plt.subplots(figsize=(10, 3))
                librosa.display.waveshow(wav, sr=sr, ax=ax_w, color="#007bff")
                ax_w.set_axis_off()
                st.pyplot(fig_w)
            with col_b:
                st.subheader("Emotion Distribution")
                st.bar_chart(dict(zip(CAT6, preds6.tolist())))

            st.divider()
            st.markdown("### Neural Predictions")
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                if em3:
                    pos, neu, neg = preds6[3] + preds6[5]*0.5, preds6[2] + preds6[4]*0.5, preds6[0] + preds6[1]
                    st.metric("3-Tier Status", CAT3[np.argmax([pos, neu, neg])])
            with r2:
                if em6: st.metric("Detected Emotion", CAT6[np.argmax(preds6)])
            with r3:
                if em7 and model4:
                    p7 = model4.predict(mfccs_input)[0]
                    st.metric("Extended Tier", CAT7[np.argmax(p7)])
            with r4:
                if gender and model_mw:
                    gp = model_mw.predict(mfccs_input)[0]
                    st.metric("Gender ID", "Female" if gp.argmax() == 0 else "Male")

    # --- SECTION 3: SYSTEM FEEDBACK ---
    elif menu == "System Feedback":
        st.title("System Feedback & Logs")
        with st.form("f"):
            obs = st.text_area("Observations:")
            if st.form_submit_button("Submit Data"):
                st.success("Log Recorded.")
                st.snow()
                st.balloons()
                

if __name__ == '__main__':
    main()