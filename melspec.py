import numpy as np
import streamlit as st
import librosa
import librosa.display
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. PROFESSIONAL UI & CATEGORIES
# -------------------------------------------------------------------------
st.set_page_config(page_title="PulsePitch AI | Signal Intelligence", page_icon="📈", layout="wide")

CAT3 = ["Positive", "Neutral", "Negative"]
CAT6 = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
CAT7 = ['Fear', 'Disgust', 'Neutral', 'Happy', 'Sad', 'Surprise', 'Angry']

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1a1c24; }
    [data-testid="stSidebar"] { background-color: #0b141a !important; }
    [data-testid="stSidebar"] * { color: #ffffff !important; font-size: 15px; }
    
    .hero-title {
        font-size: 100px; font-weight: 950; color: #111b21; letter-spacing: -5px; 
        line-height: 0.85; margin-bottom: 10px;
        background: -webkit-linear-gradient(#111b21, #4b5563);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    
    .pillar-card {
        background-color: #f8f9fa; padding: 30px; border-radius: 12px;
        border-top: 5px solid #007bff; height: 100%;
    }
    
    .status-panel {
        background-color: #111b21; color: white; padding: 25px; border-radius: 12px;
    }
    .stButton>button { background-color: #007bff; color: white; border-radius: 4px; font-weight: 700; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# 2. MODEL MANAGEMENT
# -------------------------------------------------------------------------
@st.cache_resource
def load_all_models():
    # Load primary, extended, and gender models
    m3 = load_model("model3.h5") if os.path.exists("model3.h5") else None
    m4 = load_model("model4.h5") if os.path.exists("model4.h5") else None
    mg = load_model("model_mw.h5") if os.path.exists("model_mw.h5") else None
    return m3, m4, mg

model3, model4, model_mw = load_all_models()

# -------------------------------------------------------------------------
# 3. ACCURACY-OPTIMIZED SIGNAL ENGINE
# -------------------------------------------------------------------------
@st.cache_data
def process_audio_high_res(path, limit=91):
    # Load and Trim: Alignment Logic
    y, sr = librosa.load(path, sr=22050)
    y, _ = librosa.effects.trim(y, top_db=20) 
    
    # Feature Extraction (40 MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # ACCURACY: Z-Score Normalization
    # Balances high-pitch bursts to prevent "Surprise" bias
    mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
    
    # Temporal Consistency: Lock to 91 frames
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
            em3 = st.checkbox("3-Tier Mode", True)
            em6 = st.checkbox("6-Tier Mode", True)
            em7 = st.checkbox("7-Tier Mode")
            gender_active = st.checkbox("Gender Logic")
        
        st.sidebar.caption("Enterprise v10.0 | Radhika Patil | CSE AI-ML")

    # --- SECTION 1: PROJECT INFORMATION ---
    if menu == "Project Information":
        st.markdown('<p class="hero-title">PULSEPITCH</p>', unsafe_allow_html=True)
        st.markdown("<p style='font-size: 24px; color: #4b5563;'>Enterprise framework for Acoustic Signal Intelligence.</p>", unsafe_allow_html=True)
        st.divider()

        col_v1, col_v2 = st.columns([2, 1])
        with col_v1:
            st.markdown("### Strategic Vision")
            st.write("PulsePitch utilizes 1D-Convolutional Neural Networks to map the vocal tract filter and frequency responses, enabling language-agnostic emotional detection.")
            
            p1, p2 = st.columns(2)
            with p1:
                st.markdown('<div class="pillar-card"><b>Digital Signal Processing</b><br><br>Utilizing 40-dimensional MFCC extraction at 22.05kHz.</div>', unsafe_allow_html=True)
            with p2:
                st.markdown('<div class="pillar-card"><b>Neural Inference</b><br><br>Powered by multi-layered CNNs with Metal-acceleration on M1 chips.</div>', unsafe_allow_html=True)
        
        with col_v2:
            st.markdown('<div class="status-panel"><b>PLATFORM SPECS</b><br><br>🟢 System: Operational<br>🟢 Normalization: Active<br>🟢 Optimization: Dual-Shape Logic<br>🟢 Accuracy: 90%+ Target</div>', unsafe_allow_html=True)

    # --- SECTION 2: ANALYSIS LABORATORY ---
    elif menu == "Analysis Laboratory":
        st.header("Acoustic Lab & Multi-Tier Inference")
        uploaded_file = st.file_uploader("Upload Audio Sample", type=['wav', 'mp3'])
        
        if uploaded_file:
            path = "active_sample.wav"
            with open(path, "wb") as f: f.write(uploaded_file.getbuffer())

            with st.spinner("Analyzing Signal Dynamics..."):
                # 1. Preprocessing and Feature Scaling
                mfccs, wav, sr = process_audio_high_res(path, limit=91)
                
                # 2. DUAL-INPUT SHAPE LOGIC
                # Shape A: (1, 40, 91) for Primary/Gender models
                input_A = np.expand_dims(mfccs, axis=0)
                
                # Shape B: (1, 91, 40) for 7-Tier model (Transpose to fix ValueError)
                input_B = np.expand_dims(mfccs.T, axis=0)
                
                # 3. Inference
                preds6 = model3.predict(input_A)[0] if model3 else np.zeros(6)

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
            st.subheader("Neural Classification Results")
            r1, r2, r3, r4 = st.columns(4)
            
            with r1:
                if em3:
                    pos, neu, neg = preds6[3] + preds6[5]*0.5, preds6[2] + preds6[4]*0.5, preds6[0] + preds6[1]
                    st.metric("3-Tier Status", CAT3[np.argmax([pos, neu, neg])])
            with r2:
                if em6: st.metric("Primary Detection", CAT6[np.argmax(preds6)])
            with r3:
                if em7 and model4:
                    # Use Input B (1, 91, 40) to satisfy the 7-tier model's layer requirements
                    p7 = model4.predict(input_B)[0]
                    st.metric("7-Tier Tier", CAT7[np.argmax(p7)])
            with r4:
                if gender_active and model_mw:
                    gp = model_mw.predict(input_A)[0]
                    st.metric("Gender ID", "Female" if gp.argmax() == 0 else "Male")

if __name__ == '__main__':
    main()