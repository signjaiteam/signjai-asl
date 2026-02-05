import streamlit as st
import cv2
import joblib
import numpy as np
import os
import gdown
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_autorefresh import st_autorefresh

# --- 1. ตั้งค่าพื้นฐาน MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    model_complexity=0
)

# --- 2. โหลดโมเดลจาก Google Drive ---
MODEL_PATH = "asl_rf.pkl"
GOOGLE_DRIVE_ID = "1OdCW3HuSmrCpB2YdN-5pjagEtI7Pa1MH"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
    
    data = joblib.load(MODEL_PATH)
    # เช็คว่าข้างในเป็น Dict หรือเป็น Model เลย
    if isinstance(data, dict):
        model = data.get('model') or data.get('classifier')
        le = data.get('label_encoder') or data.get('labels')
        return model, le
    return data[0], data[1] # กรณีเป็น list หรือ tuple

try:
    model, label_encoder = load_model()
except Exception as e:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {e}")
    st.stop()

# --- 3. ส่วนแสดงผล (UI) ---
st.set_page_config(page_title="Signjai ASL", layout="wide")
st.title("👋 Signjai - ภาษามืออัจฉริยะ")

if 'current_char' not in st.session_state:
    st.session_state.current_char = "-"

st_autorefresh(interval=800, key="refresh")

col1, col2 = st.columns([2, 1])

with col1:
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                prediction = model.predict([landmarks])
                st.session_state.current_char = label_encoder.inverse_transform(prediction)[0]
        else:
            st.session_state.current_char = "-"
        return frame

    webrtc_streamer(
        key="signjai",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 25px; border-radius: 15px; border: 4px solid #31333f; text-align: center;">
            <p style="color: #666; font-weight: bold; margin: 0;">DETECTED</p>
            <h1 style="color: #31333f; font-size: 80px; margin: 10px 0;">{st.session_state.current_char}</h1>
        </div>
    """, unsafe_allow_html=True)
