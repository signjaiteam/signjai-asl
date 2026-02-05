import streamlit as st
import cv2
import joblib
import numpy as np
import os
import gdown
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 0. ตั้งค่า MediaPipe และ Drawing ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils # ตัววาดเส้นจุดบนมือ
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)

# --- 1. โหลดโมเดลจาก Drive ---
MODEL_PATH = "asl_rf.pkl"
GOOGLE_DRIVE_ID = "1OdCW3HuSmrCpB2YdN-5pjagEtI7Pa1MH"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)
    data = joblib.load(MODEL_PATH)
    if isinstance(data, dict):
        return data.get('model') or data.get('classifier'), data.get('label_encoder') or data.get('labels')
    return data[0], data[1]

model, label_encoder = load_model()

# --- 2. UI Layout ---
st.set_page_config(page_title="Signjai ASL", layout="wide")
st.title("👋 Signjai - ภาษามืออัจฉริยะ")

if 'text_output' not in st.session_state:
    st.session_state.text_output = ""
if 'current_char' not in st.session_state:
    st.session_state.current_char = "-"

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Live Camera")
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        # กลับด้านภาพเหมือนกระจกเงา (คนใช้จะได้ไม่สับสน)
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # --- วาดเส้นบนมือ ---
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # --- เก็บข้อมูลส่งให้โมเดล (x, y, z 21 จุด = 63 ค่า) ---
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                try:
                    prediction = model.predict([landmarks])
                    st.session_state.current_char = label_encoder.inverse_transform(prediction)[0]
                except:
                    pass
        else:
            st.session_state.current_char = "-"
        
        # แปลงกลับเป็น BGR เพื่อแสดงผลใน WebRTC
        return frame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="signjai",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    # กรอบแสดงผลอักษรที่ตรวจจับได้
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 15px; border: 3px solid #31333f; text-align: center; margin-bottom: 20px;">
            <p style="color: #666; font-weight: bold; margin: 0;">DETECTED</p>
            <h1 style="color: #31333f; font-size: 80px; margin: 10px 0;">{st.session_state.current_char}</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # ปุ่มฟังก์ชัน
    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ บันทึกอักษร", use_container_width=True):
            if st.session_state.current_char != "-":
                st.session_state.text_output += st.session_state.current_char
    with c2:
        if st.button("🧹 ล้างข้อความ", use_container_width=True):
            st.session_state.text_output = ""

    st.subheader("📝 ข้อความสะสม:")
    st.success(st.session_state.text_output if st.session_state.text_output else "ยังไม่มีข้อมูล")
