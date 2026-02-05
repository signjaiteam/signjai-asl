import streamlit as st
import cv2
import joblib
import numpy as np
import os
import gdown
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 1. ตั้งค่าพื้นฐาน MediaPipe (ลดความละเอียดเพื่อความลื่น) ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5, # ปรับลดลงเล็กน้อยเพื่อให้ตรวจจับเร็วขึ้น
    model_complexity=0
)

# --- 2. โหลดโมเดลจาก Drive ---
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

# --- 3. UI Layout ---
st.set_page_config(page_title="Signjai ASL", layout="wide")
st.title("👋 Signjai - ภาษามืออัจฉริยะ")

# ส่วนที่ใช้แสดงผลแบบ Real-time โดยไม่ต้อง Refresh ทั้งหน้า
placeholder = st.empty()

if 'text_output' not in st.session_state:
    st.session_state.text_output = ""
if 'current_char' not in st.session_state:
    st.session_state.current_char = "-"

col1, col2 = st.columns([2, 1])

with col1:
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [lm.x for lm in hand_landmarks.landmark] + \
                            [lm.y for lm in hand_landmarks.landmark] + \
                            [lm.z for lm in hand_landmarks.landmark]
                
                try:
                    prediction = model.predict([landmarks[:63]]) # ปรับให้ตรงกับ 21 จุด (x,y,z)
                    st.session_state.current_char = label_encoder.inverse_transform(prediction)[0]
                except:
                    pass
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
    # แสดงกรอบผลลัพธ์
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 15px; border: 3px solid #31333f; text-align: center;">
            <p style="color: #666; font-weight: bold;">DETECTED</p>
            <h1 style="color: #31333f; font-size: 70px;">{st.session_state.current_char}</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # เพิ่มปุ่มฟังก์ชันที่หายไป
    st.write("---")
    if st.button("➕ บันทึกตัวอักษร"):
        if st.session_state.current_char != "-":
            st.session_state.text_output += st.session_state.current_char
    
    if st.button("🧹 ล้างข้อความ"):
        st.session_state.text_output = ""

    st.subheader("📝 ข้อความ:")
    st.success(st.session_state.text_output if st.session_state.text_output else "(ว่าง)")
