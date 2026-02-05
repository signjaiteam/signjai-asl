import streamlit as st
import cv2
import joblib
import numpy as np
import os
import gdown
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import queue # เพิ่มตัวนี้เพื่อใช้ส่งข้อมูลจากกล้องมาหน้าเว็บ

# --- 0. ตั้งค่า MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)

# --- 1. โหลดโมเดล ---
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

# --- 2. สร้าง Queue สำหรับส่งค่าจากกล้องมาที่ UI ---
result_queue = queue.Queue()

# --- 3. UI ---
st.set_page_config(page_title="Signjai ASL", layout="wide")
st.title("👋 Signjai - ภาษามืออัจฉริยะ")

if 'text_output' not in st.session_state:
    st.session_state.text_output = ""

col1, col2 = st.columns([2, 1])

with col1:
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        char_detected = "-"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                try:
                    input_data = np.array([landmarks], dtype=np.float32)
                    prediction = model.predict(input_data)
                    char_detected = label_encoder.inverse_transform(prediction)[0]
                except:
                    char_detected = "Error"
        
        # ส่งค่าที่ตรวจจับได้เข้า Queue
        result_queue.put(char_detected)
        return frame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="signjai-final",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    # ส่วนดึงค่าจาก Queue มาแสดงผล
    current_char_placeholder = st.empty()
    
    # วนลูปเช็คค่าล่าสุดจากกล้อง
    actual_char = "-"
    try:
        # ดึงค่าล่าสุดจาก Queue (ไม่รอ)
        while not result_queue.empty():
            actual_char = result_queue.get_nowait()
    except:
        pass

    current_char_placeholder.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 25px; border-radius: 15px; border: 4px solid #31333f; text-align: center;">
            <p style="font-weight: bold; margin: 0;">ตรวจพบท่าทาง:</p>
            <h1 style="color: #1E88E5; font-size: 100px; margin: 10px 0;">{actual_char}</h1>
        </div>
    """, unsafe_allow_html=True)

    if st.button("➕ บันทึกอักษร", use_container_width=True):
        if actual_char not in ["-", "Error"]:
            st.session_state.text_output += str(actual_char)
            st.rerun()

    if st.button("🧹 ล้างข้อความ", use_container_width=True):
        st.session_state.text_output = ""
        st.rerun()

    st.subheader("📝 ประโยค:")
    st.info(st.session_state.text_output if st.session_state.text_output else "...")
