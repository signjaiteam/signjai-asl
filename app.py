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
mp_drawing = mp.solutions.drawing_utils
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
    
    # โหลดไฟล์ pkl
    data = joblib.load(MODEL_PATH)
    
    # ตรวจสอบโครงสร้างไฟล์ pkl (เผื่อเป็น dictionary หรือ list)
    if isinstance(data, dict):
        m = data.get('model') or data.get('classifier')
        le = data.get('label_encoder') or data.get('labels')
        return m, le
    elif isinstance(data, (list, tuple)):
        return data[0], data[1]
    return data, None # กรณีมีแต่โมเดลอย่างเดียว

model, label_encoder = load_model()

# --- 2. UI Layout ---
st.set_page_config(page_title="Signjai ASL", layout="wide")
st.title("👋 Signjai - ภาษามืออัจฉริยะ")

# ใช้ Session State เก็บค่าตัวอักษรและข้อความ
if 'current_char' not in st.session_state:
    st.session_state.current_char = "-"
if 'text_output' not in st.session_state:
    st.session_state.text_output = ""

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 กล้องตรวจจับ (มองเห็นจุดสีเขียว = ระบบทำงาน)")
    
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        # กลับด้านภาพเพื่อให้ผู้ใช้ขยับมือตามได้ง่าย (Mirror Effect)
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. วาดเส้นจุดบนมือ
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

                # 2. เตรียมข้อมูล (21 จุด * 3 ค่า x,y,z = 63 ค่า)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                # 3. ส่งให้โมเดลทำนาย
                try:
                    # จัดรูปทรงข้อมูลให้เป็น (1, 63)
                    input_data = np.array([landmarks], dtype=np.float32)
                    prediction = model.predict(input_data)
                    
                    # แปลงผลลัพธ์ (รองรับทั้งแบบมีและไม่มี Label Encoder)
                    if label_encoder:
                        char = label_encoder.inverse_transform(prediction)[0]
                    else:
                        char = prediction[0]
                    
                    st.session_state.current_char = str(char)
                except Exception as e:
                    # ถ้า Error ให้แสดงจำนวนจุดที่ส่งเข้าไป (ไว้เช็คว่าตรงกับที่ Train ไหม)
                    st.session_state.current_char = f"Error: {len(landmarks)} pts"
        else:
            st.session_state.current_char = "-"
        
        return frame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="signjai-main",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    # กรอบแสดงผลอักษรขนาดใหญ่
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 25px; border-radius: 15px; border: 4px solid #31333f; text-align: center; margin-bottom: 20px;">
            <p style="color: #666; font-weight: bold; font-size: 20px; margin: 0;">ตรวจพบท่าทาง:</p>
            <h1 style="color: #1E88E5; font-size: 100px; font-weight: 900; margin: 10px 0;">
                {st.session_state.current_char}
            </h1>
        </div>
    """, unsafe_allow_html=True)
    
    # ปุ่มควบคุม
    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ บันทึกอักษร", use_container_width=True):
            if st.session_state.current_char not in ["-", "Error"]:
                st.session_state.text_output += st.session_state.current_char
    with c2:
        if st.button("🧹 ล้างข้อความ", use_container_width=True):
            st.session_state.text_output = ""

    st.subheader("📝 ประโยคที่ได้:")
    st.info(st.session_state.text_output if st.session_state.text_output else "ยังไม่มีข้อมูลบันทึก")

    # ปุ่ม Refresh หน้าจอหากค่าไม่เปลี่ยน
    if st.button("🔄 รีเฟรชค่าการตรวจจับ"):
        st.session_state.current_char = "-"
