import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
import joblib
import numpy as np
import os
import gdown  # ต้องเพิ่มใน requirements.txt ด้วย
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_autorefresh import st_autorefresh

# --- 1. จัดการเรื่องดาวน์โหลดโมเดลจาก Google Drive ---
MODEL_PATH = "asl_rf.pkl"
GOOGLE_DRIVE_ID = "1OdCW3HuSmrCpB2YdN-5pjagEtI7Pa1MH"

@st.cache_resource # ใช้ cache เพื่อให้ดาวน์โหลดแค่ครั้งเดียว
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("กำลังดาวน์โหลดโมเดลจาก Google Drive (ครั้งแรกเท่านั้น)..."):
            url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

# โหลดโมเดล
# --- แก้ไขบรรทัด 24-30 ใน app.py ---
try:
    model_data = load_model_from_drive()
    
    if isinstance(model_data, dict):
        # 1. งมหาตัว Model
        model = model_data.get('model') or model_data.get('classifier')
        
        # 2. งมหาตัว Label Encoder
        label_encoder = model_data.get('label_encoder') or model_data.get('labels')
        
        # 3. ถ้าหาชื่อไม่เจอจริงๆ ให้หยิบตามลำดับ (ตัวแรกคือโมเดล ตัวสองคือเลเบล)
        if model is None or label_encoder is None:
            keys = list(model_data.keys())
            model = model_data[keys[0]]
            label_encoder = model_data[keys[1]]
    else:
        # กรณีไฟล์ .pkl ไม่ใช่ dict (เป็นโมเดลเพียวๆ)
        st.error("ไฟล์โมเดลมีรูปแบบไม่รองรับ กรุณาตรวจสอบการ Save")
        st.stop()

except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    st.stop()

# --- 2. ตั้งค่า MediaPipe และตัวแปรกลาง ---

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    model_complexity=0 # ปรับเป็น 0 เพื่อความลื่น
)

# ใช้ Session State เก็บค่าที่ทายได้
if 'current_char' not in st.session_state:
    st.session_state.current_char = "-"

# ตั้งค่า Autorefresh เพื่อให้หน้าจออัปเดตเรียลไทม์
st_autorefresh(interval=800, key="datarefresh")

# --- 3. ส่วนการตั้งค่าหน้าตาเว็บ (UI) ---
st.set_page_config(page_title="Signjai ASL", layout="wide")
st.title("👋 Signjai - ภาษามืออัจฉริยะ")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 กล้องตรวจจับ")
    
    # ฟังก์ชัน Callback สำหรับดึงภาพจากกล้อง
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # ดึงข้อมูลจุดเชื่อมมือ 21 จุด
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                # ทำนายผล
                prediction = model.predict([landmarks])
                char = label_encoder.inverse_transform(prediction)[0]
                st.session_state.current_char = char
        else:
            st.session_state.current_char = "-"

        return frame

    webrtc_streamer(
        key="sign-lang",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.subheader("📝 ผลลัพธ์")
    
    # กรอบแสดงผลสีเทา-ดำ (Luxury Grey) ตามที่คุณต้องการ
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 25px; border-radius: 15px; border: 4px solid #31333f; text-align: center; box-shadow: 6px 6px 0px #31333f;">
            <p style="color: #666; font-size: 16px; font-weight: bold; letter-spacing: 2px; margin: 0;">CURRENT DETECTED</p>
            <h1 style="color: #31333f; font-size: 100px; font-weight: 900; margin: 10px 0;">
                {st.session_state.current_char}
            </h1>
        </div>
    """, unsafe_allow_html=True)

    # ส่วนบันทึกข้อความ (สามารถเพิ่มปุ่ม Save/Clear ได้ที่นี่)

    st.info("ระบบจะเปลี่ยนตัวอักษรตามท่าทางมือของคุณแบบเรียลไทม์")


