import streamlit as st
import cv2
import joblib
import numpy as np
import os
import gdown
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_autorefresh import st_autorefresh

# --- 0. พยายามโหลด MediaPipe แบบพิเศษ (แก้ปัญหา Python 3.13) ---
try:
    import mediapipe as mp
    from mediapipe.solutions import hands as mp_hands
    from mediapipe.solutions import drawing_utils as mp_drawing
except (ImportError, AttributeError):
    try:
        import mediapipe as mp
        from mediapipe.python.solutions import hands as mp_hands
        from mediapipe.python.solutions import drawing_utils as mp_drawing
    except Exception as e:
        st.error(f"MediaPipe Load Error: {e}")
        st.stop()

# --- 1. จัดการเรื่องดาวน์โหลดโมเดลจาก Google Drive ---
MODEL_PATH = "asl_rf.pkl"
GOOGLE_DRIVE_ID = "1OdCW3HuSmrCpB2YdN-5pjagEtI7Pa1MH"

@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("กำลังดาวน์โหลดโมเดลจาก Google Drive (ครั้งแรกเท่านั้น)..."):
            url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}'
            try:
                gdown.download(url, MODEL_PATH, quiet=False)
            except Exception as e:
                st.error(f"Download Error: {e}")
                st.stop()
    return joblib.load(MODEL_PATH)

# โหลดโมเดลและจัดการเรื่อง Key ใน Dictionary
try:
    model_data = load_model_from_drive()
    if isinstance(model_data, dict):
        model = model_data.get('model') or model_data.get('classifier')
        label_encoder = model_data.get('label_encoder') or model_data.get('labels')
        
        if model is None or label_encoder is None:
            keys = list(model_data.keys())
            model = model_data[keys[0]]
            label_encoder = model_data[keys[1]]
    else:
        model = model_data  # กรณีเซฟมาแบบโมเดลเพียวๆ
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    st.stop()

# --- 2. ตั้งค่า MediaPipe ---
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    model_complexity=0 
)

# ใช้ Session State เก็บค่า
if 'current_char' not in st.session_state:
    st.session_state.current_char = "-"

# อัปเดตหน้าจอ
st_autorefresh(interval=800, key="datarefresh")

# --- 3. UI ---
st.set_page_config(page_title="Signjai ASL", layout="wide")
st.title("👋 Signjai - ภาษามืออัจฉริยะ")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 กล้องตรวจจับ")
    
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                try:
                    prediction = model.predict([landmarks])
                    char = label_encoder.inverse_transform(prediction)[0]
                    st.session_state.current_char = char
                except:
                    st.session_state.current_char = "Error"
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
    st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 25px; border-radius: 15px; border: 4px solid #31333f; text-align: center; box-shadow: 6px 6px 0px #31333f;">
            <p style="color: #666; font-size: 16px; font-weight: bold; letter-spacing: 2px; margin: 0;">CURRENT DETECTED</p>
            <h1 style="color: #31333f; font-size: 100px; font-weight: 900; margin: 10px 0;">
                {st.session_state.current_char}
            </h1>
        </div>
    """, unsafe_allow_html=True)
    st.info("ระบบจะเปลี่ยนตัวอักษรตามท่าทางมือของคุณแบบเรียลไทม์")
