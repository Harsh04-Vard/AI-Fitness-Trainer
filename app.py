import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ---------------- MEDIAPIPE ----------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# ---------------- ANGLE FUNCTION ----------------
def calculate_angle(a, b, c):
    def get_xy(p):
        return np.array([p.x, p.y])

    a, b, c = get_xy(a), get_xy(b), get_xy(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# ---------------- PROCESSOR ----------------
class FitnessProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose()
        self.counter = 0
        self.stage = None
        self.mode = "Squat"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
            r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

            # -------- SQUAT --------
            if self.mode == "Squat":
                l_knee = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
                r_knee = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                l_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                r_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                left_knee = calculate_angle(l_hip, l_knee, l_ankle)
                right_knee = calculate_angle(r_hip, r_knee, r_ankle)

                avg_angle = (left_knee + right_knee) / 2

            # -------- PUSH-UP --------
            else:
                l_el = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                r_el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                l_wr = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
                r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                left_elbow = calculate_angle(l_sh, l_el, l_wr)
                right_elbow = calculate_angle(r_sh, r_el, r_wr)

                avg_angle = (left_elbow + right_elbow) / 2

            # -------- COUNTER LOGIC --------
            if avg_angle > 150:
                self.stage = "up"

            if avg_angle < 100 and self.stage == "up":
                self.stage = "down"
                self.counter += 1

            # Draw skeleton
            mp_draw.draw_landmarks(image, results.pose_landmarks,
                                   mp_pose.POSE_CONNECTIONS)

        return image

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}

/* Big Title */
.big-title {
    font-size:90px;
    font-weight:900;
    text-align:center;
    color:#00FFAA;
    margin-bottom:30px;
}

/* Counter Card */
.counter-card {
    background: rgba(255,255,255,0.05);
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    margin: auto;
    width: 300px;
}

.counter-card h3 {
    font-size:22px;
    color:#ccc;
}

.counter-card h1 {
    font-size:80px;
    font-weight:800;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<p class="big-title">🏋️ AI Fitness Trainer</p>', unsafe_allow_html=True)

# ---------------- MODE ----------------
mode = st.sidebar.selectbox("Select Exercise", ["Squat", "Push-up"])

# ---------------- COUNTER UI ----------------
counter_box = st.empty()

# ---------------- WEBRTC ----------------
ctx = webrtc_streamer(
    key="fitness",
    video_transformer_factory=FitnessProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)

# ---------------- LIVE UPDATE ----------------
if ctx.video_transformer:
    processor = ctx.video_transformer
    processor.mode = mode

    counter_box.markdown(f"""
    <div class="counter-card">
        <h3>Reps</h3>
        <h1>{processor.counter}</h1>
    </div>
    """, unsafe_allow_html=True)