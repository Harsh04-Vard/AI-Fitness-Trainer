import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

# ---------------- MEDIAPIPE ----------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- ANGLE FUNCTION ----------------
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle

    return angle

# ---------------- FOLDERS ----------------
folders = [
    ("Correct sequence", 1),
    ("Wrong sequence", 0)
]

data = []

# ---------------- PROCESS VIDEOS ----------------
for folder, label in folders:
    for file in os.listdir(folder):

        if not file.endswith(".mp4"):
            continue

        path = os.path.join(folder, file)
        cap = cv2.VideoCapture(path)

        print("Processing:", path)

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 🔥 Resize for better detection
            frame = cv2.resize(frame, (640, 480))

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # DEBUG
            if results.pose_landmarks:
                print(f"Pose detected in frame {frame_count}")

                lm = results.pose_landmarks.landmark

                try:
                    # ELBOW ANGLES
                    left_elbow = calculate_angle(
                        lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                        lm[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                        lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    )

                    right_elbow = calculate_angle(
                        lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                        lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                        lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                    )

                    avg_elbow = (left_elbow + right_elbow) / 2

                    # BODY STRAIGHTNESS
                    hip_angle = calculate_angle(
                        lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                        lm[mp_pose.PoseLandmark.LEFT_HIP.value],
                        lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    )

                    # SAVE DATA
                    data.append([
                        left_elbow,
                        right_elbow,
                        avg_elbow,
                        hip_angle,
                        label
                    ])

                except:
                    continue

            # 🔥 Optional: Show video (press q to stop)
            cv2.imshow("Processing", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

cv2.destroyAllWindows()

# ---------------- SAVE CSV ----------------
df = pd.DataFrame(data, columns=[
    "left_elbow",
    "right_elbow",
    "avg_elbow",
    "hip_angle",
    "label"
])

df.to_csv("pushup_data.csv", index=False)

print("\n========================")
print("Dataset size:", df.shape)
print("CSV created successfully ✅")