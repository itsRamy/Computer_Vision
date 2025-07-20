import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance
from pygame import mixer
import time
from collections import deque

class DrowsinessDetector:
    EAR_THRESHOLD_CLOSE = 0.22
    EAR_THRESHOLD_OPEN = 0.27
    EYES_CLOSED_SECONDS = 3

    MAR_THRESHOLD_YAWN = 0.35  # lowered for better detection
    YAWN_RESET_SECONDS = 300  # 5 minutes in seconds
    YAWN_ALARM_THRESHOLD = 3  # Trigger alarm if >=3 yawns in last 5 mins

    def __init__(self, alarm_sound_path):
        mixer.init()
        mixer.music.load(alarm_sound_path)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            r"C:\Users\DELL\Desktop\Project_in_Computer_Vision\EyeAware\models\shape_predictor.dat"
        )

        # Eye landmarks
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # Mouth landmarks (outer mouth: 48–68)
        self.mStart, self.mEnd = 48, 68

        self.cap = None
        self.eyes_closed_start = None
        self.alarm_on = False
        self.alarm_stopped = False  # prevent restarting alarm after voice stop
        self.ear_history = deque(maxlen=5)

        # Yawning state
        self.yawn_times = deque()
        self.total_yawns = 0
        self.mouth_open = False  # Track mouth open/close state for yawning

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def mouth_aspect_ratio(self, mouth):
        A = distance.euclidean(mouth[13], mouth[19])  # vertical distances
        B = distance.euclidean(mouth[14], mouth[18])
        C = distance.euclidean(mouth[15], mouth[17])
        vertical = (A + B + C) / 3.0
        horizontal = distance.euclidean(mouth[12], mouth[16])  # horizontal
        return vertical / horizontal

    def start_camera(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)

    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.stop_alarm()
        self.alarm_stopped = False  # Reset on stop

    def stop_alarm(self):
        if self.alarm_on:
            mixer.music.stop()
            self.alarm_on = False

    def stop_alarm_by_voice(self):
        """
        Stop alarm triggered by voice and suspend restarting until eyes open again.
        Also reset yawn counter.
        """
        print("[INFO] Alarm stopped by voice. Resetting yawn counter.")
        self.stop_alarm()
        self.alarm_stopped = True
        self.total_yawns = 0  # ✅ Reset yawn counter
        self.yawn_times.clear()  # ✅ Clear yawn timestamps

    def reset_alarm_stop_flag(self):
        """Call this when eyes open again to allow alarm restarting"""
        self.alarm_stopped = False

    def process_frame(self):
        if self.cap is None:
            raise RuntimeError("Camera not started")

        ret, frame = self.cap.read()
        if not ret:
            return None, None, "Camera error", False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)

        ear_avg = None
        mar = None
        status = "No face detected"

        if len(faces) > 0:
            face = faces[0]
            shape = self.predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # Eyes
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            self.ear_history.append(ear)
            ear_avg = sum(self.ear_history) / len(self.ear_history)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Mouth
            mouth = shape[self.mStart:self.mEnd]
            mar = self.mouth_aspect_ratio(mouth)
            print(f"[DEBUG] MAR: {mar:.2f}")  # Debug: print MAR
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 1)

            # Yawning logic (count only when mouth closes after being open)
            if mar is not None:
                if mar > self.MAR_THRESHOLD_YAWN:
                    if not self.mouth_open:
                        print("[DEBUG] Mouth opened (yawn start)")
                        self.mouth_open = True
                else:
                    if self.mouth_open:
                        print("[DEBUG] Mouth closed (yawn end) - counting yawn")
                        self.mouth_open = False
                        self.total_yawns += 1
                        self.yawn_times.append(time.time())

            # Remove old yawns outside the time window
            current_time = time.time()
            self.yawn_times = deque(
                [t for t in self.yawn_times if current_time - t <= self.YAWN_RESET_SECONDS]
            )

            # Eye alarm logic
            if ear_avg < self.EAR_THRESHOLD_CLOSE:
                if self.alarm_stopped:
                    pass  # alarm stopped manually
                else:
                    if self.eyes_closed_start is None:
                        self.eyes_closed_start = current_time
                    else:
                        elapsed = current_time - self.eyes_closed_start
                        if elapsed >= self.EYES_CLOSED_SECONDS and not self.alarm_on:
                            print("[ALARM] Eyes closed too long! Triggering alarm.")
                            mixer.music.play(-1)
                            self.alarm_on = True
            else:
                self.eyes_closed_start = None
                self.reset_alarm_stop_flag()

            # Yawn alarm logic
            if len(self.yawn_times) >= self.YAWN_ALARM_THRESHOLD and not self.alarm_on:
                if not self.alarm_stopped:
                    print("[ALARM] Too many yawns! Triggering alarm.")
                    mixer.music.play(-1)
                    self.alarm_on = True

            status = "Closed" if ear_avg < self.EAR_THRESHOLD_CLOSE else "Open"

        else:
            # No face detected, reset states
            self.eyes_closed_start = None
            self.ear_history.clear()

        # Draw EAR, MAR and yawns text on frame
        color = (0, 255, 0) if status == "Open" else (0, 0, 255)
        ear_text = f"EAR: {ear_avg:.2f}" if ear_avg is not None else "EAR: N/A"
        mar_text = f"MAR: {mar:.2f}" if mar is not None else "MAR: N/A"
        yawns_text = f"Yawns: {self.total_yawns}"

        cv2.putText(frame, ear_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, mar_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, yawns_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame_rgb, ear_avg, status, self.alarm_on
