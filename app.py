import streamlit as st
import threading
import speech_recognition as sr
import time
from detection.detection import DrowsinessDetector

detector = DrowsinessDetector(r"C:\Users\DELL\Desktop\Project_in_Computer_Vision\EyeAware\music\music.wav")

# Global flag for stopping listening thread
stop_listening_flag = False

def listen_for_awake_phrase():
    global stop_listening_flag
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
    while not stop_listening_flag:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=8, phrase_time_limit=5)
            text = recognizer.recognize_google(audio)
            print(f"[DEBUG] Recognized speech: {text}")
            print(f"[SPEECH DEBUG] You said: {text}")
            if "i'm awake" in text.lower() or "i am awake" in text.lower() or "am awake" in text.lower() or "am" in text.lower() or "awake" in text.lower():
                print("[DEBUG] 'I'm awake' phrase detected!")
                detector.stop_alarm_by_voice()
                break
        except sr.UnknownValueError:
            print("[DEBUG] Could not understand audio")
        except sr.WaitTimeoutError:
            print("[DEBUG] Listening timed out, no speech detected")
        except sr.RequestError as e:
            st.error(f"Speech recognition error: {e}")
            print(f"[DEBUG] Request error: {e}")
            break

st.title("EYEAWARE : Real-Time Eye Fatigue Detection")
start = st.button("Start Detection")
stop = st.button("Stop Detection")
FRAME_WINDOW = st.image([])

# New placeholders for MAR and yawn count
mar_placeholder = st.empty()
yawn_placeholder = st.empty()

if "run" not in st.session_state:
    st.session_state.run = False

if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False
    stop_listening_flag = True
    detector.stop_camera()

listener_thread = None

if st.session_state.run:
    detector.start_camera()
    stop_listening_flag = False

    while st.session_state.run:
        frame_rgb, ear_avg, status, alarm_on = detector.process_frame()

        if frame_rgb is None:
            st.warning("Camera error or not found.")
            break

        # Start listening thread only if alarm is on and thread is not alive
        if alarm_on and (listener_thread is None or not listener_thread.is_alive()):
            stop_listening_flag = False
            listener_thread = threading.Thread(target=listen_for_awake_phrase, daemon=True)
            listener_thread.start()
            st.warning("Say 'I'm awake' to stop the alarm!")

        # Notify user alarm stopped, but do NOT stop detection loop or camera
        if not alarm_on and detector.alarm_stopped:
            st.success("Alarm stopped, you said you're awake!")
            # No break, no stopping camera

        FRAME_WINDOW.image(frame_rgb)

        # Display MAR and yawn count
        mar_placeholder.markdown(f"**MAR (Mouth Aspect Ratio):** {detector.mouth_aspect_ratio if hasattr(detector, 'mouth_aspect_ratio') else 'N/A'}")
        yawn_placeholder.markdown(f"**Yawns Detected:** {detector.total_yawns}")

        time.sleep(0.03)

    detector.stop_camera()
    FRAME_WINDOW.empty()
    mar_placeholder.empty()
    yawn_placeholder.empty()
