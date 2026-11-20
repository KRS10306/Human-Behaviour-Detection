import cv2
import joblib
import time
from collections import deque
from pose_extractor_yolo26 import YOLO26PoseExtractor, select_main_person
from feature_engine import normalize_skeleton, make_windows, extract_tsfel_features
from escalation import trigger_bandwidth_call

# --- Configuration ---
WINDOW = 30
STRIDE = 15
FPS = 30

FALL_CONFIRM_SEC = 2     # Time to confirm the initial fall
EMERGENCY_SEC = 300      # 5 minutes for incapacitation

# --- Repeated Fall Configuration ---
REPEAT_WINDOW_SEC = 300  # 5 minutes
REPEAT_THRESHOLD = 10    # 10 falls

BEHAVIOUR_NAMES = {
    0: "Lying Down", 1: "Sit down", 2: "Standing", 3: "Stand up",
    4: "Walking", 5: "Sitting", 6: "Fall Down",
    7: "Blowing Nose", 8: "Cleaning", 9: "Eating",
    10: "Watching TV", 11: "Leaning"
}

extractor = YOLO26PoseExtractor("yolo26n-pose.pt")
model = joblib.load("behavior_model_yolo26.pkl")

buffer = deque(maxlen=WINDOW)
frame_index = 0
current_label = "Waiting..."

# --- State Machine Variables ---
is_armed = False           
fall_timer = 0
emergency_timer = 0
interruption_counter = 0 
fall_warning_issued = False

# --- Tracking Variables ---
fall_history = deque() 
total_emergencies_triggered = 0 

# Use video file or 0 for webcam
cap = cv2.VideoCapture("fall_detection_9.mp4") 

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    persons = extractor.extract(frame)
    person = select_main_person(persons)
    
    # --- FIX: CHECK FOR PERSON PRESENCE ---
    if person is not None:
        norm = normalize_skeleton(person)
        if norm is not None:
            buffer.append(norm)
    else:
        # NO PERSON DETECTED IN FRAME
        current_label = "No Person"
        buffer.clear()          # Clear the stale data so it doesn't hallucinate!
        is_armed = False        # Disarm emergency timers
        fall_timer = 0
        emergency_timer = 0
        interruption_counter = 0
        fall_warning_issued = False

    frame_index += 1

    # Only run the model if we actually have 30 frames of a person
    if len(buffer) == WINDOW and frame_index % STRIDE == 0:
        window_df = make_windows(list(buffer), window=WINDOW, stride=WINDOW)[0]
        feat = extract_tsfel_features(window_df)
        
        # --- FIX: CONFIDENCE THRESHOLD FOR "NO ACTION" ---
        probabilities = model.predict_proba(feat)[0]
        max_confidence = max(probabilities)
        label = int(model.classes_[probabilities.argmax()])

        if max_confidence < 0.60:
            current_label = "No Action (Low Confidence)"
            label = -1 # Prevent triggering state machine for low confidence
        else:
            current_label = BEHAVIOUR_NAMES.get(label, "Unknown")

        # ==========================================
        # --- STRICT SEQUENCE STATE MACHINE ---
        # ==========================================
        
        # 1. ARM THE SYSTEM: Must detect a Fall Down first
        if label == 6 and not is_armed:
            print("\n>>> FALL DETECTED. Arming emergency sequence...")
            is_armed = True
            fall_timer = 1
            emergency_timer = 1
            interruption_counter = 0
            fall_warning_issued = False
            
            # --- REPEATED FALL LOGIC & LOGGING ---
            current_time = time.time()
            fall_history.append(current_time)
            print(f"[TERMINAL UPDATE] Fall added. Recent Fall Count: {len(fall_history)}/{REPEAT_THRESHOLD}")
            
            # Remove falls older than 5 minutes and log the reset
            removed_count = 0
            while fall_history and current_time - fall_history[0] > REPEAT_WINDOW_SEC:
                fall_history.popleft()
                removed_count += 1
            
            if removed_count > 0:
                print(f"[TERMINAL UPDATE] {removed_count} old fall(s) expired from the {REPEAT_WINDOW_SEC/60}-minute window.")
                print(f"[TERMINAL UPDATE] Recent Fall Count reset to: {len(fall_history)}/{REPEAT_THRESHOLD}")

            # Check if there are 10 falls in the last 5 minutes
            if len(fall_history) >= REPEAT_THRESHOLD:
                print("\n" + "*"*50)
                print(f">>> EXTREME EMERGENCY: {REPEAT_THRESHOLD} falls detected in {REPEAT_WINDOW_SEC/60} minutes!")
                print(">>> POSSIBLE SEIZURE OR VIOLENT CRISIS. CALLING 911 IMMEDIATELY...")
                print("*"*50 + "\n")
                
                total_emergencies_triggered += 1
                print(f"[STATUS] Total Emergencies Triggered: {total_emergencies_triggered}")
                fall_history.clear() 
                print(f"[TERMINAL UPDATE] Fall history cleared. Recent Fall Count: 0/{REPEAT_THRESHOLD}")
                trigger_bandwidth_call("Extreme emergency: Multiple repeated falls detected. Possible seizure or violent crisis.")
            
        # 2. SUSTAIN THE SEQUENCE: If armed and person is Fallen OR Lying
        elif is_armed and (label == 6 or label == 0):
            interruption_counter = 0 
            fall_timer += 1
            emergency_timer += 1
            
            current_fall_sec = fall_timer * (STRIDE / FPS)
            current_emerg_sec = emergency_timer * (STRIDE / FPS)

            # Phase 1: Warning (2 seconds)
            if current_fall_sec >= FALL_CONFIRM_SEC and not fall_warning_issued:
                print(">>> WARNING: Person is down. Please confirm if you are okay.")
                fall_warning_issued = True

            # Phase 2: Emergency (5 minutes)
            if current_emerg_sec >= EMERGENCY_SEC:
                print("\n" + "="*50)
                print(">>> CRITICAL EMERGENCY: Prolonged incapacitation (5 mins)!")
                print(">>> CALLING AMBULANCE / CAREGIVER NOW...")
                print("="*50 + "\n")
                
                total_emergencies_triggered += 1
                print(f"[STATUS] Total Emergencies Triggered: {total_emergencies_triggered}")
                
                is_armed = False
                fall_timer = 0
                emergency_timer = 0
                fall_warning_issued = False
                trigger_bandwidth_call("Critical alert: Prolonged incapacitation after a fall. Immediate assistance required.")

        # 3. DISARM THE SYSTEM: If armed, but person does something else
        elif is_armed and (label != 6 and label != 0):
            interruption_counter += 1
            if interruption_counter > 4: 
                print(f">>> SEQUENCE BROKEN: Person is '{current_label}'. System disarmed.")
                print(f"[TERMINAL UPDATE] Incapacitation timer reset.")
                is_armed = False
                fall_timer = 0
                emergency_timer = 0
                fall_warning_issued = False
                interruption_counter = 0

    # --- UI Overlay ---
    cv2.putText(frame, f"Action: {current_label}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
    if is_armed:
        current_emerg_sec = emergency_timer * (STRIDE/FPS)
        timer_color = (0, 0, 255) if fall_warning_issued else (0, 255, 255)
        text = f"EMERGENCY: {current_emerg_sec:.1f}s" if fall_warning_issued else f"Confirming Fall: {current_emerg_sec:.1f}s"
        cv2.putText(frame, text, (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, timer_color, 3)
        
    if len(fall_history) > 0:
        cv2.putText(frame, f"Recent Falls: {len(fall_history)}/{REPEAT_THRESHOLD}", (20, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
    if total_emergencies_triggered > 0:
        cv2.putText(frame, f"Total Alerts Fired: {total_emergencies_triggered}", (20, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Human Behaviour Detection - Crisis Logic", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# 0 for webcam, or RTSP URL
# Instead of: 
# cap = cv2.VideoCapture(0)