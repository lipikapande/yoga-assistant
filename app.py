import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import pickle
from tensorflow import keras
from collections import Counter
import time

hold_start_time = None
holding_pose = False
prev_pose_name = None

# Load MoveNet
module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = module.signatures['serving_default']

# Load trained classifier
classifier = keras.models.load_model('yoga_pose_classifier.h5')

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load ideal poses
ideal_poses = pd.read_csv('ideal_pose_keypoints_top40.csv', index_col=0)

# Pose smoothing
pose_history = []
history_size = 10

# Webcam
cap = cv2.VideoCapture(0)
print("🧘 Yoga Pose Detector (MoveNet) Started! Press 'q' to quit.")

def extract_keypoints_movenet(keypoints):
    """Extract and flatten keypoints from MoveNet output"""
    return keypoints[0, 0, :, :2].flatten()

# Joint names mapping for feedback (indices 5-16 only, skipping face keypoints 0-4)
# MoveNet keypoints: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
# 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow, 9=left_wrist, 10=right_wrist,
# 11=left_hip, 12=right_hip, 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
joint_index_to_name = {
    5: 'left shoulder',
    6: 'right shoulder',
    7: 'left elbow',
    8: 'right elbow',
    9: 'left wrist',
    10: 'right wrist',
    11: 'left hip',
    12: 'right hip',
    13: 'left knee',
    14: 'right knee',
    15: 'left ankle',
    16: 'right ankle'
}

def get_sequential_feedback(user_kp, ideal_kp, tolerance=0.03):
    """
    Check joints in priority order (Foundation First: Hips → Knees → Ankles → Shoulders → Elbows → Wrists)
    Returns first misalignment found with correction instruction
    Skips face keypoints (0-4)
    """
    # Foundation First priority order
    feedback_indices = [
        11, 12,  # Left hip, Right hip (FOUNDATION)
        13, 14,  # Left knee, Right knee
        15, 16,  # Left ankle, Right ankle
        5, 6,    # Left shoulder, Right shoulder (UPPER BODY)
        7, 8,    # Left elbow, Right elbow
        9, 10    # Left wrist, Right wrist
    ]
    
    for idx in feedback_indices:
        # Get user and ideal positions for this joint
        ux, uy = user_kp[idx*2], user_kp[idx*2+1]
        ix, iy = ideal_kp[idx*2], ideal_kp[idx*2+1]
        
        # Calculate difference
        diff_x, diff_y = ux - ix, uy - iy
        magnitude = np.sqrt(diff_x**2 + diff_y**2)
        
        # Check if misalignment exceeds tolerance
        if magnitude > tolerance:
            joint_name = joint_index_to_name[idx]
            
            # Determine primary direction of correction needed
            if abs(diff_x) > abs(diff_y):
                direction = "right" if diff_x > 0 else "left"
                return f"Move {joint_name} more {direction}", idx
            else:
                direction = "down" if diff_y > 0 else "up"
                return f"Move {joint_name} {direction}", idx
    
    return "Perfect form!", None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    input_img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 192, 192)
    input_img = tf.cast(input_img, dtype=tf.int32)

    outputs = movenet(input_img)
    keypoints = outputs['output_0'].numpy()
    flat_kp = extract_keypoints_movenet(keypoints)

    prediction = classifier.predict(flat_kp.reshape(1, -1), verbose=0)
    pose_idx = np.argmax(prediction)
    confidence = prediction[0][pose_idx] * 100

    # Pose smoothing - use most common pose from recent history
    pose_history.append(pose_idx)
    if len(pose_history) > history_size:
        pose_history.pop(0)

    if len(pose_history) >= 5:
        most_common = Counter(pose_history).most_common(1)[0][0]
        pose_name = label_encoder.inverse_transform([most_common])[0]
    else:
        pose_name = "Getting ready..."

    # Only process confident detections
    if confidence < 50:
        pose_name = "No pose detected"
        holding_pose = False
        hold_start_time = None
        prev_pose_name = None

    # Compute feedback and similarity if valid pose detected
    feedback_text, joint_idx = "No feedback", None
    similarity = 0
    
    if pose_name in ideal_poses.index and pose_name != "No pose detected" and pose_name != "Getting ready...":
        ideal = ideal_poses.loc[pose_name].values
        feedback_text, joint_idx = get_sequential_feedback(flat_kp, ideal, tolerance=0.03)
        difference = np.abs(flat_kp - ideal).mean()
        similarity = max(0, 100 - difference*100)

    # === TIMER AND FEEDBACK LOGIC ===
    # Zone 1: >88% = EXCELLENT 🟢 (Green, timer runs)
    # Zone 2: 85-88% = EXCELLENT 🟡 (Yellow, timer runs with warning)
    # Zone 3: 70-85% = GOOD 🟠 (Orange, no timer)
    # Zone 4: <70% = NEEDS WORK 🔴 (Red, no timer)
    
    if similarity >= 85:  # Timer zone (both green and yellow)
        # Start or continue timer
        if hold_start_time is None or pose_name != prev_pose_name:
            hold_start_time = time.time()
            prev_pose_name = pose_name
        
        holding_pose = True
        elapsed = time.time() - hold_start_time
        
        # Timer display
        if elapsed >= 30:
            timer_text = "✅ Great job! Pose held 30s!"
        else:
            timer_text = f"Hold for 30s - {int(elapsed)}s"
        
        # Color coding based on sub-zones
        if similarity >= 88:
            display_text = f"EXCELLENT! 🟢 {timer_text}"
            acc_color = (0, 255, 0)  # Green
        else:  # 85-88% buffer zone
            display_text = f"EXCELLENT! 🟡 {timer_text}"
            acc_color = (0, 255, 255)  # Yellow
    
    elif similarity >= 70:  # GOOD zone
        display_text = "GOOD - Keep improving!"
        acc_color = (0, 165, 255)  # Orange
        holding_pose = False
        hold_start_time = None
        prev_pose_name = None
    
    else:  # NEEDS WORK zone
        display_text = "NEEDS WORK"
        acc_color = (0, 0, 255)  # Red
        holding_pose = False
        hold_start_time = None
        prev_pose_name = None

    # Draw all keypoints on frame
    for i in range(17):
        x = int(keypoints[0,0,i,1] * frame.shape[1])
        y = int(keypoints[0,0,i,0] * frame.shape[0])
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    # Display pose name and confidence
    cv2.rectangle(frame, (5,5), (600,35), (0,0,0), -1)
    cv2.putText(frame, f"{pose_name.upper()} ({confidence:.1f}%)", (10,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Display accuracy/status
    cv2.rectangle(frame, (5,40), (600,70), (0,0,0), -1)
    cv2.putText(frame, display_text, (10,63), cv2.FONT_HERSHEY_SIMPLEX, 0.6, acc_color, 2)

    # Display corrections - ALWAYS show if there's a misalignment (even in yellow zone)
    if joint_idx is not None and feedback_text != "Perfect form!":
        # Highlight the problematic joint with red circle
        jx = int(keypoints[0,0,joint_idx,1] * frame.shape[1])
        jy = int(keypoints[0,0,joint_idx,0] * frame.shape[0])
        cv2.circle(frame, (jx, jy), 12, (0,0,255), 3)  # Red outline
        
        # Display correction text
        cv2.rectangle(frame, (5,75), (600,105), (0,0,0), -1)
        cv2.putText(frame, feedback_text, (10,98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow("Yoga Pose Detector (MoveNet)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()