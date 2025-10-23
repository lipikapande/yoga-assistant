import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
from tensorflow import keras

# Load trained model
model = keras.models.load_model('yoga_pose_classifier.h5')

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load ideal poses
ideal_poses = pd.read_csv('ideal_pose_keypoints_top40.csv', index_col=0)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_keypoints(results):
    """Extract keypoints from MediaPipe results"""
    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark[:17]:  # First 17 landmarks
            keypoints.extend([landmark.x, landmark.y])
        return np.array(keypoints)
    return None

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    a = np.array([p1[0], p1[1]])
    b = np.array([p2[0], p2[1]])
    c = np.array([p3[0], p3[1]])
    
    ba = a - b
    bc = c - b
    
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

def get_detailed_feedback(user_keypoints, ideal_keypoints, pose_name):
    """Generate specific joint feedback"""
    feedback_list = []
    
    # Joint names for first 17 landmarks
    joint_names = [
        'nose', 'left eye', 'right eye', 'left ear', 'right ear',
        'left shoulder', 'right shoulder', 'left elbow', 'right elbow',
        'left wrist', 'right wrist', 'left hip', 'right hip',
        'left knee', 'right knee', 'left ankle', 'right ankle'
    ]
    
    # Calculate differences for each joint
    differences = []
    for i in range(17):
        x_diff = user_keypoints[i*2] - ideal_keypoints[i*2]
        y_diff = user_keypoints[i*2+1] - ideal_keypoints[i*2+1]
        
        # Calculate magnitude of difference
        magnitude = np.sqrt(x_diff**2 + y_diff**2)
        differences.append((i, magnitude, x_diff, y_diff))
    
    # Sort by magnitude (biggest problems first)
    differences.sort(key=lambda x: x[1], reverse=True)
    
    # Generate feedback for top differences
    threshold = 0.08  # Adjust sensitivity
    
    for i, magnitude, x_diff, y_diff in differences[:5]:  # Top 5 differences
        if magnitude > threshold:
            # Determine primary direction
            if abs(x_diff) > abs(y_diff):
                direction = "right" if x_diff > 0 else "left"
                feedback_list.append(f"Move {joint_names[i]} more {direction}")
            else:
                direction = "down" if y_diff > 0 else "up"
                feedback_list.append(f"Move {joint_names[i]} {direction}")
    
    # Return top 3 most important corrections
    return feedback_list[:3]

# Start webcam
cap = cv2.VideoCapture(0)

print("🧘 Yoga Pose Detector Started!")
print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame
    results = pose.process(rgb_frame)
    
    # Draw skeleton
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
        )
        
        # Extract keypoints
        keypoints = extract_keypoints(results)
        
        if keypoints is not None:
            # Predict pose
            prediction = model.predict(keypoints.reshape(1, -1), verbose=0)
            pose_idx = np.argmax(prediction)
            pose_name = label_encoder.inverse_transform([pose_idx])[0]
            confidence = prediction[0][pose_idx] * 100
            
            # Display pose name and confidence
            cv2.rectangle(frame, (5, 5), (400, 35), (0, 0, 0), -1)
            cv2.putText(frame, f"{pose_name.upper()}", (10, 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Get ideal pose keypoints and feedback
            if pose_name in ideal_poses.index:
                ideal = ideal_poses.loc[pose_name].values
                
                # Get detailed feedback
                detailed_feedback = get_detailed_feedback(keypoints, ideal, pose_name)
                
                # Calculate overall similarity
                difference = np.abs(keypoints - ideal).mean()
                similarity = max(0, 100 - difference * 100)
                
                # Display accuracy with color coding
                if similarity > 85:
                    acc_color = (0, 255, 0)  # Green
                    status = "EXCELLENT!"
                elif similarity > 70:
                    acc_color = (0, 255, 255)  # Yellow
                    status = "GOOD"
                else:
                    acc_color = (0, 165, 255)  # Orange
                    status = "NEEDS WORK"
                
                cv2.rectangle(frame, (5, 40), (300, 70), (0, 0, 0), -1)
                cv2.putText(frame, f"Accuracy: {similarity:.1f}% - {status}", 
                           (10, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.6, acc_color, 2)
                
                # Display detailed feedback
                y_pos = 95
                if detailed_feedback:
                    cv2.rectangle(frame, (5, 75), (450, 75 + len(detailed_feedback)*30 + 10), (0, 0, 0), -1)
                    cv2.putText(frame, "CORRECTIONS:", (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_pos += 30
                    
                    for fb in detailed_feedback:
                        cv2.putText(frame, f"- {fb}", (15, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        y_pos += 25
                else:
                    cv2.rectangle(frame, (5, 75), (250, 105), (0, 0, 0), -1)
                    cv2.putText(frame, "Perfect form!", (10, 95),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow('Yoga Pose Detector', frame)
    
    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()