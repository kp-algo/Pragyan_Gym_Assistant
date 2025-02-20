from flask import Flask, render_template, request, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

#calculate angle
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

#biceps_fxn
def biceps_ex():
        
    angles_down = [([37.25185403265382, 94.19870913857697], [171.5518890146694, 172.26146417164586])]
    angles_up = [([169.3186049602137, 177.41855852431516], [170.55922795894838, 174.84829433782323])]

    elbow_angle_final = sum(angles_down[0][0])/len(angles_down[0][0])
    elbow_angle_start = 160

    hips_angle_start = sum(angles_up[0][1])/len(angles_up[0][1])
    hips_angle_final = sum(angles_down[0][1])/len(angles_down[0][1])

    hips_angle_avg = (hips_angle_start + hips_angle_final)/2

    cap = cv2.VideoCapture(0)
    
    # bicep curl rep counter
    reps = 0
    position = "start"  # Start position

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit loop if video ends

            # Convert BGR to RGB for Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Get joint coordinates
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
                r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                # Calculating elbow angles
                left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

                #Calculating hip angles
                left_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
                right_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)            

                # Bicep_curl rep detection logic
                if (left_elbow_angle  + right_elbow_angle)/2 <= elbow_angle_final:  # Curling up elbow
                    position = "final"
                if (left_elbow_angle + right_elbow_angle)/2 >= elbow_angle_start and position == "final":  #Straightening elbow  
                    reps += 1
                    position = "start"
                    print(f"Reps: {reps}")
                
                if (left_hip_angle + right_hip_angle)/2 >= hips_angle_avg:
                    cv2.putText(image, "Your back is aligned perfectly.", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
                else:
                    cv2.putText(image, "Keep your back straight.", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)

                # Displays rep and angle on video
                cv2.putText(image, f"Reps: {reps}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                if position == "start":
                    cv2.putText(image, "Squeeze the biceps!", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                else:
                    cv2.putText(image, "Control the descent!", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                # Draw pose landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

           
            
            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(10) & 0xFF == ord('x'):  # Press 'x' to exit
                break
             

    cap.release()
    cv2.destroyAllWindows()

#overhead_press_fxn
def overhead_ex():
    elbow_angle_final = 160
    elbow_angle_start = 90

    hip_angle = 170

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils


    cap = cv2.VideoCapture(0)

    # Overhead rep counter
    reps = 0
    position = "start"  # Start position

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit loop if video ends

            # Convert BGR to RGB for Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Get joint coordinates
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
                r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                # Calculating elbow angles
                left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

                #Calculating hip angles
                left_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
                right_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)            

                # Push-up rep detection logic
                if (left_elbow_angle  + right_elbow_angle)/2 >= elbow_angle_final:  # Going down
                    position = "final"
                if (left_elbow_angle + right_elbow_angle)/2 <= elbow_angle_start and position == "final":  # Going up
                    reps += 1
                    position = "start"
                    print(f"Reps: {reps}")
                
                if (left_hip_angle + right_hip_angle)/2 >= hip_angle:
                    cv2.putText(image, "Your back is perfectly aligned", (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
                else:
                    cv2.putText(image, "Please keep your back straight", (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
                
                if position == "start":
                    cv2.putText(image, "Push to the top!", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
                else:
                    cv2.putText(image, "Lower the weights!", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
                # Displays rep and angle on video
                cv2.putText(image, f"Reps: {reps} ", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                # Draw pose landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(10) & 0xFF == ord('x'):  # Press 'x' to exit
                break

    cap.release()
    cv2.destroyAllWindows()

#pushup_fxn
def push_ups_ex():
    angles_up = [([172.6943547830095, 172.26249413226606], [141.76535926501808, 146.37529439708538])]
    angles_down = [([130.956350672782, 64.6995757962917], [175.32746563318298, 172.49762245663476])]

    elbow_angle_up = sum(angles_up[0][0])/len(angles_up[0][0])
    elbow_angle_down = sum(angles_down[0][0])/len(angles_down[0][0])

    hips_angle_avg = (sum(angles_up[0][1])/len(angles_up[0][1]) + sum(angles_down[0][1])/len(angles_down[0][1]))/2

    cap = cv2.VideoCapture(0)

    # Push-up rep counter
    reps = 0
    position = "up"  # Start position

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit loop if video ends

            # Convert BGR to RGB for Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Get joint coordinates
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            
                r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

                # Calculating elbow angles
                left_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                right_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

                #Calculating hip angles
                left_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
                right_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)            

                # Push-up rep detection logic
                if (left_elbow_angle  + right_elbow_angle)/2 <= elbow_angle_down:  # Going down
                    position = "down"
                if (left_elbow_angle + right_elbow_angle)/2 >= elbow_angle_up and position == "down":  # Going up
                    reps += 1
                    position = "up"
                    print(f"Reps: {reps}")
                
                if (left_hip_angle + right_hip_angle)/2 >= hips_angle_avg:
                    cv2.putText(image, "Your back is perfectly aligned", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
                else:
                    cv2.putText(image, "Your back has an arc", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)

                # Displays rep and angle on video
                cv2.putText(image, f"Reps: {reps}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                if position == "up":
                    cv2.putText(image, "Lower Down!", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
                else:
                    cv2.putText(image, "Explode Up!", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
                # Draw pose landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Show video
            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


            if cv2.waitKey(10) & 0xFF == ord('x'):  # Press 'x' to exit
                break

    cap.release()
    cv2.destroyAllWindows()
    
#squats_fxn
def squats_ex():
    angles_down = [([54.66668058091231, 59.595719445315574], [95.03274568894824, 91.08346462152144])]
    angles_up = [([2.885713170338925, 0.31385444050733596], [174.9452678641332, 179.4527614446527])]

    knee_angle_up = sum(angles_up[0][0])/len(angles_up[0][0])
    knee_angle_down = sum(angles_down[0][0])/len(angles_down[0][0])


    hip_angle_down = sum(angles_down[0][1])/len(angles_down[0][1])
    hip_angle_up = 170

    cap = cv2.VideoCapture(0)

    # Push-up rep counter
    reps = 0
    position = "start"  # Start position

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit loop if video ends

            # Convert BGR to RGB for Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Get joint coordinates
            
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                l_ankel = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                r_ankel = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                l_knee_angle = calculate_angle(l_hip, l_knee, l_shoulder)
                r_knee_angle = calculate_angle(r_hip, r_knee, r_shoulder)

                l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
                r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)

                # Push-up rep detection logic
                if (l_hip_angle + r_hip_angle)/2 <= hip_angle_down:  # Going down
                    position = "final"
                if (l_hip_angle + r_hip_angle)/2 >= hip_angle_up and position == "final":  # Going up
                    reps += 1
                    position = "start"
                    print(f"Reps: {reps}")
                
                if (l_knee_angle + r_knee_angle)/2 >= knee_angle_down:
                    cv2.putText(image, "Knees are protruding out", (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
                else:
                    cv2.putText(image, "Your form is perfect", (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
                if (position == "start"):
                    cv2.putText(image, "Squat down", (20,90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
                else:
                    cv2.putText(image, "Squat up", (20,90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

                # Displays rep and angle on video
                cv2.putText(image, f"Reps: {reps}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                # Draw pose landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
                

            # Show video
            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(10) & 0xFF == ord('x'):  # Press 'x' to exit
                break

    cap.release()
    

@app.route('/')
def home():
    return render_template("index.html") 

@app.route('/biceps')
def biceps():
    return Response(biceps_ex(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/overhead_press')
def overhead_press():
    return Response(overhead_ex(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/push_ups')
def push_ups():
    return Response(push_ups_ex(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/squats')
def squats():
    return Response(squats_ex(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
