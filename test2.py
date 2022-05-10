from imageai.Detection import ObjectDetection
import cv2
import os
import mediapipe as mp
import math as m
import numpy as np
import imutils
from imageai.Detection import ObjectDetection

# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree


"""
Function to send alert. Use this function to send alert when bad posture detected.
Feel free to get creative and customize as per your convenience.
"""


def sendWarning(x):
    pass


font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
execution_path = os.getcwd()

cap = cv2.VideoCapture('video1.mp4')

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolo.h5")
detector.loadModel()


while True:
    session, image = cap.read()

    _, detections = detector.detectObjectsFromImage(input_image=image, input_type="array", output_type='array', output_image_path='new.jpg')
    # print(detections)
    for obj in detections:
        cord = obj['box_points']
        cv2.rectangle(image, (cord[0], cord[1]), (cord[2], cord[3]), (0, 255, 0), 2)
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        good_frames = 0
        bad_frames = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Get height and width.
        h, w = image.shape[:2]

        # h, w = z, z*2
        # print(h, w)

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Cropp image
        # abc_plus = int(w / len(faces))
        b_image = np.zeros(image.shape, dtype=np.uint8)
        b_image[cord[1]:cord[2], cord[1]: cord[3]] = 255
        cropp = cv2.bitwise_and(image, b_image)
        # cropp = image[0:, abc:int(w / len(faces))]

        # cv2.imwrite(f"./data/cv_cut_thor{num_loc}.jpg", cropp)
        # num_loc += 1
        # abc += int(w / len(faces))

        # Process the image.
        keypoints = pose.process(cropp)
        # print(image)

        # Convert the image back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Use lm and lmPose as representative of the following methods.
        lm = keypoints.pose_landmarks
        # print(lm)
        # print(lm)
        if not lm:
            pass
        else:
            lmPose = mp_pose.PoseLandmark
            # print(lmPose)

            # Acquire the landmark coordinates.
            # Once aligned properly, left or right should not be a concern.
            # Left shoulder.
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            # print(lm.landmark[lmPose.LEFT_SHOULDER].x)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            # Right shoulder
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
            # Left ear.
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
            # Left hip.
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

            # l_shldr_x = w + h
            # l_shldr_y = (h + ((y + h) / 2))
            # r_shldr_x = (x - h)
            # r_shldr_y = (y - ((y + h) / 2))
            # l_ear_x = ((w + h) / 2)
            # l_ear_y = h
            # l_hip_x = (x - h * 3)
            # l_hip_y = ((h + y) + h / 2)
            # print(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y, l_ear_x, l_ear_y, l_hip_x, l_hip_y)

            # Calculate distance between left shoulder and right shoulder points.
            offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
            # print(offset)
            # print(offset)
            # print(offset)

            # Assist to align the camera to point at the side view of the person.
            # Offset threshold 30 is based on results obtained from analysis over 100 samples.
            # if offset < 100:
            #     cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
            # else:
            #     cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

            # Calculate angles.
            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

            # Draw landmarks.
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
            cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)

            # Let's take y - coordinate of P3 100px above x1,  for display elegance.
            # Although we are taking y = 0 while calculating angle between P1,P2,P3.
            cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
            cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

            # Similarly, here we are taking y - coordinate 100px above x1. Note that
            # you can take any value for y, not necessarily 100 or 200 pixels.
            cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

            # Put text, Posture and angle inclination.
            # Text string for display.
            angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

            # Determine whether good posture or bad posture.
            # The threshold angles have been set based on intuition.
            if neck_inclination < 40 and torso_inclination < 10:
                bad_frames = 0
                good_frames += 1

                # cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
                # cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
                # cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)

                # Join landmarks.
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)

            else:
                good_frames = 0
                bad_frames += 1

                # cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
                # cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
                # cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

                # Join landmarks.
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

            # Calculate the time of remaining in a particular posture.
            good_time = (1 / fps) * good_frames
            bad_time = (1 / fps) * bad_frames

            # Pose time.
            if good_time > 0:
                time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
                # cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
            else:
                time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
                # cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)

            # If you stay in bad posture for more than 3 minutes (180s) send an alert.
            if bad_time > 180:
                sendWarning()
            # Write frames.
        # video_output.write(image)
    cv2.imwrite('name.jpg', image)
    cv2.imshow("Text", image)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
