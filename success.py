import cv2 as cv
import mediapipe as mp
import os
import numpy as np
import webbrowser as wb
import pyautogui as pg


capture = cv.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
flex = "#I am geniusssssssssssssssssssssssssssssssssssssssssssssssssssss"

Tpose = False
left_dab = False
right_dab = False
jesus_pose = False
right_leg_streched = False
left_leg_streched = False
neutral_pose = False
left_hand_oath = False
right_hand_oath = False
bow_down = False


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while capture.isOpened:
        ret, frame = capture.read()

        frame = cv.resize(frame, (900, 700))

        # Detect stuff and render stuff

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detections
        results = pose.process(image)

        try:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            ]

            left_elbow = [
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            ]

            left_wrist = [
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            ]

            right_shoulder = [
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
            ]

            right_elbow = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
            ]

            right_wrist = [
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
            ]

            right_hip = [
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
            ]

            left_hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
            ]

            left_knee = [
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
            ]

            right_knee = [
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
            ]

            left_ankle = [
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
            ]

            right_ankle = [
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
            ]

            left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)

            right_shoulder_angle = calculate_angle(
                right_hip, right_shoulder, right_elbow
            )

            left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            right_elbow_angle = calculate_angle(
                right_shoulder, right_elbow, right_wrist
            )

            left_knee_angle = calculate_angle(left_ankle, left_knee, left_hip)

            right_knee_angle = calculate_angle(right_ankle, right_knee, right_hip)

            right_hip_angle = calculate_angle(left_hip, right_hip, right_knee)

            left_hip_angle = calculate_angle(right_hip, left_hip, left_knee)

            left_collar_angle = calculate_angle(
                left_elbow, left_shoulder, right_shoulder
            )
            right_collar_angle = calculate_angle(
                right_elbow, right_shoulder, left_shoulder
            )

            left_bow_angle = calculate_angle(left_knee, left_hip, left_shoulder)
            right_bow_angle = calculate_angle(right_knee, right_hip, right_shoulder)

            # cv.putText(
            #     image,
            #     str(right_bow_angle),
            #     tuple(np.multiply(right_shoulder, [640, 480]).astype(int)),
            #     cv.FONT_HERSHEY_SIMPLEX,
            #     1,
            #     (0, 0, 0),
            #     2,
            #     cv.LINE_AA,
            # )

            # cv.putText(
            #     image,
            #     str(right_shoulder_angle),
            #     tuple(np.multiply(right_shoulder, [640, 480]).astype(int)),
            #     cv.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 0, 0),
            #     2,
            #     cv.LINE_AA,
            # )
            if neutral_pose == True:
                Tpose = False
                left_leg_raised = False
                right_leg_raised = False
                jesus_pose = False
                right_leg_streched = False
                left_leg_streched = False
                left_hand_oath = False
                left_dab = False
                right_dab = False
                right_hand_oath = False
                bow_down = False
                neutral_pose = False

            if (
                int(left_shoulder_angle) > 80
                and int(left_shoulder_angle) < 110
                and int(right_shoulder_angle) > 80
                and int(right_shoulder_angle) < 110
                and int(right_elbow_angle) < 181
                and int(right_elbow_angle) > 165
                and int(left_elbow_angle) < 181
                and int(left_elbow_angle) > 165
                and int(right_knee_angle) > 165
                and int(right_knee_angle) < 181
                and int(right_hip_angle) > 80
                and int(right_hip_angle) < 120
                and int(left_hip_angle) > 80
                and int(left_hip_angle) < 120
            ):
                if Tpose == False:
                    wb.open("https://youtu.be/dQw4w9WgXcQ")
                    Tpose = True

                else:
                    pass
            if (
                int(left_shoulder_angle) > 60
                and int(left_shoulder_angle) < 130
                and int(right_shoulder_angle) > 60
                and int(right_shoulder_angle) < 130
                and int(right_elbow_angle) < 120
                and int(right_elbow_angle) > 85
                and int(left_elbow_angle) < 120
                and int(left_elbow_angle) > 85
                and int(right_knee_angle) > 165
                and int(right_knee_angle) < 181
                and int(right_hip_angle) > 80
                and int(right_hip_angle) < 120
                and int(left_hip_angle) > 80
                and int(left_hip_angle) < 120
            ):
                if jesus_pose == False:
                    pg.press("space")
                    jesus_pose = True
                if jesus_pose == True:
                    pass

            if (
                int(left_shoulder_angle) > 80
                and int(left_shoulder_angle) < 160
                and int(right_shoulder_angle) > 0
                and int(right_shoulder_angle) < 100
                and int(right_elbow_angle) < 90
                and int(right_elbow_angle) > 1
                and int(left_elbow_angle) < 181
                and int(left_elbow_angle) > 100
                and int(right_knee_angle) > 165
                and int(right_knee_angle) < 181
                and int(right_hip_angle) > 80
                and int(right_hip_angle) < 120
                and int(left_hip_angle) > 80
                and int(left_hip_angle) < 120
                and int(left_collar_angle) > 100
                and int(left_collar_angle) < 181
                and int(right_collar_angle) > 0
                and int(right_collar_angle) < 100
            ):
                if left_dab == False:
                    pg.keyDown("alt")
                    pg.press("tab")
                    pg.press("right")
                    pg.keyUp("alt")

                    left_dab = True
                if left_dab == True:
                    pass

            if (
                int(left_shoulder_angle) > 0
                and int(left_shoulder_angle) < 100
                and int(right_shoulder_angle) > 80
                and int(right_shoulder_angle) < 160
                and int(right_elbow_angle) < 181
                and int(right_elbow_angle) > 100
                and int(left_elbow_angle) < 90
                and int(left_elbow_angle) > 1
                and int(right_knee_angle) > 165
                and int(right_knee_angle) < 181
                and int(right_hip_angle) > 80
                and int(right_hip_angle) < 120
                and int(left_hip_angle) > 80
                and int(left_hip_angle) < 120
                and int(left_collar_angle) > 0
                and int(left_collar_angle) < 100
                and int(right_collar_angle) > 100
                and int(right_collar_angle) < 181
            ):
                if right_dab == False:
                    pg.typewrite(flex)
                    right_dab = True
                if right_dab == True:
                    pass

            if int(right_bow_angle) < 120 or int(left_bow_angle) < 120:
                if bow_down == False:
                    pg.press("d")
                    bow_down = True
                if bow_down == True:
                    pass

            if (
                int(left_shoulder_angle) > -1
                and int(left_shoulder_angle) < 20
                and int(right_shoulder_angle) > -1
                and int(right_shoulder_angle) < 20
                and int(left_elbow_angle) > 165
                and int(left_elbow_angle) < 181
                and int(right_elbow_angle) > 165
                and int(right_elbow_angle) < 181
                and int(right_knee_angle) > 165
                and int(right_knee_angle) < 181
                and int(left_knee_angle) > 165
                and int(left_knee_angle) < 181
                and int(right_hip_angle) > 0
                and int(right_hip_angle) < 98
                and int(left_hip_angle) > 0
                and int(left_hip_angle) < 98
                and int(right_knee_angle) > 165
            ):
                neutral_pose = True
        except:
            pass

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # Rendering stuff
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(189, 123, 25), thickness=4, circle_radius=2),
            mp_drawing.DrawingSpec(color=(138, 94, 18), thickness=3, circle_radius=2),
        )

        cv.imshow("test", image)

        if cv.waitKey(20 & 0xFF) == ord("d"):
            break

capture.release()
cv.destroyAllWindows()
