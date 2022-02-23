import os
import mediapipe as mp
import cv2 as cv
from icecream import ic
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def resize_images(imgs):
    DESIRE_WIDTH = 480
    DESIRE_HEIGHT = 480

    resized_imgs = []
    for img in imgs:

        h, w = img.shape[:2]
        if h < w:
            img = cv.resize(img, (DESIRE_WIDTH, math.floor(h / (w / DESIRE_WIDTH))))
        else:
            img = cv.resize(img, (math.floor(w / (h / DESIRE_HEIGHT)), DESIRE_HEIGHT))
        resized_imgs.append(img)
    return resized_imgs


def show_images(imgs):
    if isinstance(imgs, list):
        for img in imgs:
            cv.imshow('img', img)
            cv.waitKey(0)
    elif isinstance(imgs, np.ndarray):
        cv.imshow('img', imgs)
        cv.waitKey(0)
        # cv.destroyAllWindows()


def main():
    imgs = []
    for img_loc in os.listdir("imgs"):
        img_path = os.path.join("imgs", img_loc)
        img = cv.imread(img_path)
        imgs.append(img)

    imgs = resize_images(imgs)
    # show_images(imgs)
    for img in imgs:
        with mp_holistic.Holistic(static_image_mode=True, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5) as holistic:
            res = holistic.process(cv.cvtColor(img, cv.COLOR_BGR2RGB))

            # getting nose coords
            if res.pose_landmarks:
                print(f'Nose dims: {res.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x}, '
                      f'{res.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y}')

            # drawing landmarks
            annot_img = img.copy()
            mp_drawing.draw_landmarks(annot_img, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(annot_img, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(annot_img, res.face_landmarks,
                                      mp_holistic.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(annot_img,
                                      res.pose_landmarks,
                                      mp_holistic.POSE_CONNECTIONS,
                                      landmark_drawing_spec= mp_drawing_styles.get_default_pose_landmarks_style())
            mp_drawing.plot_landmarks(res.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)

            show_images(annot_img)


if __name__ == '__main__':
    main()
