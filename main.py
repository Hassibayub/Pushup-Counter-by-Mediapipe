import os
import mediapipe as mp
import cv2 as cv
from icecream import ic
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def resize_images(imgs):
    pass


def show_images(imgs):
    pass

def main():
    l = os.listdir("imgs")
    ic(l)

if __name__ == '__main__':
    main()