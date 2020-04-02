import cv2
import numpy as np
import Localization
import Recognize


def get_frames(query_path, video_base_path):
    query_array = []
    video = cv2.VideoCapture(query_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    got_frame = True

    while got_frame:
        got_frame, frame = video.read()
        if frame is None:
            break
        width = len(frame[0])
        height = len(frame)
        if height == 0 or width == 0:
            break
        if width < height:
            frame = np.rot90(frame, 1)
        query_array.append(frame)

    query_array_localized = Localization.find_screen(query_array, fps)
    Recognize.recognize(query_array_localized, video_base_path, fps)