import cv2
import Localization



def get_frames(query_path, video_base_path):
    query_array = []
    video = cv2.VideoCapture(query_path)
    got_frame = True

    while got_frame:
        got_frame, frame = video.read()
        query_array.append(frame)

    query_array_localized = Localization.find_screen(query_array)