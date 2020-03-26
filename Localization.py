import numpy as np
import cv2

def find_screen(query_array):
    query_array_localized = []

    for n in range(10, 11):
        frame_temp = np.zeros_like(cv2.cvtColor(query_array[n], cv2.COLOR_BGR2GRAY))
        for m in range(n, n-11, -1):
            # cv2.imshow('n', cv2.cvtColor(query_array[n], cv2.COLOR_BGR2GRAY))
            # cv2.imshow('m', cv2.cvtColor(query_array[m], cv2.COLOR_BGR2GRAY))
            frame_temp += np.abs(cv2.cvtColor(query_array[n], cv2.COLOR_BGR2GRAY) - cv2.cvtColor(query_array[m], cv2.COLOR_BGR2GRAY))
            cv2.imshow('image', np.abs(cv2.cvtColor(query_array[n], cv2.COLOR_BGR2GRAY) - cv2.cvtColor(query_array[m], cv2.COLOR_BGR2GRAY)))
            cv2.waitKey()
        div = np.sum(frame_temp)/ (4*len(query_array)*len(query_array[0]))

        frame_temp = np.true_divide(frame_temp, (10))
        ret, frame_tempdiff = cv2.threshold(frame_temp, 20, 255, cv2.THRESH_BINARY)

