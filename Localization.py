import numpy as np
import cv2


def find_screen(query_array, fps):
    query_array_localized = []
    N = len(query_array)
    prev_frame = None
    temp_diff = []

    for n in range(N - 1):
        frame = query_array[n]

        if prev_frame is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            temp_diff = np.zeros_like(gray)
        else:
            diff = cv2.subtract(frame, prev_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            gray_diff[gray_diff < 80] = 0
            temp_diff += gray_diff

        prev_frame = frame

    temp_diff = cv2.medianBlur(temp_diff, 5)
    # cv2.imshow('Gray', temp_diff)
    # cv2.waitKey()

    # checking all contours
    contours, hierarchy = cv2.findContours(temp_diff, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    crop = []
    screen_rectangle = ((0, 0), (0, 0), 0)
    max_area = 0
    for contour in contours:
        rectangle = cv2.minAreaRect(contour)
        if is_screen(temp_diff, rectangle):
            crop = crop_screen(frame, rectangle, False)
            area = len(crop) * len(crop[0])
            if area > max_area:
                max_area = area
                screen_rectangle = rectangle
                crop_max = crop

    box = cv2.boxPoints(screen_rectangle)
    box = np.int0(box)
    cv2.drawContours(frame, [box], 0, (200, 0, 0), 2)
    # cv2.imwrite('cont.jpg', frame)
    # cv2.imshow('crop.jpg', crop_max[:, :])
    # cv2.waitKey()
    # query_array_localized.append(crop_max[:, :])
    for n in range(N - 1):
        frame = query_array[n]
        crop = crop_screen(frame, screen_rectangle, False)

        query_array_localized.append(crop)
        # cv2.imshow('crop.jpg', crop[:, :])
        # cv2.waitKey()

    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (len(query_array_localized[0][0]), len(query_array_localized[0])))
    for f in query_array_localized:
        out.write(f)
    out.release()
    return query_array_localized


def temporal_diff(frame1, frame2, threshold=10):
    if frame1 is None or frame2 is None:
        return None
    diff = np.abs(frame1.astype('int16') - frame2.astype('int16'))
    # diff_t = diff > threshold
    ret, diff_t = cv2.threshold(diff, threshold, 255, 0)
    return diff_t


def is_screen(thresh, rectangle):
    crop = crop_screen(thresh, rectangle, False)
    height = len(crop)
    if (height == 0):
        return False
    return True


def crop_screen(image, rectangle, draw):
    rectangle = rotate_rectangle(rectangle)
    # rotate img
    angle = rectangle[2]
    rows, cols = image.shape[0], image.shape[1]
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rotated = cv2.warpAffine(image, matrix, (cols, rows))
    # rotate bounding box
    box = cv2.boxPoints(rectangle)
    pts = np.int0(cv2.transform(np.array([box]), matrix))[0]
    pts[pts < 0] = 0
    crop = img_rotated[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]
    box = cv2.boxPoints(rectangle)
    box = np.int0(box)
    if draw:
        cv2.drawContours(image, [box], 0, (200, 0, 0), 2)
    return crop


def rotate_rectangle(rectangle):
    if rectangle[1][0] < rectangle[1][1]:
        if rectangle[2] < -45:
            rectangle = (rectangle[0], (rectangle[1][1], rectangle[1][0]), rectangle[2] + 90)
        if rectangle[2] > 45:
            rectangle = (rectangle[0], (rectangle[1][1], rectangle[1][0]), rectangle[2] - 90)

    return rectangle
