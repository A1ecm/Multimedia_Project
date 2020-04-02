import sys

import numpy as np
import cv2
import glob
import sqlite3 as sqlite
import pickle
from os.path import basename


def recognize(query, vids_path, fps):
    ch_array = []
    chd_array = []
    td_array = []
    bd_array = []
    prev_ch = None
    prev_frame = None
    bd = None

    for frame in query:
        # cv2.imshow("frame", frame)
        # cv2.waitKey()
        # td = temporal_diff(prev_frame, frame, 10)
        #
        # ch = colorhist(frame)
        # chd = colorhist_diff(prev_ch, ch)
        # prev_ch = ch

        bd = block_difference(prev_frame, frame)
        #
        # if ch is not None:
        #     ch_array.append(ch)
        # if chd is not None:
        #     chd_array.append(chd)
        # if td is not None:
        #     td_array.append(td)
        if bd is not None:
            bd_array.append(bd)

        prev_frame = frame

    # Compare with database

    video_types = ('*.mp4', '*.avi')

    # grab all video file names
    video_list = []
    for type_ in video_types:
        files = vids_path + '/' + type_
        video_list.extend(glob.glob(files))

    print(len(video_list))

    video_list.sort()
    query_results = []

    with open('db/vid_database.pkl', 'rb') as f:
        vid_feats = pickle.load(f)

    bd_array = np.asarray(bd_array)
    print(bd_array)
    i = 0
    for video in video_list:
        w = np.array(bd_array)
        # print(video)
        x = vid_feats[i]

        frame, score = sliding_window(x, w, euclidean_norm_mean)

        query_results.append((video, frame, score))
        i += 1
        # print ('Best match at:', frame / fps, 'seconds, with score of:', score)
        # print ('')
    query_results.sort(key=lambda x: x[2])
    # print(query_results[0][0])
    # print('Best match at:', query_results[0][1] / fps, 'seconds, with score of:', query_results[0][2])


    for n in range(3):
        print(query_results[n][0])
        print('Best match at:', query_results[n][1] / fps, 'seconds, with score of:', query_results[n][2])
        print('')


def sliding_window(x, w, compare_func):
    wl = len(w)
    minimum = sys.maxsize
    for i in range(len(x) - wl):
        diff = compare_func(w, x[i:(i+wl)])
        if diff < minimum:
            minimum = diff
            frame = i
    return frame, minimum

def euclidean_norm_mean(x,y):
    x = np.mean(x, axis=0)
    y = np.mean(y, axis=0)
    return np.linalg.norm(x-y)

def colorhist(im):
    colors = cv2.split(im)
    color_hist = np.zeros((256, len(colors)))
    for i in range(len(colors)):
        color_hist[:, i] = np.histogram(colors[i], bins=np.arange(256 + 1))[0] / float(
            (colors[i].shape[0] * colors[i].shape[1]))
    return color_hist


def colorhist_diff(hist1, hist2):
    if hist1 is None or hist2 is None:
        return None
    diff = np.abs(hist1 - hist2)
    return np.sum(diff)

def temporal_diff(frame1, frame2, threshold=10):
    if frame1 is None or frame2 is None:
        return None
    diff = np.abs(frame1.astype('int16') - frame2.astype('int16'))
    diff_t = diff > threshold
    return np.sum(diff_t)

def block_difference(prev_frame, frame):
    if prev_frame is None or frame is None:
        return None

    blocks_prev = []
    blocks = []
    width = frame.shape[1]
    height = frame.shape[0]

    block_w = width // 7
    block_h = height // 7
    for i in range(0, height, block_h):
        for j in range(0, width, block_w):
            blocks_prev.append(np.mean(prev_frame[i:i+block_h,j:j+block_w], axis=(0,1)))
            blocks.append(np.mean(frame[i:i + block_h, j:j + block_w], axis=(0,1)))

    diff = np.sum(abs(np.subtract(blocks_prev, blocks)))
    return diff






