import sys

import numpy as np
import cv2
import glob
import sqlite3 as sqlite
import pickle
from os.path import basename


def recognize(query, vids_path, fps):
    ch_array = []
    td_array = []
    bd_array = []
    prev_ch = None
    prev_frame = None

    for frame in query:
        # cv2.imshow("frame", frame)
        # cv2.waitKey()
        td = temporal_diff(prev_frame, frame, 10)

        ch = colorhist(frame)
        chd = colorhist_diff(prev_ch, ch)
        prev_ch = ch

        bd = block_difference(prev_frame, frame)


        if chd is not None:
            ch_array.append(chd)
        if td is not None:
            td_array.append(td)
        if bd is not None:
            bd_array.append(bd)

        prev_frame = frame

    # Compare with database

    video_types = ('*.mp4', '*.MP4', '*.avi')
    audio_types = ('*.wav', '*.WAV')

    # grab all video file names
    video_list = []
    for type_ in video_types:
        files = vids_path + '/' + type_
        video_list.extend(glob.glob(files))

    db_name = 'db/video_database.db'
    con = sqlite.connect(db_name)

    query_results = []

    for video in video_list:
        w = np.array(bd_array)
        # print(video)

        vidid = con.execute("select rowid from vidlist where filename='%s'" % basename(video)).fetchone()
        query = "select features from blockdiffs where vidid=" + str(vidid[0])
        s = con.execute(query).fetchone()
        # use pickle to decode NumPy arrays from string
        x = pickle.loads((s[0]))

        print("w")
        print(w.shape)
        print("x")
        print(x.shape)

        frame, score = sliding_window(x, w, euclidean_norm_mean)

        query_results.append((video, frame, score))

        # print ('Best match at:', frame / fps, 'seconds, with score of:', score)
        # print ('')

    query_results.sort(key=lambda x: x[2])
    # print(query_results[0][0])
    # print('Best match at:', query_results[0][1] / fps, 'seconds, with score of:', query_results[0][2])


    for n in range(7):
        print(query_results[n][0])
        print('Best match at:', query_results[n][1] / fps, 'seconds, with score of:', query_results[n][2])
        print('')


def sliding_window(x, w, compare_func):
    """ Slide window w over signal x.
        compare_func should be a functions that calculates some score between w and a chunk of x
    """
    wl = len(w)
    minimum = sys.maxsize
    for i in range(len(x) - wl):
        diff = compare_func(w, x[i:(i+wl)])
        if diff < minimum:
            minimum = diff
            frame = i
    return frame, minimum

def subtract(x,y):
    return x - y

def euclidean_norm_mean(x,y):
    x = np.mean(x, axis=0)
    y = np.mean(y, axis=0)
    return np.linalg.norm(x-y)

def colorhist(im):
    chans = cv2.split(im)
    color_hist = np.zeros((256, len(chans)))
    for i in range(len(chans)):
        color_hist[:, i] = np.histogram(chans[i], bins=np.arange(256 + 1))[0] / float(
            (chans[i].shape[0] * chans[i].shape[1]))
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

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    blocks_prev = []
    blocks = []
    width = frame.shape[1]
    height = frame.shape[0]

    block_w = width // 6
    block_h = height // 6

    for i in range(0, height, block_h):
        for j in range(0, width, block_w):
            blocks_prev.append(np.mean(prev_frame[i:i+block_h,j:j+block_w]))
            blocks.append(np.mean(frame[i:i + block_h, j:j + block_w]))
    diff = np.abs(np.asarray(blocks) - np.asarray(blocks_prev))
    return np.sum(diff)






