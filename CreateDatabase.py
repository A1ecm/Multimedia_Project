#!/usr/bin/env python

import os
import sys
import numpy as np
import cv2
import subprocess
import re
from fractions import Fraction
import matplotlib.pyplot as plt
import argparse
import pickle
import glob
from scipy.io.wavfile import read
import db_index_video


def create_database():
    # DATABASE Creating and insertion
    # If the database already exists, we can remove it and recreate it, or we can just insert new data.
    db_name = 'db/video_database.db'

    if not os.path.exists('db'):
        os.makedirs('db')

    # check if database already exists
    new = False
    if os.path.isfile(db_name):
        action = input('Database already exists. Do you want to (r)emove, (a)ppend or (q)uit? ')
        print('action =', action)
    else:
        action = 'c'

    if action == 'r':
        print
        'removing database', db_name, '...'
        os.remove(db_name)
        new = True

    elif action == 'a':
        print('appending to database ... ')

    elif action == 'c':
        print('creating database', db_name, '...')
        new = True

    else:
        print('Quit database tool')
        sys.exit(0)

    # Create indexer which can create the database tables and provides an API to insert data into the tables.
    indx = db_index_video.Indexer(db_name)
    if new == True:
        indx.create_tables()

    return indx


#
# Processing of videos
def process_videos(video_list, indx):
    total = len(video_list)
    progress_count = 0
    for video in video_list:
        progress_count += 1
        print('processing: ', video, ' (', progress_count, ' of ', total, ')')
        cap = cv2.VideoCapture(video)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        colorhists = []
        colorhist_diffs = []
        sum_of_differences = []
        block_diffs = []

        prev_colorhist = None
        prev_frame = None
        frame_nbr = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

                # calculate sum of differences
            if not prev_frame is None:
                tdiv = temporal_diff(prev_frame, frame, 10)
                # diff = np.absolute(prev_frame - frame)
                # sum = np.sum(diff.flatten()) / (diff.shape[0]*diff.shape[1]*diff.shape[2])
                sum_of_differences.append(tdiv)
                bd = block_difference(prev_frame, frame)
                block_diffs.append(bd)

            colorhis = colorhist(frame)
            colorhists.append(colorhis)
            if not prev_colorhist is None:
                ch_diff = colorhist_diff(prev_colorhist, colorhis)
                colorhist_diffs.append(ch_diff)
            prev_colorhist = colorhis
            prev_frame = frame
            frame_nbr += 1
        print('end:', frame_nbr)

        # prepare descriptor for database
        # mfccs = descr['mfcc'] # Nx13 np array (or however many mfcc coefficients there are)
        # audio = descr['audio'] # Nx1 np array
        # colhist = descr['colhist'] # Nx3x256 np array
        # tempdif = descr['tempdiff'] # Nx1 np array
        descr = {}
        descr['colhist'] = np.array(colorhists)
        descr['tempdiff'] = np.array(sum_of_differences)
        descr['chdiff'] = np.array(colorhist_diffs)
        descr['bdiff'] = np.array(block_diffs)
        indx.add_to_index(video, descr)
        print('added ' + video + ' to database')
    indx.db_commit()


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

def temporal_diff(frame1, frame2, threshold=50):
    if frame1 is None or frame2 is None:
        return None
    diff = np.abs(frame1.astype('int16') - frame2.astype('int16'))
    diff_t = diff > threshold
    return np.sum(diff_t)


def block_difference(prev_frame, frame, threshold=50):
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


parser = argparse.ArgumentParser(
    description="Video Processing tool extracts features for each frame of video and for its corresponding audio track")
parser.add_argument("training_set", help="Path to training videos and wav files")

args = parser.parse_args()

video_types = ('*.mp4', '*.MP4', '*.avi')

# grab all video file names
video_list = []
for type_ in video_types:
    files = args.training_set + '/' + type_
    video_list.extend(glob.glob(files))

# create database
indx = create_database()
process_videos(video_list, indx)

