import sys
import pickle
import glob
from Recognize import *


# grab all video file names
video_types = ('*.mp4', '*.avi')
videos = []
vids = []
for type_ in video_types:
    files = 'vids_test' + '/' + type_
    videos.extend(glob.glob(files))
videos.sort()

print(len(videos))
# loop over all videos in the database and compare frame by frame
for vid in videos:
    print(vid)
    v_array = []
    video = cv2.VideoCapture(vid)
    got_frame = True
    prev_frame = None

    while got_frame:
        got_frame, frame = video.read()
        if frame is None:
            break

        if prev_frame is not None:
            bd = block_difference(prev_frame, frame)
            v_array.append(bd)

        prev_frame = frame

    vids.append(v_array)


with open('db/vid_database.pkl', 'wb') as f:
    pickle.dump(vids, f)