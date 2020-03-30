import numpy as np
import cv2

vimax = 5
videomaxavg = []

def recognize(query, vids_path):
    vids = read_vids(vids_path)






def read_vids(vids_path):
    vids = []


def initialize(queryvideo, collpath):
    for vindex in range(vimax):
        currvideo = collpath + str(vindex) + ".mp4"
        readVideo(queryvideo, currvideo)
    print("Video Recognized is")


def readVideo(qv, cv):
    cap1 = cv2.VideoCapture(qv)
    qvf = []
    cap2 = cv2.VideoCapture(cv)
    cvf = []

    ret = True
    while (ret):
        # Capture frame-by-frame qv
        ret, frame = cap1.read()
        qvf.append(frame)
    # When everything done, release the capture
    cap1.release()
    cv2.destroyAllWindows()

    ret = True

    while (ret):
        # Capture frame-by-frame cv
        ret, frame = cap2.read()
        cvf.append(frame)
    # When everything done, release the capture
    cap2.release()
    cv2.destroyAllWindows()

    print("read qv " + qv + " and cv " + cv + ",they have " + str(len(qvf)) + " frames and " + str(
        len(cvf)) + " frames respectively.")
    cascade(qvf, cvf)


def cascade(qva, cva):
    match = np.zeros(qva.len())
    for i in range(cva.len() - qva.len()):
        for j in range(i, i + qva.len()):
            match.append(compareFrames(qva[j], cva[j]))
    avg = []
    single = 0
    for i in range(match.len()):
        for j in range(qva.len()):
            single = single + match[j]
        average = single / qva.len()
        avg.append(average)
        videomaxavg.append(max(avg))


def compareFrames(qvf, cvf):
    count = 0
    total = qvf.len() * qvf[0].len()
    for i in range(qvf.len()):
        for j in range(qvf[0].len()):
            if (qvf[i][j] == cvf[i][j]):
                count += 1
    percent = (count / total) * 100
    return count


collpath = "vids/"
videoarr = ["Asteroid_Discovery.mp4", "BlackKnight.mp4", "British_Plugs_Are_Better.mp4",
            "TUDelft_Ambulance_Drone.mp4"]

qv = "ambulance_query.mp4"
initialize(qv, collpath)