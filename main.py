import argparse
import os
import FindFrames



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, default='')
    parser.add_argument('--video_base', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    query_path = args.query
    video_base_path= args.video_base

    FindFrames.get_frames(query_path, video_base_path)

