import argparse



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()
    return args