import os
import lfw


def main(data_dir, dst_dir):
    # create destination folder
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    data_dir = os.path.abspath(data_dir)
    lfw.main(os.path.join(data_dir, 'lfw'), dst_dir)


if __name__ == "__main__":
    main('/Users/adamp/Research/Data', '/tmp/normalized_data')