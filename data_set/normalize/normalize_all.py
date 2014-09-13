import os
import SUFR
import morph
import lfw
import pubfig83
import pubfig


def main(data_dir, dst_dir):
    # create destination folder
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    data_dir = os.path.abspath(data_dir)

    lfw.main(os.path.join(data_dir, 'lfw'), dst_dir)  # <name>_<id>.jpg
    pubfig83.main(os.path.join(data_dir, 'pubfig83'), dst_dir)  # <image id>.jpg
    pubfig.main(os.path.join(data_dir, 'pubfig'), dst_dir)  # <md5>.jpg
    SUFR.main(os.path.join(data_dir, 'SUFRData'), dst_dir)  # <image id>.jpg
    morph.main(os.path.join(data_dir, 'morph'), dst_dir)  # morph_<id>_<image_no>.jpg

if __name__ == "__main__":
    main('/Users/adamp/Research/Data', '/tmp/normalized_data')