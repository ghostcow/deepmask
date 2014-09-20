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

    lfw.main(os.path.join(data_dir, 'lfw'), os.path.join(dst_dir, 'lfw'))  # <name>_<id>.jpg
    pubfig83.main(os.path.join(data_dir, 'pubfig83'), os.path.join(dst_dir, 'pubfig'))  # <image id>.jpg
    pubfig.main(os.path.join(data_dir, 'pubfig'), os.path.join(dst_dir, 'pubfig'))  # <md5>.jpg
    SUFR.main(os.path.join(data_dir, 'SUFRData'), os.path.join(dst_dir, 'SUFR'))  # <image id>.jpg
    morph.main(os.path.join(data_dir, 'morph'), os.path.join(dst_dir, 'morph'))  # morph_<id>_<image_no>.jpg

if __name__ == "__main__":
    main('/Users/adamp/Research/Data', '/tmp/normalized_data')