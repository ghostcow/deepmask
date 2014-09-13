import glob
import os
import re
import shutil

DEV_IMAGE_DIR_PATH = 'dev'
EVAL_IMAGE_DIR_PATH = 'eval'
INFO_FILE_PATH = '%s_urls.txt'


def generate_image_to_identity_dict(info_file_path):
    image_to_ident = {}

    # trim first 2 lines - headers
    info_data = file(info_file_path).readlines()[2:]
    for info_line in info_data:
        ident = info_line.split('\t')
        image_to_ident[ident[4].strip()] = ident[0].replace(' ', '_')

    return image_to_ident


def main(src_dir, dst_dir):
    for image_dir in [DEV_IMAGE_DIR_PATH, EVAL_IMAGE_DIR_PATH]:
        info_file_path = os.path.join(src_dir, INFO_FILE_PATH % image_dir)
        image_dir = os.path.join(src_dir, image_dir)
        image_to_ident = generate_image_to_identity_dict(info_file_path)

        for image_file in glob.glob(os.path.join(image_dir, '*.jpg')):
            image_id = os.path.splitext(os.path.basename(image_file))[0]
            ident = image_to_ident[image_id]
            ident_dir_path = os.path.join(dst_dir, ident)

            if not os.path.exists(ident_dir_path):
                os.mkdir(ident_dir_path)

            shutil.copyfile(image_file, os.path.join(ident_dir_path, os.path.basename(image_file)))






if __name__ == '__main__':
    main()