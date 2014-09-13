import glob
import os
import re
import shutil

IMAGE_DIR_PATH = 'image_files/SUFR-W/SUFR_in_the_wild'
INFO_FILE_PATH = 'info_files/SUFR-W/SUFR_in_the_wild/SUFR_in_the_wild_info.txt'


def generate_image_to_identity_dict(info_file_path):
    image_to_ident = {}
    info_data = file(info_file_path).readlines()
    for i in xrange(len(info_data)):
        ident = re.match("[0-9]* # ([a-zA-Z ,.'-]+)", info_data[i]).groups()[0]
        # images are counted from 1
        image_to_ident[i+1] = ident.replace(' ', '_')

    return image_to_ident


def main(src_dir, dst_dir):
    image_dir = os.path.join(src_dir, IMAGE_DIR_PATH)
    info_file_path = os.path.join(src_dir, INFO_FILE_PATH)
    image_to_ident = generate_image_to_identity_dict(info_file_path)

    for image_file in glob.glob(os.path.join(image_dir, '*.jpg')):
        image_id = int(os.path.splitext(os.path.basename(image_file))[0])
        ident = image_to_ident[image_id]
        ident_dir_path = os.path.join(dst_dir, ident)

        if not os.path.exists(ident_dir_path):
            os.mkdir(ident_dir_path)

        shutil.copyfile(image_file, os.path.join(ident_dir_path, os.path.basename(image_file)))






if __name__ == '__main__':
    main()