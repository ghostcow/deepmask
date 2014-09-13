import glob
import os
import re
import shutil


def main(src_dir, dst_dir):
    image_dir = os.path.join(src_dir, 'Album2')

    for image_file_path in glob.glob(os.path.join(image_dir, '*.JPG')):
        image_name = os.path.basename(image_file_path)
        result = re.match("([0-9]*)_([0-9]{0,2})[MF][0-9]{0,2}.JPG", image_name)
        ident = '_'.join(['morph', result.groups()[0], result.groups()[1]])
        ident_dir_path = os.path.join(dst_dir, '_'.join(['morph', result.groups()[0]]))

        if not os.path.exists(ident_dir_path):
            os.mkdir(ident_dir_path)

        shutil.copyfile(image_file_path, os.path.join(ident_dir_path, ident))


if __name__ == '__main__':
    main()