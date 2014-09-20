import os
import shutil
import csv
from collections import defaultdict


def main(src_dir, dst_dir):
    dst_dir = os.path.join(dst_dir, 'MEDS')
    idents_img_list = defaultdict(list)
    image_dir = os.path.join(src_dir, 'NIST_SD32_MEDS-II/data')
    metadata_file_path = 'NIST_SD32_MEDS-II/NIST_SD32_MEDS-II_metadata/subject_metadata.csv'

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    # parse metadata file - generate list of frontal face
    metadata_file = file(os.path.join(src_dir, metadata_file_path), 'rb')
    metadata_parser = csv.reader(metadata_file)
    metadata_parser.next() # skip header line
    for metadata_line in metadata_parser:
        if 'Profile' in metadata_line[16]:
            continue

        img_path = os.path.join(image_dir, os.path.basename(metadata_line[0]), metadata_line[1])
        subject_id = int(metadata_line[2])
        idents_img_list[subject_id].append(img_path)

    for subject_id in idents_img_list:
        ident_dir_path = os.path.join(dst_dir, '_'.join(['MEDS', str(subject_id)]))
        if not os.path.exists(ident_dir_path) and len(idents_img_list[subject_id]) != 0:
            os.mkdir(ident_dir_path)

        for img_path in idents_img_list[subject_id]:
            image_file_path = os.path.normpath(os.path.join(image_dir, img_path))
            dst_image_file_path = os.path.join(ident_dir_path, os.path.basename(img_path))

            if os.path.exists(image_file_path):
                shutil.copyfile(image_file_path, dst_image_file_path)
            else:
                print "{image_path} is missing from dataset".format(image_path=image_file_path)

if __name__ == '__main__':
    main()