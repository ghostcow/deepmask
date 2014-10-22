import distutils.core
import os


def main(src_dir, dst_dir):
    dst_dir = os.path.join(dst_dir, 'pubfig83')
    images_dirs_path = os.listdir(src_dir)

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for image_dir_path in images_dirs_path:
        if not "." in image_dir_path:
            distutils.dir_util.copy_tree(os.path.join(src_dir, image_dir_path),
                                         os.path.join(dst_dir, image_dir_path.replace(' ', '_')))

if __name__ == '__main__':
    main()