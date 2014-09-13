import distutils.core


def main(src_dir, dst_dir):
    distutils.dir_util.copy_tree(src_dir, dst_dir)

if __name__ == '__main__':
    main()