import distutils.core


def main(lfw_dir, dst_dir):
    distutils.dir_util.copy_tree(lfw_dir, dst_dir)

if __name__ == '__main__':
    main()