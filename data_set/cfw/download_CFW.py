import os
import info
import downloader
import logging

# create logger
LOG_FILENAME = r'/media/adam/FAA85049A8500713/Users/Adam/Desktop/Research/Data/CFW/thumbnails_features_deduped_publish/example.log'
DOWNLOADED_CELEBS_LOG = r'/media/adam/FAA85049A8500713/Users/Adam/Desktop/Research/Data/CFW/thumbnails_features_deduped_publish/downloaded.log'
logger = logging.getLogger("text_logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(filename=LOG_FILENAME)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name))]


def main(root_dir):
    logger.debug("Starting to download dataset to: " + root_dir)
    celeb_dirs = get_immediate_subdirectories(root_dir)
    downloaded_celebs = file(DOWNLOADED_CELEBS_LOG).read().splitlines()

    for celeb_dir in celeb_dirs:
        # check if celeb was already downloaded
        if celeb_dir in downloaded_celebs:
            continue

        logger.debug("Starting to download " + celeb_dir + " images")
        downloaded_dir_path = os.path.join(root_dir, celeb_dir, "full_size")
        if not os.path.exists(downloaded_dir_path):
            logger.debug("Creating " + celeb_dir + " directory")
            os.mkdir(downloaded_dir_path)

        print celeb_dir
        celeb_info = info.parse_info(os.path.join(root_dir, celeb_dir, "info.txt"))
        downloader.download_parallel(celeb_info, downloaded_dir_path, processes=6)


if __name__ == "__main__":
    main(r'/media/adam/FAA85049A8500713/Users/Adam/Desktop/Research/Data/CFW/thumbnails_features_deduped_publish')