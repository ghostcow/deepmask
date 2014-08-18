import info
import downloader
import logging

LOG_FILENAME = r'/tmp/example.log'
logger = logging.getLogger("text_logger")
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler(filename=LOG_FILENAME)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

a = info.parse_info(r"/tmp/info.txt")
for g in a:
    print g.image_name
    downloader.download_image(g, "/tmp")