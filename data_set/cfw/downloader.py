import os
import requests
import Image
import StringIO
from functools import partial
from multiprocessing import Pool
import logging

DOWNLOAD_TIMEOUT = 5

module_logger = logging.getLogger('text_logger')

def download_image_base(url):
    try:
        data = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
    except requests.Timeout:
        module_logger.error("Failed to download image because of timeout")
        return None
    except requests.ConnectionError:
        module_logger.error("Failed to download image because of connection error")
        return None
    except Exception, e:
        module_logger.error("Failed to download image error: " + str(e))
        return None

    try:
        stream = StringIO.StringIO(data.content)
        image = Image.open(stream)
        return image
    except IOError:
        # need to log parsing error
        return None


def get_num_of_pixels(image):
    (x_size, y_size) = image[0].size
    return x_size * y_size


def check_image(image):
    if image is None:
        return False

    (x_size, y_size) = image.size
    return x_size > 50 and y_size > 50
    return True


def gif_to_jpg(gif_image):
    module_logger.debug("Converting GIF to jpg")
    mypalette = gif_image.getpalette()
    if mypalette is not None:
        gif_image.putpalette(mypalette)

    new_im = Image.new("RGBA", gif_image.size)
    new_im.paste(gif_image)
    return new_im


def save_image_to_file(best_image, file_name, url):
    try:
        if best_image.format == "GIF":
            best_image = gif_to_jpg(best_image)
        elif best_image.format == "BMP":
            file_name = os.path.splitext(file_name)[0]+".bmp"
        elif best_image.format == "PNG":
            file_name = os.path.splitext(file_name)[0]+".png"

        best_image.save(file_name)
    except Exception, e:
        module_logger.error("Failed to save image using Image library: " + str(e) + " writing RAW")
        save_raw_image(url, file_name)


def save_raw_image(url, file_name):
    data = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
    file(file_name + "_raw", "wb").write(data.content)


def download_image(image_info, download_dir):
    module_logger.debug("Starting to download " + image_info.image_name)
    # result path
    file_name = os.path.join(download_dir, "original_" + image_info.image_name)
    # first we try the original URL
    image = download_image_base(image_info.original_url)
    if check_image(image):
        module_logger.debug("Original image from url " + image_info.original_url)
        save_image_to_file(image, file_name, image_info.original_url)
        return

    duplicate_images = []
    for url in image_info.image_urls:
        image = download_image_base(url)
        if check_image(image):
            duplicate_images.append((image, url))
    if not duplicate_images:
        # log list is empty
        return

    best_image = max(duplicate_images, key=get_num_of_pixels)
    index_of = duplicate_images.index(best_image)
    module_logger.debug("Duplicate image from url " + best_image[1] + " " + str(index_of))
    save_image_to_file(best_image[0], file_name, best_image[1])
    return


def download_parallel(image_info_list, download_dir_path, processes=5):
    p = Pool(processes)
    partial_download_image = partial(download_image, download_dir=download_dir_path)
    p.map(partial_download_image, image_info_list)
    p.close()
    p.join()