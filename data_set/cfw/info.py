class ImageInfo(object):
    def __init__(self, image_name, image_urls):
        self.image_name = image_name
        self.original_url = image_urls[0]
        self.image_urls = image_urls[1:]


def parse_info(info_file_path):
    lines = file(info_file_path).read().splitlines()
    image_list = []
    line_index = 0

    while line_index < len(lines):
        head_line = lines[line_index]
        num_of_urls, image_name, original_url = head_line.split("\t")
        last_image_line_index = line_index+int(num_of_urls)+1
        image = ImageInfo(image_name, [original_url] + lines[line_index+1:last_image_line_index])
        image_list.append(image)
        line_index = last_image_line_index

    return image_list