####################################
# Python wrapper for dlib face landmark detector (Kazemi and Sullivan, CVPR'14)
import dlib
import glob
from skimage import io
import numpy as np
import os
import sys

detector = None
predictor = None


def _shape_to_np(shape):
    xy = []
    for i in range(68):
        xy.append((shape.part(i).x, shape.part(i).y,))
    xy = np.asarray(xy, dtype='float32')
    return xy


def get_landmarks(img):
    dets = detector(img, 0)
    if len(dets) == 0:
        rect = dlib.rectangle(50, 50, 200, 200)
    else:
        rect = dets[0]

    shape = predictor(img, rect)
    lmarks = _shape_to_np(shape)
    bboxes = rect

    lmarks = np.vstack(lmarks)
    bboxes = np.asarray(bboxes)

    return lmarks, bboxes


def get_identity_dirs(dir_path):
    return [os.path.join(dir_path, name) for name in os.listdir(dir_path) if
            os.path.isdir(os.path.join(dir_path, name))]


def get_identity_images(identity_dir_path):
    identity_image_paths = []
    for f in glob.glob(os.path.join(identity_dir_path, "*.jpg")):
        identity_image_paths.append(f)

    return identity_image_paths


def align_img(img, landmarks):
    return img


def align_identity_images(identity_dir_path, aligned_dir_path):
    identity_images = get_identity_images(identity_dir_path)
    identity_name = os.path.basename(os.path.split(identity_images[0])[0])

    identity_dir = os.path.join(aligned_dir_path, identity_name)
    if not os.path.isdir(identity_dir):
        os.mkdir(identity_dir)

    for i, f in enumerate(identity_images):
        print('%s: %d/%d' % (identity_name, i, len(identity_images)))
        img = io.imread(f)
        lmarks, bboxes = get_landmarks(img)
        # align image
        aligned_img = align_img(img, lmarks)
        # save image
        io.imsave(os.path.join(identity_dir, os.path.basename(f)), aligned_img)

def main():
    if len(sys.argv) != 3:
        print(
            "Give the path to the trained shape predictor model as the first "
            "argument and then the directory containing the facial images.\n"
            "For example, if you are in the python_examples folder then "
            "execute this program by running:\n"
            "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
            "You can download a trained facial shape predictor from:\n"
            "    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2")
        exit()

    # init stuff
    predictor_path = sys.argv[1]
    source_folder_path = sys.argv[2]
    aligned_folder_path = os.path.join(sys.argv[2], "..", "aligned")
    if not os.path.isdir(aligned_folder_path):
        os.mkdir(aligned_folder_path)


    global detector, predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # collect all identities
    dirs = get_identity_dirs(source_folder_path)

    # align
    for d in dirs:
        align_identity_images(d, aligned_folder_path)


if __name__ == "__main__":
    main()
