####################################
# Python wrapper for dlib face landmark detector (Kazemi and Sullivan, CVPR'14)
import dlib
import glob
from skimage import io

io.use_plugin('matplotlib')
import numpy as np
import os
import sys
import skimage.transform

detector = None
predictor = None
dst = np.dstack(([25.0347, 34.1802, 44.1943, 53.4623, 34.1208, 39.3564, 44.9156, 31.1454, 47.8747],
                 [34.1580, 34.1659, 34.0936, 33.8063, 45.4179, 47.0043, 45.3628, 53.0275, 52.7999])).squeeze()
dst *= 3.5


def _shape_to_np(shape, indices=None):
    xy = []

    if indices is None:
        indices = range(shape.num_parts)

    for i in indices:
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
    landmarks = _shape_to_np(shape)
    bboxes = rect

    landmarks = np.vstack(landmarks)
    bboxes = np.asarray(bboxes)

    return landmarks, bboxes


def get_identity_dirs(dir_path):
    return [os.path.join(dir_path, name) for name in os.listdir(dir_path) if
            os.path.isdir(os.path.join(dir_path, name))]


def get_identity_images(identity_dir_path):
    identity_image_paths = []
    for f in glob.glob(os.path.join(identity_dir_path, "*.jpg")):
        identity_image_paths.append(f)

    return identity_image_paths


def align_img(img, landmarks):
    src = landmarks[[37, 40, 43, 46, 32, 34, 36, 49, 55]]
    affine_transform = skimage.transform.estimate_transform('similarity', src, dst)

    return skimage.transform.warp(img, affine_transform.inverse, order=3,
                                  output_shape=(300, 300))


def crop_image(img):
    # horizontal centre
    cx = dst[:, 0][[0, 1, 2, 3, 7, 8]].mean()

    # eye line
    top = dst[:, 1][[0, 1, 2, 3]].mean()

    # mouth line
    bottom = dst[:, 1][[7, 8]].mean()

    # horizontal distance between eyes
    dx = dst[:, 0][[2, 3]].mean() - dst[:, 0][[0, 1]].mean()


    # vertical distance between eyes & mouth
    dy = bottom - top

    horRatio = 1.5
    topRatio = 1.6
    bottomRatio = 0.8

    x0 = np.floor(cx - dx * horRatio)
    x1 = np.floor(cx + dx * horRatio)
    y0 = np.maximum(np.floor(top - dy * topRatio), 0)
    y1 = np.floor(bottom + dy * bottomRatio)

    return img[y0:y1, x0:x1, :]


def align_identity_images(identity_dir_path, aligned_dir_path):
    identity_images = get_identity_images(identity_dir_path)
    identity_name = os.path.basename(os.path.split(identity_images[0])[0])

    identity_dir = os.path.join(aligned_dir_path, identity_name)
    if not os.path.isdir(identity_dir):
        os.mkdir(identity_dir)

    for i, f in enumerate(identity_images):
        print('%s: %d/%d' % (identity_name, i, len(identity_images)))
        img = io.imread(f)
        landmarks, bboxes = get_landmarks(img)
        # align image
        aligned_img = align_img(img, landmarks)
        aligned_img = crop_image(aligned_img[::-1])
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
