# without normalization of exposure
from skimage.transform import resize
from imageio import imread
import numpy as np

# specify size of image to adjust
height = 2 ** 10
width = 2 ** 10

def get_img(path):
    """
    Prepare an image for image processing tasks
    """
    # imread function converts an image to a 2d grayscale array
    img = imread(path, as_gray=True).astype(int)

    # resize function resize image to a specific size; the type of returned numbers is float
    img = resize(img, (height, width), anti_aliasing=True, preserve_range=True)

    #print(img)
    return img

if __name__ == '__main__':
    img_1 = get_img('image1.jpg')
    img_2 = get_img('image2.jpg')
    pixel_sim=(1 - np.sum(np.absolute(img_1 - img_2)) / (height * width) / 255) * 100
    # For a grayscale image in 8-bit, so [0, 255] is the range of their difference.
    print(str(pixel_sim) + "%")
