# without normalization of exposure
import warnings
from skimage.transform import resize
from imageio import imread
import numpy as np

warnings.filterwarnings('ignore')  # ignore warnings

# specify size of image to adjust
height = 2 ** 10
width = 2 ** 10


def get_img(path):
    """
    Prepare an image for image processing tasks
    """
    # imread function converts an image to a 2d grayscale array
    img = imread(path, as_gray=True).astype(int)

    # resize function resize image to a soecific size; the type of returned numbers is float
    img = resize(img, (height, width), anti_aliasing=True, preserve_range=True)

    # float → int
    img = img.astype(int)

    return img


def get_histogram(img):
    # used in 'normalize_exposure' function
    """
    Get a histogram of the target image. For a grayscale image in 8-bit,
    so the histogram is a 256 units vector that represents the pixels in the image
    and all values sum to 1.
    """
    hei, wid = img.shape  # get height and width of the target image
    hist = [0.0] * 256  # create a list with 256 vacancy
    for i in range(hei):
        for j in range(wid):
            hist[img[i, j]] += 1
    return np.array(hist) / (hei * wid)

def pixel_sim(path_a, path_b):
    """
  Measure the pixel-level similarity between two images

  @returns:
    percentage% that measures structural similarity between the input images
  """
    img_1 = get_img('D:\\aa.WKU\\WKU Course\\CPS3320-W01 PYTHON PROGRAMMING\\project\\photo\\test4.jpg')
    img_2 = get_img('D:\\aa.WKU\\WKU Course\\CPS3320-W01 PYTHON PROGRAMMING\\project\\photo\\test4.jpg')
    return ( 1-np.sum(np.absolute(img_1 - img_2)) / (height * width) / 255)*100


if __name__ == '__main__':
    img_a = 'a.jpg'
    img_b = 'b.jpg'
    # get the similarity values
    pixel_sim = pixel_sim(img_a, img_b)
    print(str(pixel_sim)+"%")


# 【Reference】https://gist.github.com/duhaime/211365edaddf7ff89c0a36d9f3f7956c
