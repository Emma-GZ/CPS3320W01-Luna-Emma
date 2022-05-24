# Global Similar (Level 2): Grayscale histogram - Normalize an exposed image.
# Modified based on [Code Reference].(https://gist.github.com/duhaime/211365edaddf7ff89c0a36d9f3f7956c).

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

    # Normalize the data from an exposed image.
    img = normalize_exposure(img)

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


def normalize_exposure(img):
    # used in 'get_img' function
    """
    Normalize the data from an exposed image.
    """
    histOfImage = get_histogram(img)  # Get a darkness data from histogram of the target image.
    # get sums of whole values in each position of the histogram
    sumsArray = np.array([sum(histOfImage[:i + 1]) for i in range(len(histOfImage))])
    # define the normalization values of each unit in the sumsArray
    norm = np.uint8(255 * sumsArray)
    # normalize the data of each position in the output image
    hei, wid = img.shape
    normalized = np.zeros_like(img)
    for i in range(0, hei):
        for j in range(0, wid):
            normalized[i, j] = norm[img[i, j]]
    return normalized.astype(int)


def pixel_sim(path_a, path_b):
    """
  Measure the pixel-level similarity between two images
  @returns:
    percentage% that measures structural similarity between the input images
  """
    img_1 = get_img('D:\\aa.WKU\\WKU Course\\CPS3320-W01 PYTHON PROGRAMMING\\project\\photo\\test2.jpg')
    img_2 = get_img('D:\\aa.WKU\\WKU Course\\CPS3320-W01 PYTHON PROGRAMMING\\project\\photo\\test2_c.jpg')
    return 1-np.sum(np.absolute(img_1 - img_2)) / (height * width) / 255

if __name__ == '__main__':
    img_a = 'a.jpg'
    img_b = 'b.jpg'
    # get the similarity values
    pixel_sim = pixel_sim(img_a, img_b)
    print(str(pixel_sim*100)+"%")
