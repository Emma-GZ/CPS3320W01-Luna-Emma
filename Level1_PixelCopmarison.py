# Pixel Level Comparison
from skimage.transform import resize

from imageio import imread
# specify size of image to adjust
height = 2 ** 10
width = 2 ** 10
def get_img(path):
    """
    Prepare an image for image processing tasks
    """
    # imread function converts an image to a 2d grayscale array
    img = imread(path, as_gray=True).astype(int)

    # resize function resize image to a specific size;
    img = resize(img, (height, width), anti_aliasing=True, preserve_range=True)

    return img

if __name__ == '__main__':
    img_1 = get_img('image1.jpg')
    img_2 = get_img('image2.jpg')

if img_1.shape==img_2.shape:
    for i in range(img_1.shape[0]):
        for j in range(img_1.shape[1]):
            if img_1[i][j]!=img_2[i][j]:
                print("F")
                exit()
    print("T")
else:
    print("F")
