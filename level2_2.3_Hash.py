# Global Similar (Level2): Hash- Perceptual image hashing
# [Algorithms Reference].(https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html).
# Code Reference:
# [aHash & pHash].(https://blog.csdn.net/feimengjuan/article/details/51279629).
# [dHash].(https://blog.csdn.net/qq_43650934/article/details/108026810).
import cv2
import numpy as np
from PIL import Image

# Average Hash
def classify_aHash(image1, image2):
    image1 = cv2.resize(image1, (8, 8))
    image2 = cv2.resize(image2, (8, 8))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    hash1 = getHash(gray1)
    hash2 = getHash(gray2)
    return Hamming_distance(hash1, hash2)

# Perceptual Hash (pHash)
def classify_pHash(image1, image2):
    image1 = cv2.resize(image1, (32, 32))
    image2 = cv2.resize(image2, (32, 32))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Convert the grayscale image to floating point, and then perform dct transformation
    dct1 = cv2.dct(np.float32(gray1))
    dct2 = cv2.dct(np.float32(gray2))
    # Take the 8*8 in the upper left corner, these represent the lowest frequency of the picture
    # This operation is equivalent to the mask operation implemented by opencv in C++
    # Perform mask operation in python, you can directly take out a part of the image matrix like this
    dct1_roi = dct1[0:8, 0:8]
    dct2_roi = dct2[0:8, 0:8]
    hash1 = getHash(dct1_roi)
    hash2 = getHash(dct2_roi)
    return Hamming_distance(hash1, hash2)

# Dynamic Hash
def classify_dHash(add1, add2):
    hash1 = Get_dhash(add1)
    hash2 = Get_dhash(add2)

    return Hamming_distance(hash1, hash2)

# input greyscale image，return hash (ahash and phash)
def getHash(image):
    avreage = np.mean(image)
    hash = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash

# input greyscale image，return dhash
def Get_dhash(img):
    hash = ''
    image = Image.open(img)
    image = np.array(image.resize((9, 8), Image.ANTIALIAS).convert('L'), 'f')  # 9*8 resize，'f' means float32
    # traversing 234 pixels
    for i in range(8):
        for j in range(8):
            if image[i, j] > image[i, j + 1]:
                hash += '1'
            else:
                hash += '0'
    #print(hash)
    hash = ''.join(map(lambda x: '%x' % int(hash[x: x + 4], 2), range(0, 64, 4)))  # %x：Convert unsigned hexadecimal
    return hash


# Calculate the Hamming distance
def Hamming_distance(hash1, hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num


if __name__ == '__main__':
    add1='path{image1.jpg}'
    add2='path{image2.jpg}'
    img1 = cv2.imread(add1)
    img2 = cv2.imread(add2)
    
    degree1 = classify_aHash(img1,img2); print(degree1)
    degree2 = classify_pHash(img1,img2); print(degree2)
    degree3 = classify_dHash(add1, add2); print(degree3)

    #cv2.waitKey(0)
