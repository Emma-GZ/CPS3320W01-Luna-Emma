# Global Similar (Level2): Hash- Perceptual image hashing

import cv2
import numpy as np
from PIL import Image

# 平均哈希算法计算
def classify_aHash(image1, image2):
    image1 = cv2.resize(image1, (8, 8))
    image2 = cv2.resize(image2, (8, 8))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    hash1 = getHash(gray1)
    hash2 = getHash(gray2)
    return Hamming_distance(hash1, hash2)

# 感知哈希算法(pHash)
def classify_pHash(image1, image2):
    image1 = cv2.resize(image1, (32, 32))
    image2 = cv2.resize(image2, (32, 32))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct1 = cv2.dct(np.float32(gray1))
    dct2 = cv2.dct(np.float32(gray2))
    # 取左上角的8*8，这些代表图片的最低频率
    # 这个操作等价于c++中利用opencv实现的掩码操作
    # 在python中进行掩码操作，可以直接这样取出图像矩阵的某一部分
    dct1_roi = dct1[0:8, 0:8]
    dct2_roi = dct2[0:8, 0:8]
    hash1 = getHash(dct1_roi)
    hash2 = getHash(dct2_roi)
    return Hamming_distance(hash1, hash2)

# dHash算法
def classify_dHash(add1, add2):
    hash1 = Get_dhash(add1)
    hash2 = Get_dhash(add2)

    return Hamming_distance(hash1, hash2)

# 输入灰度图，返回hash
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

def Get_dhash(img):
    hash = ''
    image = Image.open(img)
    image = np.array(image.resize((9, 8), Image.ANTIALIAS).convert('L'), 'f')  # 9*8缩放，'f'表示整个数组都是float32类型
    # 该遍历方法正好是234个像素
    for i in range(8):
        for j in range(8):
            if image[i, j] > image[i, j + 1]:
                hash += '1'
            else:
                hash += '0'
    #print(hash)
    hash = ''.join(map(lambda x: '%x' % int(hash[x: x + 4], 2), range(0, 64, 4)))  # %x：转换无符号十六进制
    return hash


# 计算汉明距离
def Hamming_distance(hash1, hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num


if __name__ == '__main__':
    add1='D:\\aa.WKU\\WKU Course\\CPS3320-W01 PYTHON PROGRAMMING\\project\\photo\\test2.jpg'
    add2='D:\\aa.WKU\\WKU Course\\CPS3320-W01 PYTHON PROGRAMMING\\project\\photo\\test3.jpg'
    img1 = cv2.imread(add1)
    img2 = cv2.imread(add2)
    
    degree1 = classify_aHash(img1,img2); print(degree1)
    degree2 = classify_pHash(img1,img2); print(degree2)
    degree3 = classify_dHash(add1, add2); print(degree3)

    #cv2.waitKey(0)


# 【Code Reference】
# 【aHash & pHash】https://blog.csdn.net/feimengjuan/article/details/51279629
# 【dHash】https://blog.csdn.net/qq_43650934/article/details/108026810
