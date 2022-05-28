# Reference:https://blog.csdn.net/weixin_43823854/article/details/102017382
import numpy as np
import cv2
from matplotlib import pyplot as plt
imgname_01 = 'path{image1.jpg}'
imgname_02 = 'path{image2.jpg}'

# creat an ORB object
orb = cv2.ORB_create()

# detect feature points
img_01 = cv2.imread(imgname_01)
keypoint_01, descriptor_01 = orb.detectAndCompute(img_01,None)
img_02 = cv2.imread(imgname_02)
keypoint_02, descriptor_02 = orb.detectAndCompute(img_02,None)
img_03 = cv2.drawKeypoints(img_01, keypoint_01, img_01, color=(255, 255, 0))
img_04 = cv2.drawKeypoints(img_02, keypoint_02, img_02, color=(255, 255, 0))

img_original = np.hstack((img_03, img_04)) # Horizontal stitching
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)


fig,ax1=plt.subplots(1, 1, figsize = (20,20))
ax1.imshow(img_original)
ax1.set_title("img_original")
plt.savefig('img_original.png')


# BFMatcher
img_01 = cv2.imread(imgname_01)
img_02 = cv2.imread(imgname_02)

bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptor_01,descriptor_02, k=2)
print(type(matches))#<class 'list'>

# adjust ratio
good = []
ratio = 0.7
for m,n in matches:
    if m.distance < ratio * n.distance:
        good.append([m])

img_05 = cv2.drawMatchesKnn(img_01,keypoint_01,img_02,keypoint_02,good,None,flags=2)
img_ORB = cv2.cvtColor(img_05, cv2.COLOR_BGR2RGB)

fig,ax1=plt.subplots(1, 1, figsize = (20,20))
ax1.imshow(img_ORB)
ax1.set_title("img_ORB")
plt.savefig('img_ORB.png')

cv2.destroyAllWindows()
