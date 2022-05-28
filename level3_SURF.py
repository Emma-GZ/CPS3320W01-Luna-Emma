# Reference:https://blog.csdn.net/weixin_43823854/article/details/102017382
import numpy as np
import cv2
from matplotlib import pyplot as plt
imgname_01 = 'D:\\aa.WKU\\WKU Course\\CPS3320-W01 PYTHON PROGRAMMING\\project\\photo\\mouse4.jpg'
imgname_02 = 'D:\\aa.WKU\\WKU Course\\CPS3320-W01 PYTHON PROGRAMMING\\project\\photo\\mouse4.jpg'

# Set a threshold, the larger the threshold, the fewer features that can be recognized
#surf = cv2.xfeatures2d.SURF_create(1000)
surf = cv2.xfeatures2d.SURF_create()

# Get two similar images img_01 and img_02

img_01 = cv2.imread(imgname_01)
keypoint_01, descriptor_01 = surf.detectAndCompute(img_01,None)
img_02 = cv2.imread(imgname_02)
keypoint_02, descriptor_02 = surf.detectAndCompute(img_02,None)
img_03 = cv2.drawKeypoints(img_01, keypoint_01, img_01, color=(255, 255, 0))
img_04 = cv2.drawKeypoints(img_02, keypoint_02, img_02, color=(255, 255, 0))

img_original = np.hstack((img_03, img_04)) # Horizontal stitching
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB) 


fig,ax1=plt.subplots(1, 1, figsize = (20,20))
ax1.imshow(img_original)
ax1.set_title("img_original")
plt.savefig('img_original.png')

'''
FLANN(Fast_Library_for_Approximate_Nearest_Neighbors)，is a collection of nearest neighbor search algorithms for large data sets and high-dimensional features,
and these algorithms have been optimized and its effect is better than BFMatcher when facing large data sets.

Using FLANN matching requires passing in two dictionary parameters:
1. IndexParams
For SIFT&SURF, index_params=dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
For ORB, index_params=dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12)
2. SearchParams
search_params=dict(checks=100), which specifies the number of recursive traversals. 
The higher the value, the more accurate the result, but the more time it takes.
'''

img_01 = cv2.imread(imgname_01)
img_02 = cv2.imread(imgname_02)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params,search_params)


matches = flann.knnMatch(descriptor_01, descriptor_02, k = 2)
ratio = 0.6

good = []
for m,n in matches:
    if m.distance < 0.6 * n.distance:
        good.append([m])
img_04 = cv2.drawMatchesKnn(img_01, keypoint_01, img_02 ,keypoint_02, good, None, flags=2)
img_SURF = cv2.cvtColor(img_04, cv2.COLOR_BGR2RGB) 


fig,ax1=plt.subplots(1, 1, figsize = (20,20))
ax1.imshow(img_SURF)
ax1.set_title("img_SURF")
plt.savefig('img_SURF.png')

cv2.destroyAllWindows()
