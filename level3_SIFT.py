# Reference:https://blog.csdn.net/weixin_43823854/article/details/102017382
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

# input
imgname_01 = 'path{image1.jpg}'
imgname_02 = 'path{image2.jpg}'

#creat a SIFT object
sift = cv2.xfeatures2d.SIFT_create()



img_01 = cv2.imread(imgname_01)
img_02 = cv2.imread(imgname_02)

#show original images with horizontal stitching
hmerge_original = np.hstack((img_01, img_02)) 
# Since the processing of plt and cv2 are different, cv2 is BGR and plt is RGB,
# so it needs to be converted by the cv2.cvtColor()
img_original = cv2.cvtColor(hmerge_original, cv2.COLOR_BGR2RGB)


fig,ax1=plt.subplots(1, 1, figsize = (20,20))
ax1.imshow(img_original)
ax1.set_title("Image_original")
plt.savefig('Image_original.png')

keypoint_01, descriptor_01 = sift.detectAndCompute(img_01,None)  
keypoint_02, descriptor_02 = sift.detectAndCompute(img_02,None)

# test
print(type(keypoint_01))
print(len(str(keypoint_01)))
print(type(descriptor_01))
print(descriptor_01.shape)



img_03 = cv2.drawKeypoints(img_01,keypoint_01,img_01,color=(255,0,255)) #draw feature points as red circles
img_04 = cv2.drawKeypoints(img_02,keypoint_02,img_02,color=(255,0,255)) 
hmerge_change = np.hstack((img_03, img_04)) # horizontal stitching
img_change = cv2.cvtColor(hmerge_change, cv2.COLOR_BGR2RGB)

fig, ax1 = plt.subplots(1, 1, figsize = (20,20))
ax1.imshow(img_change)
ax1.set_title("Image_Keypoints")
plt.savefig('img_Keypoints.png')



# BFmatcher（Brute-Force Matching）
img_01 = cv2.imread(imgname_01)
img_02 = cv2.imread(imgname_02)
keypoint_01, descriptor_01 = sift.detectAndCompute(img_01, None)
keypoint_02, descriptor_02 = sift.detectAndCompute(img_02, None)
bf = cv2.BFMatcher()

# k = 2 Returns the two closest matching points in point set 2 for each description point in point set 1, using the matches variable to store.

matches = bf.knnMatch(descriptor_01, descriptor_02, k = 2)
# adjust ratio， ratio test： threshold value
ratio = 0.5
good = []

# My understanding: Compare the Euclidean distance from mn to the point set 1 descriptor
for m,n in matches:
    if m.distance < ratio * n.distance:
        good.append([m])

img5 = cv2.drawMatchesKnn(img_01, keypoint_01, img_02, keypoint_02, good, None, flags=2)

img_sift = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB) # Grayscale processes the image

plt.rcParams['figure.figsize'] = (20.0, 20.0)
plt.imshow(img_sift)
plt.savefig('img_SIFT.png')

cv2.destroyAllWindows()
