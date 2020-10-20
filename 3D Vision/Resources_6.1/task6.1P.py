import numpy as np
import os

ROOT_DIR = "/Users/johnathontoh/Desktop/Task 6.1P/Resources_6.1"

#setup camera with a simple camera matrix P
f = 100
cx = 200
cy = 200
# s = 0 because most of the modern camera do not have any scale factor
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
#[[1,0,0],[0,1,0],[0,0,1]]
# #i.e., R
I = np.eye(3)
t = np.array([[0], [0], [0]])
# P is a 3x4 matrix
P = np.dot(K, np.hstack((I, t)))

def project(P, X): #X is an array of 3D points, X is represented in homogeneous coordinates, 4 elements
    # P is a 3x4 matrix and X is a 4x1 matrix P.X --> 3x1 matrix
    x = np.dot(P, X) # x is a 3x1 matrix
    for i in range(3): #convert to inhomogeneous coordinates
        x[i] /= x[2] # divide x by the last element [x,y,scalar] for points on 2D image
    return x

#load data
#T means tranpose
# contains 3 points x,y,z in 3D
points_3D = np.loadtxt(os.path.join(ROOT_DIR, 'house.p3d')).T

# making it homogeneous
# points_3D is an array of 4 rows and n columns, n is the number of points in house.p3d
points_3D = np.vstack((points_3D, np.ones(points_3D.shape[1])))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
fig = plt.figure(figsize = [15,15])
ax = fig.gca(projection = "3d")
ax.view_init(elev = None, azim = None) #you can set elevation and azimuth with different values
ax.plot(points_3D[0], points_3D[1], points_3D[2], 'o')
plt.draw()
plt.show()

#projection
points_2D = project(P, points_3D)

#plot projection
from matplotlib import pyplot as plt
plt.plot(points_2D[0], points_2D[1], 'k.')
plt.show()

print(points_2D.shape)
print(points_3D.shape)


n_array_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for j in n_array_points:

    n_points = int(j*points_3D.shape[1])
    points_3D_sampled = points_3D[:,:n_points]
    points_2D_sampled = points_2D[:,:n_points]

    #print(points_3D_sampled)
    #print(points_2D_sampled)

    A = np.zeros((2*n_points, 12), np.float32)

    for i in range(n_points):
        A[2*i,:4] = points_3D_sampled[:,i].T
        A[2*i,8:12] = -points_2D_sampled[0,i] * points_3D_sampled[:,i].T
        A[2*i+1,4:8] = points_3D_sampled[:,i].T
        A[2*i+1,8:12] = -points_2D_sampled[1,i] * points_3D_sampled[:,i].T

    from scipy import linalg

    U, S, V = linalg.svd(A)

    minS = np.min(S)
    conditon = (S == minS)
    minID = np.where(conditon)
    print('index of the smallest singular value is: ', minID[0])

    P_hat = V[minID[0],:].reshape(3, 4) / minS

    #print(P)
    #print(P_hat)

    # estimated 2D points from P_hat
    x_P_hat = project(P_hat, points_3D_sampled[:, 0])
    #print(x_P_hat)

    # Original 2D points from P
    x_P = points_2D_sampled[:,0]
    #print(x_P)


    x_P = points_2D
    x_P_hat = project(P_hat, points_3D)
    dist = 0
    for i in range(x_P.shape[1]):
        dist += np.linalg.norm(x_P[:,i] - x_P_hat[:,i])
    # the smaller the distance, the more accurate the estimation of P is
    dist /= x_P.shape[1]
    print("For {}% of the total n_points ({} n_points), the distance is {}\n".format(j*100,n_points,dist))

"""
points_3D --> get from house.p3d

section 1:
manually define P and use P as the camera matrix to get points_2D

section 2:
points_2D_sampled and points_3D_sampled to estimate a camera matrix P_hat

P_hat + points_3D --> x_P_hat (2D points)

Compare x_P_hat with points_2D (original points)
"""


import homography
import sfm
import ransac

import cv2 as cv
sift = cv.xfeatures2d.SIFT_create()
img1 = cv.imread(os.path.join(ROOT_DIR,'alcatraz1.jpg' ))
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) 
kp1, des1 = sift.detectAndCompute(img1_gray, None)

img2 = cv.imread(os.path.join(ROOT_DIR,'alcatraz2.jpg'))
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY) 
kp2, des2 = sift.detectAndCompute(img2_gray, None)


img1_kp = img1.copy()
img1_kp = cv.drawKeypoints(img1, kp1, img1_kp)
print("Number of detected keypoints in img1: %d" % (len(kp1)))

img2_kp = img2.copy()
img2_kp = cv.drawKeypoints(img2, kp2, img2_kp)
print("Number of detected keypoints in img2: %d" % (len(kp2)))

img1_2_kp = np.hstack((img1_kp, img2_kp))
plt.figure(figsize = (20, 10)) 
plt.imshow(img1_2_kp[:,:,::-1])
plt.axis('off')
plt.show()

#crossCheck = True means we want to find consistent matches
bf = cv.BFMatcher(crossCheck = True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)
print("Number of consistent matches: %d" % len(matches))

img1_2_matches = cv.drawMatches(img1, kp1, img2, kp2, matches[:20],
                                None,
                                flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize = (20, 10))
plt.imshow(img1_2_matches[:,:,::-1])
plt.axis('off')
plt.show()

n_matches = 1000
kp1_array = np.zeros((2, n_matches), np.float32)
for i in range(n_matches):
    kp1_array[0][i] = kp1[matches[i].queryIdx].pt[0]
    kp1_array[1][i] = kp1[matches[i].queryIdx].pt[1]

kp2_array = np.zeros((2, n_matches), np.float32)


for i in range(n_matches):
    kp2_array[0][i] = kp2[matches[i].trainIdx].pt[0]
    kp2_array[1][i] = kp2[matches[i].trainIdx].pt[1]


x1 = homography.make_homog(kp1_array)
x2 = homography.make_homog(kp2_array)

K = np.array([[2394,0,932], [0,2398,628], [0,0,1]])
P1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])

x1n = np.dot(linalg.inv(K), x1)
x2n = np.dot(linalg.inv(K), x2)

#estimate E with RANSAC
model = sfm.RansacModel()
E, inliers = sfm.F_from_ransac(x1n, x2n, model)


#compute camera matrices (P2 will be list of four solutions)
P2_all = sfm.compute_P_from_essential(E)

#pick the solution with points in front of cameras
ind = 0
maxres = 0
for i in range(4):
    #triangulate inliers and compute depth for each camera
    X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2_all[i])
    d1 = np.dot(P1, X)[2]
    d2 = np.dot(P2_all[i], X)[2]
    s = sum(d1 > 0) + sum(d2 > 0)
    if s > maxres:
        maxres = s
        ind = i
        infront = (d1 > 0) & (d2 > 0)
P2 = P2_all[ind]

#triangulate inliers and remove points not in front of both cameras
X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2)
X = X[:, infront]

print(len(X[0]))

#3D plot
fig = plt.figure(figsize = [20,20])
ax = fig.gca(projection = "3d")
ax.view_init(elev = None, azim = None) #you can set elevation and azimuth with different values
ax.plot(X[0], X[1], X[2], 'o')
plt.draw()
#plt.show()

x_hat = project(P2, X)
print(x_hat.shape)
img_coor = np.dot(K, x_hat)


plt.figure()
plt.imshow(img2)
plt.plot(kp2_array[0], kp2_array[1], 'r.')
plt.show()

plt.figure()
plt.imshow(img2)
plt.plot(img_coor[0], img_coor[1], 'g.')
plt.show()