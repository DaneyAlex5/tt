import sys
import math
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture 
from sklearn.cluster import KMeans
from mpl_toolkits import mplot3d

X_1 = np.random.multivariate_normal(mean=[4, 0], cov=[[1, 0], [0, 1]], size=75)
X_2 = np.random.multivariate_normal(mean=[6, 6], cov=[[2, 0], [0, 2]], size=250)
X_3 = np.random.multivariate_normal(mean=[1, 5], cov=[[1, 0], [0, 2]], size=20)

fig = plt.figure(figsize=(18,10))
ax0 = fig.add_subplot(111)
ax0.scatter(X_1[:,0], X_1[:,1],s=50, c='y',marker='s', edgecolor='black',label='cluster 1')
ax0.scatter(X_2[:,0], X_2[:,1],s=50, c='g',marker='o', edgecolor='black',label='cluster 2')
ax0.scatter(X_3[:,0], X_3[:,1],s=50, c='b',marker='v', edgecolor='black',label='cluster 3')
plt.legend(scatterpoints=1)
plt.grid(True)
plt.show(block=False)
figure = plt.gcf()
figure.set_size_inches(18, 10)
plt.savefig('Input_dataset.png',dpi=200)

X=np.concatenate((X_1, X_2, X_3))
print('\nlength of dataset is : ',len(X))
print('\nThe dataset is\n',X)

for i in range(0,5):
    rand_num1=random.uniform(-1, 1)
    rand_num2=random.uniform(1, 3)
    rand_num3=random.uniform(3, 5)
    centroid_init=np.asarray([[rand_num1,rand_num1],[rand_num2,rand_num2],[rand_num3,rand_num3]])

    km = KMeans(n_clusters=3, init=centroid_init,n_init=1,max_iter=100, tol=0.0001, random_state=None)
    y_km = km.fit_predict(X)
    centroid_km= km.cluster_centers_[0, :], km.cluster_centers_[1, :], km.cluster_centers_[2, :]
    centroid_km=np.asarray(centroid_km)
    print('Initial values of centroids is :\n',centroid_init)
    print('Centroids found using KMM are :\n',centroid_km)
    fig = plt.figure(figsize=(18,10))
    ax0 = fig.add_subplot(111)
    ax0.scatter(X[y_km == 0, 0], X[y_km == 0, 1],s=50, c='y',marker='s', edgecolor='black',label='cluster 1')
    ax0.scatter(X[y_km == 1, 0], X[y_km == 1, 1],s=50, c='g',marker='o', edgecolor='black',label='cluster 2')
    ax0.scatter(X[y_km == 2, 0], X[y_km == 2, 1],s=50, c='b',marker='v', edgecolor='black',label='cluster 3')
    ax0.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],s=250, marker='*',c='red', edgecolor='black',label='centroids')
    plt.legend(scatterpoints=1)
    plt.grid(True)
    plt.show(block=False)
    figure = plt.gcf()
    figure.set_size_inches(18, 10)
    plt.savefig('KNN'+str(i+1)+'.png',dpi=200)

gmm = GaussianMixture(n_components=3,tol=0.0001,max_iter=100,random_state=None).fit(X)
y_gmm = gmm.fit_predict(X)
fig1 = plt.figure(figsize=(18,10))
ax0 = fig1.add_subplot(111)
ax0.scatter(X[y_gmm == 0, 0], X[y_gmm == 0, 1],s=50, c='y',marker='s', edgecolor='black',label='cluster 1')
ax0.scatter(X[y_gmm == 1, 0], X[y_gmm == 1, 1],s=50, c='g',marker='o', edgecolor='black',label='cluster 2')
ax0.scatter(X[y_gmm == 2, 0], X[y_gmm == 2, 1],s=50, c='b',marker='v', edgecolor='black',label='cluster 3')
plt.legend(scatterpoints=1)
plt.grid(True)
plt.show(block=False)
figure1 = plt.gcf()
figure1.set_size_inches(18, 10)
plt.savefig('GMM.png',dpi=200)

print('\n ************************************************Image Compression************************************************\n')
print('\nCompressing image...............')
Input_image = sys.argv[1]
image_matrix= cv2.imread(Input_image)
Shp= np.shape(image_matrix)
len_arr= Shp[0]*Shp[1]
im_2d= np.reshape(image_matrix, (len_arr, Shp[2]))
km_image= KMeans(n_clusters=30)
y_img= km_image.fit_predict(im_2d)
centroid_km=np.zeros((30,3), dtype = float)
for cl in range (0,30):
    centroid_km[cl]= km_image.cluster_centers_[cl, :]
centroid_km=np.asarray(centroid_km)
for j in range (0,len_arr):
    im_2d[j,:]=centroid_km[y_img[j],:]
cmprsd_img=np.reshape(im_2d, (Shp[0],Shp[1], Shp[2]))
cv2.imwrite('Daney_Alex2.png', cmprsd_img)




