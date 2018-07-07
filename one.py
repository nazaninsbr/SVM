import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm 
from sklearn.datasets import make_blobs

# create 40 points
# make blob is a dataset of the datasets in sklearn 
X, Y = make_blobs(n_samples=30, centers=2, random_state=6)

# fit the model, don't regularize 
# make the svm model using the linear kernel 
# fit the model on our points 
# make a scatter plot of the points 
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, Y)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=30, cmap=plt.cm.Paired)

# plot the decision function 
# we get the x limit and y limit to plot it between the limits 
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# print(xlim)
# print(ylim)


# grid to evaluate model 
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T 
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins 
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# plot support vector 
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none')

plt.show()