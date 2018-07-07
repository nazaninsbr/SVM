import matplotlib
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm, datasets

matplotlib.use('GTKAgg')

# import data, take two features of it
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
h = .02

x_min , x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min , y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.title('Data')
plt.show()


# try different kernels 
C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)

titles = ['linear kernel', 'LinearSVC', 'RBF kernel', 'Polynomial (degree=3) kernel']

for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
	plt.subplot(2, 2, i+1)
	z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	z = z.reshape(xx.shape)
	plt.contourf(xx, yy, z, cmap=plt.cm.coolwarm, alpha=0.8)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
	plt.title(titles[i])

plt.show()