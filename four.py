import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats
import seaborn as sns

sns.set()


def plot_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()

def wellSeparatedData():
	from sklearn.datasets.samples_generator import make_blobs

	# plotting the data
	X, y = make_blobs(n_samples = 50, centers=2, random_state=0, cluster_std=0.60)
	plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
	plt.show()

	# we can seperate it by hand but there is more than one correct division
	xfit = np.linspace(-1, 3.5)
	plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
	plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

	for m, b in [(1, 0.65), (0.5, 1.6)]:
		plt.plot(xfit, m*xfit+b, '-k')

	plt.show()

	# using svm/svc
	from sklearn import svm

	model = svm.SVC(kernel='linear', C=1E10)
	model.fit(X, y)
	plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
	plot_decision_function(model)


def nonLinearSeparable():
	from sklearn.datasets.samples_generator import make_circles

	X, y = make_circles(100, factor=0.1, noise=.1)
	plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
	plt.show()

	from sklearn import svm
	clf = svm.SVC(kernel='rbf', C=1E6)
	clf.fit(X, y)
	plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
	plot_decision_function(clf)

if __name__ == '__main__':
	wellSeparatedData()
	nonLinearSeparable()


