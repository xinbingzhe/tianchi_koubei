import numpy as np
'''
from sklearn.cross_validation import cross_val_score
from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import KFold
'''
#iris = datasets.load_iris()
#print(iris.data.shape, iris.target.shape)
#print(iris)
#clf = svm.SVC(kernel='linear', C=1)
#scores = cross_val_score(clf, iris.data, iris.target, cv=5)
#print(scores)
'''
X = np.array([[1.1, 2.3], [3.5, 4.6], [1.0, 2.0], [3.0, 4.0]])
y = np.array([1.0, 2.0, 3.0, 4.0])
kf = KFold(4, n_folds=2)
len(kf)
for train_index, test_index in kf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

n = [12,23,2]
un = {0.2:1,0.5:2,0.6:3}



i = [d for d in range(-21,-1,1)]
print(i)
print(m[-1])
j = 0.9*np.array(m)
print(j)
#print(ab)

for o,p in zip(m,n):
    print(o)
    print(p)

print([i for i in range(1,10)])

'''
a = [[2,3],[2,3]]
m = [12,23,2,908]
a = [[1,1],
      [1,1]]
a = np.matrix(a)
b = np.array([[1,1],
      [1,1]])
aj = np.dot(b,b)
#print(type(a))
print(a[0])