import numpy as np
import graphviz as gv
from sklearn import tree


X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)
importances = clf.feature_importances_
dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True)
graph = gv.Source(dot_data, format="png")
graph.render('graph.gv', view=True, cleanup=True)

