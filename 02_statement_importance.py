import os
from pandas import read_csv
import numpy as np
from sklearn import tree

#import graphviz as gv

# os.chdir('C:\\Users\\Nick')
#print(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

df_csv = read_csv('titanic.csv', usecols=['Pclass', 'Sex', 'Age', 'Fare', 'Survived'])
dfnonan = df_csv.dropna(how='any')
features = ['Pclass', 'Sex', 'Age', 'Fare']
data = dfnonan[features]
surv = dfnonan[['Survived']]
data["Sex"].replace("female", "0", inplace=True)
data["Sex"].replace("male", "1", inplace=True)
clf = tree.DecisionTreeClassifier(random_state=241)
clf.fit(data, surv)
print(features)
print(clf.feature_importances_)
imp = clf.feature_importances_
imp_max = max(imp)
print(features[np.argmax(imp)], imp_max)

i = 0
for item in imp:
    if item == imp_max:
        imp2 = np.delete(imp, i, None)
        features.remove(features[i])
        break
    i = i+1
imp_max = max(imp2)
print(features[np.argmax(imp2)], imp_max)

'''
dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True)
graph = gv.Source(dot_data, format="png")
graph.render('graph.gv', view=True, cleanup=True)
'''
