import pandas as pd

dataset = pd.read_csv("features_best_50.csv", index_col=0)
dataset.drop(dataset.tail(1).index, inplace = True)

dataset_test = pd.read_csv("features_test.csv", index_col=0)

X_train = dataset.drop('file', axis='columns').drop('class', axis='columns')
y_train = dataset['class']

X_test = dataset_test.drop('file', axis='columns').drop('class', axis='columns')
y_test = dataset_test['class']

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score
score_acc1 = accuracy_score(y_test, y_pred)
score_prec1 = precision_score(y_test, y_pred)
score_rec1 = recall_score(y_test, y_pred)


from sklearn.neural_network import MLPClassifier
clf2 = MLPClassifier()
clf2.fit(X_train, y_train)

y_pred2 = clf2.predict(X_test)
score_acc2 = accuracy_score(y_test, y_pred2)
score_prec2 = precision_score(y_test, y_pred2)
score_rec2 = recall_score(y_test, y_pred2)

print("Decision tree:")
print("Accuracy:", score_acc1)
print("Precision:", score_prec1)
print("Recall:", score_rec1)

print("Neural network:")
print("Accuracy:", score_acc2)
print("Precision:", score_prec2)
print("Recall:", score_rec2)