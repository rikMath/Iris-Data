from sklearn import datasets
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

iris = datasets.load_iris() # Import dados
x, y = iris["data"], iris["target"]

"""Cases de treino e teste"""
index_permutation = np.random.permutation(150)
x, y = x[index_permutation], y[index_permutation]
x_train, x_test, y_train, y_test = x[:110], x[110:], y[:110], y[110:]


# Classificador binário para 0 - iris setosa
y_train_setosa = (y_train == 0)
y_test_setosa = (y_test == 0)


#SGD
sgd = SGDClassifier()
sgd.fit(x_train, y_train_setosa)
print(f'Valores para SGD usando cross_val_score {cross_val_score(sgd, x_train, y_train_setosa, cv=3, scoring="accuracy")}')
y_train_predict = cross_val_predict(sgd, x_train, y_train_setosa)

print(f'Confusion Matrix:\n{confusion_matrix(y_train_setosa, y_train_predict)}')
print(f'Precisão {precision_score(y_train_setosa, y_train_predict)}')
print(f'Recall {recall_score(y_train_setosa, y_train_predict)}')
print(f'f1_score {f1_score(y_train_setosa, y_train_predict)}')


#Classificadores para multiplas intancias

#SGD
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
print(f'\n\nValores para SGD usando cross_val_score com oVa {cross_val_score(sgd, x_train, y_train, cv=3, scoring="accuracy")}')
y_train_predict = cross_val_predict(sgd, x_train, y_train)

print(f'Confusion Matrix:\n{confusion_matrix(y_train, y_train_predict)}')

#Tree
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
print(f'\n\nValores para Decision Tree usando cross_val_score {cross_val_score(tree, x_train, y_train, cv=3, scoring="accuracy")}')
y_train_predict = cross_val_predict(tree, x_train, y_train)

print(f'Confusion Matrix:\n{confusion_matrix(y_train, y_train_predict)}')