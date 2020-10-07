import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from utilities import visualize_classifier

# Входной файл, содержащий данные
input_file = '/home/adwiz/PycharmProjects/NN/AI with samples on Python/Artificial-Intelligence-with-Python' \
             '/Chapter 02/code/data_multivar_nb.txt'

data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Создаем наивный байесовский классификатор

classifier = GaussianNB()

# Обучаем классификатор
classifier.fit(X, y)

# Прогнозирование значений для тренировочных данных

y_prediction = classifier.predict(X)
accuracy = 100.0 * (y == y_prediction).sum() / X.shape[0]
print('Accuracy of Naive Bayes classifier = ', round(accuracy, 2), '%')
visualize_classifier(classifier, X, y)

# Разбиваем на тренировочную и тестовую выборки

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_prediction = classifier_new.predict(X_test)

# Вычисляем качество классификатора

accuracy = 100.0 * (y_test == y_test_prediction).sum() / X_test.shape[0]
print('Accuracy of Naive Bayes classifier = ', round(accuracy, 2), '%')

# Визуализайия работы классификатора
visualize_classifier(classifier_new, X_test, y_test)

num_folds = 3

accuracy_values = cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds)
print('Accuracy: ' + str(round(100*accuracy_values.mean(), 2)), '%')

precision_values = cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds)
print('Precision: ' + str(round(100*precision_values.mean(), 2)), '%')

recall_values = cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds)
print('Recall: ' + str(round(100*recall_values.mean(), 2)), '%')

f1_values = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds)
print('F1: ' + str(round(100*f1_values.mean(), 2)), '%')
