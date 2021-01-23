import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.metrics import *
from time import time

col_names = ['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 'att9', 'att10', 'att11', 'att12', 'att13',
             'att14', 'att15', 'att16', 'label']
# load dataset
DATA = pd.read_csv("Final_Data.csv", header=None, names=col_names)

# DATA.info()
feature_cols = ['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 'att9', 'att10', 'att11', 'att12',
                'att13', 'att14', 'att15', 'att16']

X = DATA[feature_cols]  # Features
y = DATA.label  # Target variable

accuracy = 0
Confusion_Matrix = 0
f1 = 0
precision = 0
recall = 0
timee = 0

for x in range(10):
    test_start = time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=x)  # 80% training and 20% test

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy")

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    accuracy = accuracy + accuracy_score(y_test, y_pred)
    f1 = f1 + f1_score(y_test, y_pred, average='weighted')
    precision = precision + precision_score(y_test, y_pred, average='weighted')
    recall = recall + recall_score(y_test, y_pred, average='weighted')
    Confusion_Matrix = Confusion_Matrix + confusion_matrix(y_test, y_pred)
    test_finish = time()
    timee = timee + test_finish - test_start

print("Overall accuracy for decision tree with gain ratio and hold out method(10 times) :", accuracy / 10)
print("Overall f1_score for decision tree with gain ratio and hold out method(10 times) :", f1 / 10)
print("Overall precision_score for decision tree with gain ratio and hold out method(10 times) :", precision / 10)
print("Overall recall_score for decision tree with gain ratio and hold out method(10 times) :", recall / 10)

print("       Confusion Matrix : ")
print(Confusion_Matrix / 10)

print("Overall time needed for decision tree with gain ratio and hold out method(10 times) :",
      ((timee / 10) * 10 * 10 * 10), " miliseconds")

scoring = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']

# Decision tree - Gini Ratio - kfold

for i in range(10):
  test_start = time()
  kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state = i )
  model = DecisionTreeClassifier(criterion="entropy")

  results = model_selection.cross_validate(model, X, y, cv=kfold, scoring=scoring)
  accuracy = accuracy + results['test_accuracy'].mean()
  f1 = f1 + results['test_f1_weighted'].mean()
  precision = precision + results['test_precision_weighted'].mean()
  recall = recall + results['test_recall_weighted'].mean()
  test_finish = time()
  timee = timee + test_finish-test_start

  y_pred = cross_val_predict(model, X, y, cv=kfold)
  Confusion_Matrix = Confusion_Matrix + confusion_matrix(y, y_pred)

print("Overall accuracy for decision tree with gain ratio and K Fold cross valdiation(10 times) :", accuracy/10)
print("Overall f1_score for decision tree with gain ratio and K Fold cross valdiation(10 times) :", f1/10)
print("Overall precision_score for decision tree with gain ratio and K Fold cross valdiation(10 times) :", precision/10)
print("Overall recall_score for decision tree with gain ratio and K Fold cross valdiation(10 times) :", recall/10)

print("       Confusion Matrix : ")
print(Confusion_Matrix/10)

print("Overall time needed for decision tree with gain ratio and K Fold cross valdiation(10 times) :", ((timee/10)*10*10*10), " miliseconds")

# Decision tree - Gini Ratio - bagging

for i in range(10):
  test_start = time()
  kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=i)
  dt = DecisionTreeClassifier(criterion="entropy")
  model = BaggingClassifier(base_estimator=dt, random_state=i)

  results = model_selection.cross_validate(model, X, y, cv=kfold, scoring=scoring)

  accuracy = accuracy + results['test_accuracy'].mean()
  f1 = f1 + results['test_f1_weighted'].mean()
  precision = precision + results['test_precision_weighted'].mean()
  recall = recall + results['test_recall_weighted'].mean()
  test_finish = time()
  timee = timee + test_finish-test_start

  y_pred = cross_val_predict(model, X, y, cv=kfold)
  Confusion_Matrix = Confusion_Matrix + confusion_matrix(y, y_pred)


print("Overall accuracy for decision tree with gain ratio and Bagging with K-Fold Cross Validation :", accuracy/10)
print("Overall f1_score for decision tree with gain ratio and Bagging with K-Fold Cross Validation :", f1/10)
print("Overall precision_score for decision tree with gain ratio and Bagging with K-Fold Cross Validation :", precision/10)
print("Overall recall_score for decision tree with gain ratio and Bagging with K-Fold Cross Validation :", recall/10)

print("       Confusion Matrix : ")
print(Confusion_Matrix/10)

print("Overall time needed for decision tree with gain ratio and Bagging with K-Fold Cross Validation :", ((timee/10)*10*10*10), " miliseconds")

# Decision tree - Gini Ratio - boosting

for i in range(10):
    test_start = time()
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=i)
    dt = DecisionTreeClassifier(criterion="entropy")
    model = AdaBoostClassifier(base_estimator=dt)

    results = model_selection.cross_validate(model, X, y, cv=kfold, scoring=scoring)
    test_finish = time()

    accuracy = accuracy + results['test_accuracy'].mean()
    f1 = f1 + results['test_f1_weighted'].mean()
    precision = precision + results['test_precision_weighted'].mean()
    recall = recall + results['test_recall_weighted'].mean()
    timee = timee + test_finish - test_start

    y_pred = cross_val_predict(model, X, y, cv=kfold)
    Confusion_Matrix = Confusion_Matrix + confusion_matrix(y, y_pred)

print("Overall accuracy for decision tree with Boosting with K-Fold Cross Validation :", accuracy / 10)
print("Overall f1_score for decision tree with Boosting with K-Fold Cross Validation :", f1 / 10)
print("Overall precision_score for decision tree with Boosting with K-Fold Cross Validation :", precision / 10)
print("Overall recall_score for decision tree with Boosting with K-Fold Cross Validation :", recall / 10)

print("       Confusion Matrix : ")
print(Confusion_Matrix / 10)

print("Overall time needed for decision tree with Boosting with K-Fold Cross Validation :",
      ((timee / 10) * 10 * 10 * 10), " miliseconds")


# Decision tree - Gini Index - holdout

for x in range(10):
    test_start = time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=x)  # 80% training and 20% test

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="gini")

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    accuracy = accuracy + metrics.accuracy_score(y_test, y_pred)
    f1 = f1 + f1_score(y_test, y_pred, average='weighted')
    precision = precision + precision_score(y_test, y_pred, average='weighted')
    recall = recall + recall_score(y_test, y_pred, average='weighted')
    Confusion_Matrix = Confusion_Matrix + confusion_matrix(y_test, y_pred)
    test_finish = time()
    timee = timee + test_finish - test_start

print("Overall accuracy for decision tree with gini index and hold out method(10 times) :", accuracy / 10)
print("Overall f1_score for decision tree with gini index and hold out method(10 times) :", f1 / 10)
print("Overall precision_score for decision tree with gini index and hold out method(10 times) :", precision / 10)
print("Overall recall_score for decision tree with gini index and hold out method(10 times) :", recall / 10)

print("       Confusion Matrix : ")
print(Confusion_Matrix / 10)

print("Overall time needed for decision tree with gini index and hold out method(10 times) :",
      ((timee / 10) * 10 * 10 * 10), " miliseconds")

# Decision tree - Gini Index - kfold

for i in range(10):
  test_start = time()
  kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state = i )
  model = DecisionTreeClassifier(criterion="gini")

  results = model_selection.cross_validate(model, X, y, cv=kfold, scoring=scoring)
  accuracy = accuracy + results['test_accuracy'].mean()
  f1 = f1 + results['test_f1_weighted'].mean()
  precision = precision + results['test_precision_weighted'].mean()
  recall = recall + results['test_recall_weighted'].mean()
  test_finish = time()
  timee = timee + test_finish-test_start

  y_pred = cross_val_predict(model, X, y, cv=kfold)
  Confusion_Matrix = Confusion_Matrix + confusion_matrix(y, y_pred)

print("Overall accuracy for decision tree with gini index and K Fold cross valdiation(10 times) :", accuracy/10)
print("Overall f1_score for decision tree with gini index and K Fold cross valdiation(10 times) :", f1/10)
print("Overall precision_score for decision tree with gini index and K Fold cross valdiation(10 times) :", precision/10)
print("Overall recall_score for decision tree with gini index and K Fold cross valdiation(10 times) :", recall/10)

print("       Confusion Matrix : ")
print(Confusion_Matrix/10)

print("Overall time needed for decision tree with gini index and K Fold cross valdiation(10 times) :", ((timee/10)*10*10*10), " miliseconds")

# Decision tree - Gini Index - Bagging

for i in range(10):
  test_start = time()
  kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=i)
  dt = DecisionTreeClassifier(criterion="gini")
  model = BaggingClassifier(base_estimator=dt, random_state=i)

  results = model_selection.cross_validate(model, X, y, cv=kfold, scoring=scoring)
  test_finish = time()

  accuracy = accuracy + results['test_accuracy'].mean()
  f1 = f1 + results['test_f1_weighted'].mean()
  precision = precision + results['test_precision_weighted'].mean()
  recall = recall + results['test_recall_weighted'].mean()
  timee = timee + test_finish-test_start

  y_pred = cross_val_predict(model, X, y, cv=kfold)
  Confusion_Matrix = Confusion_Matrix + confusion_matrix(y, y_pred)


print("Overall accuracy for decision tree with gain ratio and Bagging with K-Fold Cross Validation :", accuracy/10)
print("Overall f1_score for decision tree with gain ratio and Bagging with K-Fold Cross Validation :", f1/10)
print("Overall precision_score for decision tree with gain ratio and Bagging with K-Fold Cross Validation :", precision/10)
print("Overall recall_score for decision tree with gain ratio and Bagging with K-Fold Cross Validation :", recall/10)

print("       Confusion Matrix : ")
print(Confusion_Matrix/10)

print("Overall time needed for decision tree with gain ratio and Bagging with K-Fold Cross Validation :", ((timee/10)*10*10*10), " miliseconds")

# Decision tree - Gini Index - Boosting

for i in range(10):
    test_start = time()

    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=i)
    dt = DecisionTreeClassifier(criterion="gini")
    model = AdaBoostClassifier(base_estimator=dt, n_estimators=100)
    results = model_selection.cross_validate(model, X, y, cv=kfold, scoring=scoring)
    test_finish = time()

    accuracy = accuracy + results['test_accuracy'].mean()
    f1 = f1 + results['test_f1_weighted'].mean()
    precision = precision + results['test_precision_weighted'].mean()
    recall = recall + results['test_recall_weighted'].mean()

    timee = timee + test_finish - test_start

    y_pred = cross_val_predict(model, X, y, cv=kfold)
    Confusion_Matrix = Confusion_Matrix + confusion_matrix(y, y_pred)

print("Overall accuracy for decision tree with Boosting with K-Fold Cross Validation :", accuracy / 10)
print("Overall f1_score for decision tree with Boosting with K-Fold Cross Validation :", f1 / 10)
print("Overall precision_score for decision tree with Boosting with K-Fold Cross Validation :", precision / 10)
print("Overall recall_score for decision tree with Boosting with K-Fold Cross Validation :", recall / 10)

print("       Confusion Matrix : ")
print(Confusion_Matrix / 10)

print("Overall time needed for decision tree with Boosting with K-Fold Cross Validation :",
      ((timee / 10) * 10 * 10 * 10), " miliseconds")

# Naive Bayes - Holdout

for x in range(10):
    test_start = time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=x)  # 80% training and 20% test

    gnb = GaussianNB()
    gnb.fit(X_train, y_train.ravel())
    y_pred = gnb.predict(X_test)

    accuracy = accuracy + metrics.accuracy_score(y_test, y_pred)
    f1 = f1 + f1_score(y_test, y_pred, average='weighted')
    precision = precision + precision_score(y_test, y_pred, average='weighted')
    recall = recall + recall_score(y_test, y_pred, average='weighted')
    Confusion_Matrix = Confusion_Matrix + confusion_matrix(y_test, y_pred)
    test_finish = time()
    timee = timee + test_finish-test_start

print("Overall accuracy for Naive Bayes with hold out method :", accuracy/10)
print("Overall f1_score for Naive Bayes with hold out method :", f1/10)
print("Overall precision_score for Naive Bayes with hold out method :", precision/10)
print("Overall recall_score for Naive Bayes with hold out method :", recall/10)

print("       Confusion Matrix : ")
print(Confusion_Matrix/10)

print("Overall time needed for Naive Bayes with hold out method :", ((timee/10)*10*10*10), " miliseconds")

# Naive Bayes - kfold

for x in range (10):
    test_start = time()
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=x)
    gnb = GaussianNB()

    results = model_selection.cross_validate(gnb, X, y, cv=kfold, scoring=scoring)
    accuracy = accuracy + results['test_accuracy'].mean()
    f1 = f1 + results['test_f1_weighted'].mean()
    precision = precision + results['test_precision_weighted'].mean()
    recall = recall + results['test_recall_weighted'].mean()
    test_finish = time()
    timee = timee + test_finish - test_start

    y_pred = cross_val_predict(gnb, X, y, cv=kfold)
    Confusion_Matrix = Confusion_Matrix + confusion_matrix(y, y_pred)

print("Overall accuracy for Naive Bayes with k-fold cross validation :", accuracy / 10)
print("Overall f1_score for Naive Bayes with k-fold cross validation :", f1 / 10)
print("Overall precision_score for Naive Bayes with k-fold cross validation :", precision / 10)
print("Overall recall_score for Naive Bayes with k-fold cross validation :", recall / 10)

print("       Confusion Matrix : ")
print(Confusion_Matrix / 10)

print("Overall time needed for Naive Bayes with k-fold cross validation :", ((timee / 10) * 10 * 10 * 10), " miliseconds")

# Naive Bayes - Bagging

for x in range(10):
    test_start = time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=x)  # 80% training and 20% test
    gnb = GaussianNB()
    model = BaggingClassifier(base_estimator=gnb, random_state=x)

    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy + metrics.accuracy_score(y_test, y_pred)
    f1 = f1 + f1_score(y_test, y_pred, average='weighted')
    precision = precision + precision_score(y_test, y_pred, average='weighted')
    recall = recall + recall_score(y_test, y_pred, average='weighted')
    Confusion_Matrix = Confusion_Matrix + confusion_matrix(y_test, y_pred)
    test_finish = time()
    timee = timee + test_finish - test_start

print("Overall accuracy for Naive Bayes with Bagging Ensemble Method :", accuracy / 10)
print("Overall f1_score for Naive Bayes with Bagging Ensemble Method :", f1 / 10)
print("Overall precision_score for Naive Bayes with Bagging Ensemble Method :", precision / 10)
print("Overall recall_score for Naive Bayes with Bagging Ensemble Method :", recall / 10)

print("       Confusion Matrix : ")
print(Confusion_Matrix / 10)

print("Overall time needed for Naive Bayes with Bagging Ensemble Method :", ((timee / 10) * 10 * 10 * 10), " miliseconds")

# Naive Bayes - Boosting

test_start = time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training and 20% test

gnb = GaussianNB()
model = AdaBoostClassifier(base_estimator=gnb, random_state=0, algorithm='SAMME')
model = model.fit(X_train,y_train)

y_pred = model.predict(X_test)

test_finish = time()
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
Confusion_Matrix = confusion_matrix(y_test, y_pred)

timee = timee + test_finish-test_start

print("Overall accuracy for Naive Bayes with Boosting Ensemble Method :", accuracy)
print("Overall f1_score for Naive Bayes with Boosting Ensemble Method :", f1)
print("Overall precision_score for Naive Bayes with Boosting Ensemble Method :", precision)
print("Overall recall_score for Naive Bayes with Boosting Ensemble Method :", recall)

print("       Confusion Matrix : ")
print(Confusion_Matrix)

print("Overall time needed for Naive Bayes with Boosting Ensemble Method :", ((timee)*10*10*10), " miliseconds")

# SVM - Holdout

for x in range(10):
    test_start = time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=x)  # 80% training and 20% test

    # Create Decision Tree classifer object
    model = SVC()

    # Train Decision Tree Classifer
    model = model.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = model.predict(X_test)

    accuracy = accuracy + accuracy_score(y_test, y_pred)
    f1 = f1 + f1_score(y_test, y_pred, average='weighted')
    precision = precision + precision_score(y_test, y_pred, average='weighted')
    recall = recall + recall_score(y_test, y_pred, average='weighted')
    Confusion_Matrix = confusion_matrix(y_test, y_pred)
    test_finish = time()
    timee = timee + test_finish - test_start

print("Overall accuracy for SVM with hold out method(10 times) :", accuracy / 10)
print("Overall f1_score for SVM with hold out method(10 times) :", f1 / 10)
print("Overall precision_score for SVM with hold out method(10 times) :", precision / 10)
print("Overall recall_score for SVM with hold out method(10 times) :", recall / 10)

print("       Confusion Matrix : ")
print(Confusion_Matrix)

print("Overall time needed for SVM with hold out method(10 times) :", ((timee / 10) * 10 * 10 * 10), " miliseconds")

# SVM - kfold

test_start = time()
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state = 100 )
model = SVC()
results = model_selection.cross_validate(model, X, y, cv=kfold, scoring=scoring)
test_finish = time()

accuracy = results['test_accuracy'].mean()
f1 = results['test_f1_weighted'].mean()
precision = results['test_precision_weighted'].mean()
recall = results['test_recall_weighted'].mean()
timee = test_finish-test_start

y_pred = cross_val_predict(model, X, y, cv=kfold)
Confusion_Matrix = Confusion_Matrix + confusion_matrix(y, y_pred)

print("Accuracy for SVM with K-Fold cross validation:", accuracy)
print("F1_score for SVM with K-Fold cross validation:", f1)
print("Precision_score for SVM with K-Fold cross validation:", precision)
print("Recall_score for SVM with K-Fold cross validation:", recall)

print("       Confusion Matrix : ")
print(Confusion_Matrix)

print("Time needed for SVM with K-Fold cross validation:", ((timee)*10*10*10), " miliseconds")

# SVM - Bagging

test_start = time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10) # 80% training and 20% test

# Create Decision Tree classifer object
model = BaggingClassifier(base_estimator=SVC(), n_estimators=20, random_state=0)

# Train Decision Tree Classifer
model = model.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)

test_finish = time()
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
Confusion_Matrix = confusion_matrix(y_test, y_pred)
timee = test_finish-test_start

print("Accuracy for SVM with Bagging implemented with Hold Out method :", accuracy)
print("F1_score for SVM with Bagging implemented with Hold Out method :", f1)
print("Precision_score for SVM with Bagging implemented with Hold Out method :", precision)
print("Recall_score for SVM with Bagging implemented with Hold Out method :", recall)

print("       Confusion Matrix : ")
print(Confusion_Matrix)

print("Time needed for SVM with Bagging implemented with Hold Out method :", ((timee)*10*10*10), " miliseconds")

# SVM - Boosting

test_start = time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 10) # 80% training and 20% test

# Create Decision Tree classifer object
model = AdaBoostClassifier(base_estimator=SVC(probability=True), n_estimators=1, random_state= 100)

# Train Decision Tree Classifer
model = model.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)

test_finish = time()
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
Confusion_Matrix = confusion_matrix(y_test, y_pred)
timee = test_finish-test_start

print("Accuracy for SVM with Boosting implemented with Hold Out method :", accuracy)
print("F1_score for SVM with Boosting implemented with Hold Out method :", f1)
print("Precision_score for SVM with Boosting implemented with Hold Out method :", precision)
print("Recall_score for SVM with Boosting implemented with Hold Out method :", recall)

print("       Confusion Matrix : ")
print(Confusion_Matrix)

print("Time needed for SVM with Boosting implemented with Hold Out method :", ((timee)*10*10*10), " miliseconds")

#######  Single Hidden Layer ANN

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

dataset = loadtxt('labeled.csv', delimiter=',')

# split into input (X) and output (y) variables
X = dataset[:, 0:16]
y = dataset[:, 16]

# Define the keras model
# Default activator: linear
model = Sequential()
model.add(Dense(8, input_dim=16))
model.add(Dense(4))
model.add(Dense(1))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
# model.fit(X, y, epochs=100, batch_size=25)
model.fit(X, y, epochs=100, batch_size=32)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy * 100), '%')

# Single Hidden Layer ANN with Holdout

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier
from time import time

test_start = time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf = MLPClassifier(solver='adam', hidden_layer_sizes=(4,), max_iter=100, random_state=0)

clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

accuracy = accuracy + accuracy_score(y_test, y_pred)
f1 = f1 + f1_score(y_test, y_pred, average='weighted')
precision = precision + precision_score(y_test, y_pred, average='weighted')
recall = recall + recall_score(y_test, y_pred, average='weighted')
Confusion_Matrix = Confusion_Matrix + confusion_matrix(y_test, y_pred)
test_finish = time()
time = test_finish - test_start

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall Score:", recall)

print("       Confusion Matrix : ")
print(Confusion_Matrix)

print("Execution Time :", (time * 10 * 10 * 10)," milliseconds")

# Single Hidden Layer ANN with Holdout and Bagging

for x in range(1):
    test_start = time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=x)
    ann = MLPClassifier(solver='adam', hidden_layer_sizes=(4, 1), max_iter=100, random_state=0)
    model = BaggingClassifier(base_estimator=ann, n_estimators=10, random_state=1)
    model = model.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = model.predict(X_test)

    accuracy = accuracy + accuracy_score(y_test, y_pred)
    f1 = f1 + f1_score(y_test, y_pred, average='weighted')
    precision = precision + precision_score(y_test, y_pred, average='weighted')
    recall = recall + recall_score(y_test, y_pred, average='weighted')
    Confusion_Matrix = confusion_matrix(y_test, y_pred)
    test_finish = time()
    timee = timee + test_finish - test_start

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)

print("       Confusion Matrix : ")
print(Confusion_Matrix)

print("Exec Time:", ((timee) * 10 * 10 * 10), " milliseconds")

# Dual Hidden Layer ANN

dataset = loadtxt('labeled.csv', delimiter=',')

# split into input (X) and output (y) variables
X = dataset[:, 0:16]
y = dataset[:, 16]

# Define the keras model
# Default activator: linear
model = Sequential()
model.add(Dense(16, input_dim=16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
# model.fit(X, y, epochs=100, batch_size=25)
model.fit(X, y, epochs=100, batch_size=32)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy * 100))


# Dual Hidden Layer ANN with Holdout

test_start = time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf = MLPClassifier(solver='adam', hidden_layer_sizes=(8,4), max_iter=100, random_state=0)

clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

accuracy = accuracy + accuracy_score(y_test, y_pred)
f1 = f1 + f1_score(y_test, y_pred, average='weighted')
precision = precision + precision_score(y_test, y_pred, average='weighted')
recall = recall + recall_score(y_test, y_pred, average='weighted')
Confusion_Matrix = Confusion_Matrix + confusion_matrix(y_test, y_pred)
test_finish = time()
time = test_finish - test_start

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall Score:", recall)

print("       Confusion Matrix : ")
print(Confusion_Matrix)

print("Execution Time :", (time * 10 * 10 * 10)," milliseconds")


# Dual Hidden Layer ANN with Holdout and Bagging

for x in range(1):
    test_start = time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=x)
    ann = MLPClassifier(solver='adam', hidden_layer_sizes=(8, 4), max_iter=100, random_state=0)
    model = BaggingClassifier(base_estimator=ann, n_estimators=10, random_state=1)
    model = model.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = model.predict(X_test)

    accuracy = accuracy + accuracy_score(y_test, y_pred)
    f1 = f1 + f1_score(y_test, y_pred, average='weighted')
    precision = precision + precision_score(y_test, y_pred, average='weighted')
    recall = recall + recall_score(y_test, y_pred, average='weighted')
    Confusion_Matrix = confusion_matrix(y_test, y_pred)
    test_finish = time()
    timee = timee + test_finish - test_start

print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)

print("       Confusion Matrix : ")
print(Confusion_Matrix)

print("Exec Time:", ((timee) * 10 * 10 * 10), " milliseconds")






