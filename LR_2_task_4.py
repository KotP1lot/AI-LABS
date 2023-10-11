import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

input_file = 'income_data.txt'
dataset = pd.read_csv(input_file, sep=',', header=None, names=[
    'Age', 'Workclass', 'fnlwgt', 'Education', 'Education_Num', 'Marital_Status',
    'Occupation', 'Relationship', 'Race', 'Sex', 'Capital_Gain', 'Capital_Loss',
    'Hours_Per_Week', 'Native_Country', 'Income'
])

dataset_encoded = pd.get_dummies(dataset, columns=[
    'Workclass', 'Education', 'Marital_Status',
    'Occupation', 'Relationship', 'Race', 'Sex', 'Native_Country'
])

X = dataset_encoded.drop('Income', axis=1)
y = dataset_encoded['Income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='scale')))

results = []
names = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
