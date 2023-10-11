import numpy as np
from sklearn.datasets import load_iris

iris_dataset = load_iris()

print("Ключі iris_dataset: \n{}\n".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Назви відповідей: {}".format(iris_dataset['target_names']))
print("Назва ознак: \n{}".format(iris_dataset['feature_names']))
print("Тип масиву data: {}".format(type(iris_dataset['data'])))
print("Форма масиву data: {}".format(iris_dataset['data'].shape))
print("Тип масиву target:{}".format(type(iris_dataset['target'])))
print("Відповіді:\n{}\n".format(iris_dataset['target']))

print(format(iris_dataset['feature_names']))
for i in range(5):
    print('{}'.format(i+1) + ") " + "{}".format(iris_dataset['data'][i]))
