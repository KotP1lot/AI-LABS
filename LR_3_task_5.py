import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# Побудова поліноміальних ознак (квадратичних) для X
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X.reshape(-1, 1))

# Виведення значень ознак X[0] та X_poly на екран
print("Значення ознак X[0]:", X[0])
print("Значення ознак після перетворення:", X_poly[0])

# Підгонка лінійної моделі до розширених даних
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Виведення значень коефіцієнтів полінома
intercept = lin_reg.intercept_
coefficients = lin_reg.coef_
print("Значення intercept:", intercept)
print("Значення коефіцієнтів:", coefficients)

# Генерація значень для побудови графіку
X_plot = np.linspace(0, 6, 100)
y_plot = lin_reg.predict(poly_features.transform(X_plot.reshape(-1, 1)))

# Побудова графіку
plt.scatter(X, y, label='Дані')
plt.plot(X_plot, y_plot, label='Поліноміальна регресія', color='red')
plt.xlabel('X')
plt.ylabel('у')
plt.legend()
plt.show()