import cv2
import numpy as np


def segment_coins(image_path):
    # Завантаження зображення
    image = cv2.imread(image_path)
    original = image.copy()

    # Перетворення зображення в відтінки сірого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Застосування порогового перетворення для виділення монет
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Видалення шуму за допомогою операцій морфології
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Знаходження впевненої області переднього плану
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Знаходження невідомої області
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Маркування міток та використання алгоритму вододілу
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)

    # Виділення монет різними кольорами
    for coin_label in np.unique(markers):
        if coin_label == 0:
            continue  # Пропустити фон

        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[markers == coin_label] = 255

        # Виділити контури монет
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Забарвити монету в рандомний колір
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

        # Зафарбувати контур монети на оригінальному зображенні
        cv2.drawContours(original, contours, -1, color, 2)

    # Показати результат
    cv2.imshow('Segmented Coins', original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Запустити сегментацію монет на зображенні
segment_coins('coins_2.JPG')
