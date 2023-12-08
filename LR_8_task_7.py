import cv2
import numpy as np


def segment_coins(image_path):

    image = cv2.imread(image_path)
    original = image.copy()


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)


    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)


    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)


    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)


    for coin_label in np.unique(markers):
        if coin_label == 0:
            continue

        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[markers == coin_label] = 255


        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))


        cv2.drawContours(original, contours, -1, color, 2)


    cv2.imshow('Segmented Coins', original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



segment_coins('coins_2.JPG')
