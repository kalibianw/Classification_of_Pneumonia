import numpy as np
import cv2


nploader = np.load("APP/Pnemonia_augmentation_all_(400,500).npz")

for key in nploader:
    print(key)

x_data, label = nploader["x_data"], nploader["label"]

cv2.imshow("test", x_data[0, :, :])
cv2.waitKey(0)
cv2.destroyAllWindows()
