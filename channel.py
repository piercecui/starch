import cv2
import numpy as np

img = cv2.imread('crop_0.jpg',1)
blur=cv2.GaussianBlur(img,(7,7),0)
edge=cv2.Canny(blur,50,130)
b,g,r = cv2.split(img)
combination = np.stack((b, g, r, edge), 2)
cv2.imshow('edge',edge)
k=cv2.waitKey(0)
if k==27:
	cv2.destroyAllWindows()
cv2.imwrite('revised.png', combination)



#img = cv2.imread('', cv2.IMREAD_UNCHANGED)  #read 4-channel image