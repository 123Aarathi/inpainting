import cv2
from google.colab.patches import cv2_imshow

result=cv2.imread("/content/result/result.png", cv2.IMREAD_UNCHANGED)
cv2_imshow(result)
input=cv2.imread("/content/result/input.png", cv2.IMREAD_UNCHANGED)
cv2_imshow(input)
