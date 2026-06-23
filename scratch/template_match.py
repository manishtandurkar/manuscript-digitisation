import cv2
import numpy as np

target_path = r"C:\Users\nanda_4h6zihz\.gemini\antigravity-ide\brain\9bb206b2-f39c-4a1f-befa-83279631baab\media__1782219053166.png"
img = cv2.imread(target_path)

# Let's crop the main region (0 to 436 height, 0 to 509 width)
crop1 = img[0:436, 0:509]
cv2.imwrite("scratch/crop1.png", crop1)
print(f"Saved crop1 with shape {crop1.shape}")

# Let's count black and white pixels in crop1
gray_crop1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
total = gray_crop1.size
black = np.sum(gray_crop1 < 50)
white = np.sum(gray_crop1 > 200)
other = total - black - white
print(f"crop1: black={black} ({black/total*100:.2f}%), white={white} ({white/total*100:.2f}%), other={other} ({other/total*100:.2f}%)")
