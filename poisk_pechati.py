import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
src = cv2.imread(file_path, cv2.IMREAD_UNCHANGED )

file_path_on_real = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
src_on_real = cv2.imread(file_path_on_real, cv2.IMREAD_UNCHANGED )

h, w = src.shape

r_min = int(w * 0.15 / 2)
r_max = int(w * 0.30 / 2)

contours = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, dp=1, minDist=2 * r_min)

boxes = []

output = src.copy()
output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
output2 = src_on_real

if contours is not None:
    for contour in contours[0]:
        xc, yc, r = np.uint16(np.around(contour))
        dop = 3
        if r_min <= r <= r_max:
            x1 = xc - r - dop
            y1 = yc - r - dop
            x2 = xc + r + dop
            y2 = yc + r + dop
            boxes.append([x1, y1, x2, y2])

for box in boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(output2, (x1, y1), (x2, y2), (0, 255, 0), 2)

save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
cv2.imwrite(save_path, output)


save_path2 = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
cv2.imwrite(save_path2, output2)