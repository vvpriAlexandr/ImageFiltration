import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# opencv при загрузке в bgr конвертит. PIL работает с rgb.
class proektik:
    def __init__(self, root):
        self.root = root
        self.root.title("Обработка изображения")
        self.root.geometry("1200x800")

        self.show_color = tk.IntVar()
        self.original_image = None
        self.current_image = None
        self.image_path = None
        
        self.interface()
        
    def interface(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        parametrs_frame = ttk.Frame(main_frame, width=300)
        parametrs_frame.pack(side=tk.RIGHT, fill=tk.Y)
        parametrs_frame.pack_propagate(False)
        
        load_frame = ttk.LabelFrame(parametrs_frame, text="Файл")
        load_frame.pack(fill=tk.X)
        
        ttk.Button(load_frame,
                    text="Загрузить изображение",
                    command=self.load_image).pack()
        
        ttk.Button(load_frame,
                    text="Сохранить результат", 
                    command=self.save_image).pack()
        
#        ttk.Button(load_frame, 
#                  text="Загрузить прошлую версию", 
#                   command=self.load_image).pack()
        

        f1_frame = ttk.LabelFrame(parametrs_frame)
        f1_frame.pack(fill=tk.X)
        

        ttk.Button(f1_frame, 
                   text="Бинаризация", 
                   command=self.binarisation).pack(fill=tk.X)
        
        ttk.Button(f1_frame, 
                   text="Выделить границы",
                   command=self.edges).pack(fill=tk.X)
        
        ttk.Button(f1_frame, 
                   text="Выделить углы", 
                   command=self.ugli).pack(fill=tk.X)
        
        ttk.Button(f1_frame, 
                   text="Морфологическое закрытие", 
                   command=self.mClose).pack(fill=tk.X)
        
        f2_frame = ttk.LabelFrame(parametrs_frame)
        f2_frame.pack(fill=tk.X)
        
        f3_frame = ttk.Frame(f2_frame)
        f3_frame.pack(fill=tk.X)
        


        ttk.Label(f3_frame, text="Яркость:").pack(side=tk.LEFT)
        self.brightness_entry = ttk.Entry(f3_frame, width=8)
        self.brightness_entry.pack(side=tk.RIGHT)
        self.brightness_entry.insert(0, "0")
        
        contrast_frame = ttk.Frame(f2_frame)
        contrast_frame.pack(fill=tk.X)
        
        ttk.Label(contrast_frame, text="Контрастность:").pack(side=tk.LEFT)
        self.contrast_entry = ttk.Entry(contrast_frame, width=8)
        self.contrast_entry.pack(side=tk.RIGHT)
        self.contrast_entry.insert(0, "1.0")
        

        ttk.Button(f2_frame, 
                   text="Применить яркость", 
                   command=self.brightness).pack(fill=tk.X)
        
        ttk.Button(f2_frame, 
                   text="Применить контраст", 
                   command=self.contrast).pack(fill=tk.X)

        ttk.Button(f2_frame, 
                   text="Повысить резкость", 
                   command=self.sharpen_image).pack(fill=tk.X)
        

        channels_frame = ttk.LabelFrame(parametrs_frame, text="Цветовые каналы")
        channels_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(channels_frame,
                        text = "Показать в цвете",
                        variable = self.show_color).pack() 

        ttk.Button(channels_frame, 
                   text="Красный канал", 
                   command=lambda: self.split_channels(2)).pack(fill=tk.X)
        
        ttk.Button(channels_frame, 
                   text="Зеленый канал", 
                   command=lambda: self.split_channels(1)).pack(fill=tk.X)
        
        ttk.Button(channels_frame, 
                   text="Синий канал", 
                   command=lambda: self.split_channels(0)).pack(fill=tk.X)
          
        noise_frame = ttk.LabelFrame(parametrs_frame, text="Убирание шумов")
        noise_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(noise_frame, 
                   text="Медианный фильтр", 
                   command=self.median_blur).pack(fill=tk.X)
        
        ttk.Button(noise_frame, 
                   text="Гауссовский фильтр", 
                   command=self.gaussian_blur).pack(fill=tk.X)
        
        colormodel_frame = ttk.LabelFrame(parametrs_frame, text="Цветовое пространство")
        colormodel_frame.pack(fill=tk.X)

        ttk.Button(colormodel_frame, 
                   text="В HSV формат", 
                   command=self.to_HSV).pack(fill=tk.X)
        
        ttk.Button(colormodel_frame, 
                   text="В grayscale формат", 
                   command=self.to_grayscale).pack(fill=tk.X)
        
        hsv_frame = ttk.LabelFrame(parametrs_frame, text="HSV-Маска")
        hsv_frame.pack(fill=tk.X)
        
        ttk.Label(hsv_frame, 
                  text="Нижний порог H S V:").pack(fill=tk.X)
        self.LowHSV_entry = ttk.Entry(hsv_frame)
        self.LowHSV_entry.pack(fill=tk.X)
        self.LowHSV_entry.insert(0, "0 0 0")

        ttk.Label(hsv_frame, 
                  text="Верхний порог H S V:").pack(fill=tk.X)
        self.HighHSV_entry = ttk.Entry(hsv_frame)
        self.HighHSV_entry.pack(fill=tk.X)
        self.HighHSV_entry.insert(0, "180 255 255")

        ttk.Button(hsv_frame, 
                   text="Применить маску", 
                   command=self.hsv_mask
                   ).pack(fill=tk.X)


#####################################################################################################################################

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        self.image_path = file_path
        self.original_image = cv2.imread(file_path)
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.display_image()

    def display_image(self):
        if self.current_image is not None:
            # для корректного отображения в интерфейсе
            if len(self.current_image.shape) == 2:
                display_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2RGB)
            else:
                display_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            display_pil = Image.fromarray(display_rgb)
            display_pil.thumbnail((600, 600))
            display_photo = ImageTk.PhotoImage(display_pil)
            
            self.image_label.configure(image=display_photo)
            self.image_label.image = display_photo
    
    # сохранение картинки
    def save_image(self):
        if self.current_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, self.current_image)

############################################################################################################################################
    # бинаризация
    def binarisation(self):
        if self.current_image is not None:
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image
                
            # метод Оцу
            _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.current_image = thresholded
            self.display_image()
    
    # выделить грани
    def edges(self):
        if self.current_image is not None:
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image
            
            # детектор кэнни
            edges = cv2.Canny(gray, 10, 100)
            self.current_image = edges
            self.display_image()
    
    # выделить углы
    def ugli(self):
        if self.current_image is not None:
            if len(self.current_image.shape) == 3: 
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image
                
            gray = np.float32(gray)
            
            corners = cv2.cornerHarris(gray, 2, 3, 0.04)
            corners = cv2.dilate(corners, None)
            
            if len(self.current_image.shape) == 3:
                result = self.current_image.copy()
            else:
                result = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2BGR)
                
            result[corners > 0.01 * corners.max()] = [0, 0, 255]
            
            self.current_image = result
            self.display_image()
    
    def mClose(self):
        if self.current_image is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(self.current_image, cv2.MORPH_CLOSE, kernel)

            self.current_image = closed
            self.display_image()

    # яркость
    def brightness(self):
        if self.current_image is not None:
            brightness = float(self.brightness_entry.get())

            after_b = cv2.convertScaleAbs(self.current_image, beta=brightness)
            self.current_image = after_b
            self.display_image()

    # контраст
    def contrast(self):
        if self.current_image is not None:
            contrast = float(self.contrast_entry.get())
        
            after_c = cv2.convertScaleAbs(self.current_image, alpha=contrast)
            self.current_image = after_c
            self.display_image()

    # резкость
    def sharpen_image(self):
        if self.current_image is not None:
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            sharpened = cv2.filter2D(self.current_image, -1, kernel)
            self.current_image = sharpened
            self.display_image()
    
    # На разные цветовые каналы
    def split_channels(self, channel):
        if self.current_image is not None:
            b,g,r = cv2.split(self.current_image)
            show_color = self.show_color.get()

            if show_color:
                if channel == 0:  # Синий
                    result = cv2.merge([b, np.zeros_like(g), np.zeros_like(r)])
                    #result = b
                elif channel == 1:  # Зеленый
                    result = cv2.merge([np.zeros_like(b), g, np.zeros_like(r)])
                    #result = g
                elif channel == 2:  # Красный
                    result = cv2.merge([np.zeros_like(b), np.zeros_like(g), r])
                    #result = r
            else:
                    if channel == 0:  # Синий
                        result = b
                    elif channel == 1:  # Зеленый
                        result = g
                    elif channel == 2:  # Красный
                        result = r
            
            self.current_image = result
            self.display_image()
    
    # борьба с шумами
    def median_blur(self):
        if self.current_image is not None:
            filtered = cv2.medianBlur(self.current_image, 3)
            self.current_image = filtered
            self.display_image()
    
    def gaussian_blur(self):
        if self.current_image is not None:
            filtered = cv2.GaussianBlur(self.current_image, (3, 3), 0)
            self.current_image = filtered
            self.display_image()

    # перевод в другие форматы
    def to_HSV(self):
        if self.current_image is not None:
            hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
            self.current_image = hsv
            self.display_image()

    def to_grayscale(self):
        if self.current_image is not None:
            grayscale = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            self.current_image = grayscale
            self.display_image()
    # маска для выделения цвета в хсв модели
    def hsv_mask(self):
        if self.current_image is not None:
            lowhsv = self.LowHSV_entry.get()
            lowH, lowS, lowV = [int(x) for x in lowhsv.split()]

            highhsv = self.HighHSV_entry.get()
            highH, highS, highV = [int(x) for x in highhsv.split()]

            lower = np.array([lowH, lowS, lowV])
            higher = np.array([highH, highS, highV])

            #hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(self.current_image, lower, higher)
            masked = cv2.bitwise_and(self.current_image, self.current_image, mask=mask)
            self.current_image = masked
            self.display_image()

def main():
    root = tk.Tk()
    app = proektik(root)
    root.mainloop()

if __name__ == "__main__":
    main()

