import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk


class proektik:
    def __init__(self, root):
        self.root = root
        self.root.title("Фильтрация изображения")
        self.root.geometry("1000x700")
        
        self.original_image = None
        self.current_image = None
        self.image_path = None
        
        self.interface()
        
    def interface(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        parametrs_frame = ttk.Frame(main_frame, width=300)
        parametrs_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        parametrs_frame.pack_propagate(False)
        
        load_frame = ttk.LabelFrame(parametrs_frame, text="Файл")
        load_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(load_frame, text="Загрузить изображение", 
                  command=self.load_image).pack(padx=5, pady=5)
        
        ttk.Button(load_frame, text="Сохранить результат", 
                  command=self.save_image).pack(padx=5, pady=5)
        
#        ttk.Button(load_frame, text="Загрузить прошлую версию", 
#                 command=self.load_image).pack(padx=5, pady=5)
        

        f1_frame = ttk.LabelFrame(parametrs_frame)
        f1_frame.pack(fill=tk.X, pady=5)
        

        ttk.Button(f1_frame, text="Бинаризация",
                  command=self.to_gray).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(f1_frame, text="Выделить границы",
                  command=self.edges).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(f1_frame, text="Выделить углы",
                  command=self.ugli).pack(fill=tk.X, padx=5, pady=2)
        
        f2_frame = ttk.LabelFrame(parametrs_frame)
        f2_frame.pack(fill=tk.X, pady=5)
        
        f3_frame = ttk.Frame(f2_frame)
        f3_frame.pack(fill=tk.X, padx=5, pady=2)
        


        ttk.Label(f3_frame, text="Яркость:").pack(side=tk.LEFT)
        self.brightness_entry = ttk.Entry(f3_frame, width=8)
        self.brightness_entry.pack(side=tk.RIGHT, padx=5)
        self.brightness_entry.insert(0, "0")
        
        contrast_frame = ttk.Frame(f2_frame)
        contrast_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(contrast_frame, text="Контрастность:").pack(side=tk.LEFT)
        self.contrast_entry = ttk.Entry(contrast_frame, width=8)
        self.contrast_entry.pack(side=tk.RIGHT, padx=5)
        self.contrast_entry.insert(0, "1.0")
        

        ttk.Button(f2_frame, text="Применить яркость",
                command=self.brightness).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(f2_frame, text="Применить контраст",
                command=self.contrast).pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(f2_frame, text="Повысить резкость",
                command=self.sharpen_image).pack(fill=tk.X, padx=5, pady=2)
        

        channels_frame = ttk.LabelFrame(parametrs_frame, text="Цветовые каналы")
        channels_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(channels_frame, text="Красный канал",
                  command=lambda: self.split_channels(2)).pack(fill=tk.X, padx=5, pady=1)
        
        ttk.Button(channels_frame, text="Зеленый канал",
                  command=lambda: self.split_channels(1)).pack(fill=tk.X, padx=5, pady=1)
        
        ttk.Button(channels_frame, text="Синий канал",
                  command=lambda: self.split_channels(0)).pack(fill=tk.X, padx=5, pady=1)
          
        noise_frame = ttk.LabelFrame(parametrs_frame, text="Убирание шумов")
        noise_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(noise_frame, text="Медианный фильтр",
                  command=self.median_blur).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(noise_frame, text="Размытие по Гауссу",
                  command=self.gaussian_blur).pack(fill=tk.X, padx=5, pady=2)
#####################################################################################################################################

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.current_image = self.original_image.copy()
                self.display_image()
            else:
                messagebox.showerror("Ошибка", "Не удалось загрузить изображение!")

    def display_image(self):
        if self.current_image is not None:
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
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            if file_path:
                cv2.imwrite(file_path, self.current_image)

##################################################################################################
    # бинаризация
    def to_gray(self):
        if self.current_image is not None:
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image
                
            # метод Оцу
            _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
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
    
    # яркость
    def brightness(self):
        brightness = float(self.brightness_entry.get())

        after_b = cv2.convertScaleAbs(self.current_image, beta=brightness)
        self.current_image = after_b
        self.display_image()

    # контраст
    def contrast(self):
        contrast = float(self.contrast_entry.get())
        
        after_c = cv2.convertScaleAbs(self.current_image, alpha=contrast)
        self.current_image = after_c
        self.display_image()

    # резкость
    def sharpen_image(self):
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(self.current_image, -1, kernel)
        self.current_image = sharpened
        self.display_image()
    
    # На разные цветовые каналы
    def split_channels(self, channel):
        if self.current_image is not None:
            zeros = np.zeros_like(self.current_image[:, :, 0])
            
            if channel == 0:  # Синий
                result = cv2.merge([self.current_image[:, :, 0], zeros, zeros])
            elif channel == 1:  # Зеленый
                result = cv2.merge([zeros, self.current_image[:, :, 1], zeros])
            elif channel == 2:  # Красный
                result = cv2.merge([zeros, zeros, self.current_image[:, :, 2]])
            
            self.current_image = result
            self.display_image()
    
    # борьба с шумами
    def median_blur(self):
        filtered = cv2.medianBlur(self.current_image, 5)
        self.current_image = filtered
        self.display_image()
    
    def gaussian_blur(self):
        filtered = cv2.GaussianBlur(self.current_image, (5, 5), 0)
        self.current_image = filtered
        self.display_image()

def main():
    root = tk.Tk()
    app = proektik(root)
    root.mainloop()

if __name__ == "__main__":
    main()