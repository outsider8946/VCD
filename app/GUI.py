import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from canvas_manager import CanvasManager
from sender import Sender

class GUI:
    '''Класс для взаимодействия пользователя с алгоритмом'''
    def __init__(self, root):
        self.root = root
        self.load_image()
        self.canvas_manager = CanvasManager(root, self.tk_image)
        self.sender = Sender()

        self.canvas_manager.canvas.bind("<ButtonPress-1>", self.on_button_press)

    def load_image(self):
        '''Загрузка изображения'''
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])

        if not self.file_path:
            print("Изображение не выбрано.")
            self.root.destroy()
            return

        self.image = Image.open(self.file_path)
        self.image = self.image.resize((512,512))
        self.tk_image = ImageTk.PhotoImage(self.image)
 
        self.root.geometry(f"{self.image.width}x{self.image.height}")

    def on_button_press(self, event):
        '''Событие обработки нажатия мыши'''
        rectangle = self.canvas_manager.button_press(event.x, event.y)
        
        if isinstance(rectangle, list):
            x1, y1, x2, y2 = rectangle
            self.canvas_manager.draw_rectangle(x1, y1, x2, y2)
            
            files = {'img_file': open(self.file_path,'rb')}
            data = {'rectangle_points': rectangle}
            output_dict = self.sender.get_output(files, data)

            points = output_dict['points']
            cluster1 = output_dict['first_cluster']
            cluster2 = output_dict['second_cluster']
            
            self.canvas_manager.draw_points(points=points, colour='red')
            self.canvas_manager.draw_points(points=cluster1, colour='yellow')
            self.canvas_manager.draw_points(points=cluster2, colour='blue')

    def run(self):
        self.canvas_manager.update_canvas()
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    app.run()