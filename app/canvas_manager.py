import tkinter as tk

class CanvasManager:
    def __init__(self, root, tk_image):
        self.root = root
        self.tk_image = tk_image

        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None
        self.clicks = 0

        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.update_canvas()

    def update_canvas(self):
        if self.tk_image:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def draw_rectangle(self, start_x, start_y, end_x, end_y):
        self.canvas.create_rectangle(start_x, start_y, end_x, end_y, outline="green", width=1)

    def draw_points(self, points, colour):
        for x, y in points:
            self.canvas.create_oval(x, y, x, y, fill=colour, outline=colour)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.update_canvas()

    def button_press(self, x, y):
        self.clicks += 1

        if self.clicks == 1:
            self.x1 = x
            self.y1 = y

            return None

        elif self.clicks == 2:
            self.x2 = x
            self.y2 = y
            self.clicks = 0

            return [self.x1, self.y1, self.x2, self.y2]
