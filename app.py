import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
from AI_worker import ai_worker
from Canny_worker import canny_worker

class VCDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VCD demo")
        self.root.attributes("-fullscreen", True)

        self.low_thresh_var = tk.DoubleVar(value=0)
        self.high_thresh_var = tk.DoubleVar(value=0)

        self.create_widgets()

        self.canny = canny_worker()
        self.u_net = ai_worker()

    def _run_canny(self, event):
        edges = self.canny.seg(self.img, self.low_thresh_var.get(), self.high_thresh_var.get())
        edges = Image.fromarray(edges)
        self._configure_img(edges)

    def _run_unet(self):
        mask = self.u_net.seg(self.img.convert('L'))
        self._configure_img(mask)

    def _preprocess_unet(self, event):
        bin_mask = self.u_net.make_binary(self.unet_thresh_var.get())
        self._configure_img(bin_mask)


    def _configure_img(self, img):
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk

    def _choose_img(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.img = Image.open(file_path)
            self.img = self.img.resize((800, 800), Image.Resampling.LANCZOS)
            self._configure_img(self.img)

    def _create_top_widgets(self):
        top_frame = tk.Frame(self.root, bg="lightgray", relief="raised", bd=2)
        top_frame.place(relx=1.0, rely=0.0, anchor="ne")

        minimize_button = tk.Button(top_frame, text="_", command=self.root.iconify, bg="lightgray", relief="flat")
        minimize_button.pack(side="left", padx=5)

        close_button = tk.Button(top_frame, text="X", command=self.root.destroy, bg="lightgray", relief="flat")
        close_button.pack(side="left", padx=5)

    def _create_left_widgets(self):

        left_frame = tk.Frame(self.root)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        tk.Frame(left_frame).pack(expand=True)

        self.low_thresh_var = tk.IntVar(value=25)
        self.high_thresh_var = tk.IntVar(value=35)
        self.unet_thresh_var = tk.DoubleVar(value=0.8)

        low_thresh_label = tk.Label(left_frame, text="Low Threshold")
        low_thresh_label.pack(pady=5, padx=40, anchor="w")
        low_thresh_slider = tk.Scale(left_frame, from_=0, to=255, variable=self.low_thresh_var,
                                     orient="horizontal", length=300, command=self._run_canny)
        low_thresh_slider.pack(pady=5, padx=30, anchor="w")


        high_thresh_label = tk.Label(left_frame, text="High Threshold")
        high_thresh_label.pack(pady=5, padx=40, anchor="w")
        high_thresh_slider = tk.Scale(left_frame, from_=0, to=255, variable=self.high_thresh_var,
                                      orient="horizontal", length=300, command=self._run_canny)
        high_thresh_slider.pack(pady=5, padx=30, anchor="w")

        tk.Frame(left_frame).pack(expand=True)

        unet_thresh_label = tk.Label(left_frame, text="Threshold (Unet)")
        unet_thresh_label.pack(pady=5, padx=40, anchor="w")
        unet_thresh_slider = tk.Scale(left_frame, from_=0, to=1, variable=self.unet_thresh_var,
                                  resolution=0.01, orient="horizontal", length=300, command=self._preprocess_unet)
        unet_thresh_slider.pack(pady=5, padx=30, anchor="w")

    def _create_bot_widgets(self):
        choose_img_button = tk.Button(self.root, text="Choose Image", command=self._choose_img)
        choose_img_button.grid(row=3, column=0, padx=10, pady=10)

        unet_button = tk.Button(self.root, text="U-net", command=self._run_unet)
        unet_button.grid(row=3, column=1, padx=10, pady=10)

    def _create_right_widgets(self):
        right_frame = tk.Frame(self.root)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=30, pady=30)

        tk.Frame(right_frame).pack(expand=True)

        self.img_label = tk.Label(right_frame, text="Image will appear here")
        self.img_label.pack()

        tk.Frame(right_frame).pack(expand=True)

    def _configure_widgets(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_rowconfigure(0, weight=1)

    def create_widgets(self):
        self._create_top_widgets()
        self._create_left_widgets()
        self._create_right_widgets()
        self._create_bot_widgets()
        self._configure_widgets()


if __name__ == "__main__":
    root = tk.Tk()
    app = VCDApp(root)
    root.mainloop()