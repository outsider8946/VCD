import torch
import cv2
import os
import gdown
from model import Unet
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

PATH2MODEL = 'resources/model.pth'
URL = 'https://drive.google.com/uc?export=download&id=1NkbKbaw2rzP5-1dwkJ0NLilqkblhKwD6'
SIZE = 512
class ai_worker():
    def _init_model(self):
        if not os.path.exists(PATH2MODEL):
            os.mkdir('resources')
            gdown.download(URL, PATH2MODEL, quiet=False)

        self.model = Unet()
        self.model.load_state_dict(torch.load(PATH2MODEL))
        self.model.to(self.device)
        self.model.eval()

    def _preporcess(self):
        self.img = ToTensor()(self.img)
        self.img = self.img.unsqueeze(0)
        self.img = self.img.to(self.device)

    def _postporcess(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        norm_tensor = (tensor - min_val) / (max_val - min_val)
        bin_tensor = (norm_tensor >= 0.5).float()

        return bin_tensor
    def __init__(self,path2img, name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.name = name
        self.img = cv2.resize(cv2.imread(path2img,cv2.IMREAD_GRAYSCALE),(SIZE,SIZE))
        self._init_model()
        self._preporcess()

    def seg(self):
        print('Available device: ', self.device)
        out = self.model(self.img)
        mask = self._postporcess(out)
        save_image(mask,f'output/{self.name}_unet.png')
