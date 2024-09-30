import torch
import os
import gdown
from model import Unet
from torchvision.transforms import ToTensor, ToPILImage

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

    def _preporcess(self, img):
        tensor = ToTensor()(img)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        return tensor


    def make_binary(self, thresh):
        bin_tensor = (self.norm_tensor >= thresh).float()
        img = ToPILImage()(bin_tensor)

        return img

    def _postporcess(self, tensor, thresh):
        tensor = tensor[0]
        min_val = tensor.min()
        max_val = tensor.max()
        self.norm_tensor = (tensor - min_val) / (max_val - min_val)
        img = self.make_binary(thresh)

        return img

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Available device: {self.device}')
        self._init_model()

    def seg(self, img):
        tensor = self._preporcess(img)
        out = self.model(tensor)
        mask = self._postporcess(out, 0.8)

        return mask
