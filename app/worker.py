import cv2
import torch
import numpy as np
from typing import List, Tuple
from unet_model import Unet
from omegaconf import OmegaConf
from torchvision.transforms import v2
from sklearn.cluster import SpectralClustering


class Worker():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = OmegaConf.load('app/config.yaml')
        self.seg_model = self._load_seg_model()
        self.cluster_model = self._load_cluster_model()


    def _load_seg_model(self):
        model = Unet()
        model.load_state_dict(torch.load('app/resources/model.pth', weights_only=False, map_location=self.device))
        model.to(self.device)
        
        return model.eval()
    

    def _load_cluster_model(self):
        return SpectralClustering(**self.config.cluster)


    def get_binary_mask(self, img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        tensor = v2.ToTensor()(img).unsqueeze(0).to(self.device)
        out = self.seg_model(tensor)
        mask = out[0].cpu().detach().permute(1,2,0)
        self.mask = (mask.numpy()*255).astype(np.uint8)
        return self.mask


    def get_contour_points(self) -> np.ndarray:
        edges = cv2.Canny(self.mask, 
                          threshold1=self.config.contour.threshold1,
                          threshold2=self.config.contour.threshold2)
        self.points = np.column_stack(np.where(edges > 0))
        return self.points
    

    def get_clusters(self, rect: List) -> Tuple[np.ndarray,np.ndarray]:
        x, y, w, h = rect
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        bounds = []

        for y, x in self.points:
            if x1<=x<=x2 and y1<=y<=y2:
                bounds.append([y,x])
        
        bounds = np.array(bounds)

        labels = self.cluster_model.fit_predict(bounds)
        self.cluster1 = bounds[labels==1]
        self.cluster2 = bounds[labels==0]

        return self.cluster1, self.cluster2