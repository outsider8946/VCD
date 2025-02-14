import cv2
import torch
import numpy as np
from model import Unet
from typing import List
from torchvision.transforms import v2
from sklearn.cluster import SpectralClustering
from fastapi import FastAPI, File, UploadFile, HTTPException 

PATH2MODEL = "C:\\Users\\abram\\Downloads\\model.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def _preprocessing(content):
    '''Предоработка изображения для нейронной сети'''
    buffer = np.frombuffer(content,dtype=np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512,512))

    return v2.ToTensor()(img).unsqueeze(0).to(DEVICE)

def _load_model():
    '''Загрузка нейронной'''
    model = Unet()
    model.load_state_dict(torch.load(PATH2MODEL))
    model.to(DEVICE)
    model.eval()

    return model

print(f'loading model... available device {DEVICE}')
seg_model =_load_model()

cluster_model = SpectralClustering( #модель кластеризации
    n_clusters=2,
    affinity='nearest_neighbors',
    n_neighbors=3,
    n_jobs=-1,
    random_state=42
)

app = FastAPI()

def _get_points(content):
    '''Получение координат контуров с помощью ИИ и Canny'''
    tensor  = _preprocessing(content)
    out = seg_model(tensor)
    mask = out[0].cpu().detach().permute(1,2,0)
    mask_norm = (mask.numpy()*255).astype(np.uint8)
    edges = cv2.Canny(mask_norm, 127, 255)

    canny_points = []

    for i in range(len(edges)):
        for j in range(len(edges[i])):
            if edges[i,j] != 0:
                canny_points.append([j,i])

    return canny_points

def _get_points_in_rectangle(points, rect):
    '''Выбор точек внутри прямоугольника'''
    result = []
    x1,y1,x2,y2 = rect

    for x,y in points:
        if x1<=x<=x2 and y1<=y<=y2:
            result.append([x,y])
    
    return result

def _get_clusters(points):
    '''Кластеризация точек в прямоугольнике'''
    cluster1 = []
    cluster2 = []
    labels = cluster_model.fit_predict(points)

    for i in range(len(labels)):
        if labels[i] == 1:
            cluster1.append(points[i])
        else:
            cluster2.append(points[i])
        
    return cluster1, cluster2

@app.post("/predict")
def predict(img_file:UploadFile = File(...), rectangle_points: List[int] = []):
    try:
        points = _get_points(img_file.file.read())
        points_subset = _get_points_in_rectangle(points, rectangle_points)
        cluster1, cluster2 = _get_clusters(points_subset)

        return {'points': points, 'first_cluster': cluster1, 'second_cluster': cluster2}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"Hello": "World"}