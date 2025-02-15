import requests
import cv2
import numpy as np

url = 'http://127.0.0.1:8000/points'
PATH2IMG = "C:\\Users\\abram\\Downloads\\2025-01-31\\2025-01-31\\86.png"
files = {'img_file': open(PATH2IMG,'rb')}
data =  {'rectangle_points':[0,1,2,3]}
resp = requests.post(url=url, files=files, data=data) 
print(resp.status_code)
out = resp.json()
points = np.array(out['points'])

img2paint = cv2.resize(cv2.imread(PATH2IMG, cv2.IMREAD_COLOR),(512,512))

for p in points:
    img2paint = cv2.circle(img2paint,p,0,(0,255,0),1)

cv2.imshow('asd',img2paint)
cv2.waitKey(0)