import cv2

SIZE = 512

class canny_worker():

    def _green_extractor(self, img):
        green = img.copy()
        green[:,:,0] = 0
        green[:,:,2] = 0

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
        lab = cv2.cvtColor(green,cv2.COLOR_BGR2Lab)
        lab_planes = list(cv2.split(lab))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        green_clahe = cv2.cvtColor(lab,cv2.COLOR_Lab2BGR)

        return green_clahe

    def __init__(self, path2img, name):
        self.name = name
        self.img = cv2.resize(cv2.imread(path2img, cv2.IMREAD_COLOR),(SIZE,SIZE))

    def seg(self):
        green = self._green_extractor(self.img)
        blurred = cv2.GaussianBlur(green,(5,5),1.4)
        edges = cv2.Canny(blurred[:,:,1],20,35)
        cv2.imwrite(f'output/{self.name}_canny.png',edges)

