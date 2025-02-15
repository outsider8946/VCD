import requests
from collections import defaultdict

class Sender:
    '''Класс для отправки и обработки запросов на сервер'''
    def __init__(self):
        self.url = 'http://127.0.0.1:8000/predict' 
    
    def get_output(self, files, data):
        output = requests.post(url=self.url, files=files, data=data)
        if output.status_code == 200:
            return defaultdict(list, output.json())
        return defaultdict(list) 