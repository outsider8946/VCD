import requests
from collections import defaultdict

class Sender:
    def __init__(self):
        self.url = 'http://127.0.0.1:8000/predict' 
    
    def get_output(self, files, data):
        output = requests.post(url=self.url, files=files, data=data)
        if output.status_code == 200:
            return defaultdict(list, output)
        return defaultdict(list) 