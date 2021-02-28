from flask import Flask, Blueprint, request
from flask_cors import CORS

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np

class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Model layers (includes initialized model variables):
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(0.5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dense1 = nn.Linear(64 * 7 * 7, 10)

    def logits(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = F.relu(x)
        return self.dense1(x.reshape(-1, 64 * 7 * 7))
        

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

model = ConvolutionalNeuralNetworkModel()
model.load_state_dict(torch.load("savedmodel.pt"))

def get_pred(tensor):
  forwarded = model.f(tensor)
  return torch.argmax(forwarded).item()

server = Flask(__name__)
CORS(server)
api = Blueprint('api', __name__)

@server.route("/test")
def handle1():
    return "Hello world"

@server.route('/api/predict', methods=["POST"])
def handle2():
  if request.method == "POST":
    print("printing now")
    data = np.array(request.json)
    data = np.rot90(data)
    data = np.flipud(data)
    tensor = torch.tensor([data]).float()
    tensor = (tensor.unsqueeze(1) - 33.3184) / 78.5675
    guess = get_pred(tensor)

    return str(guess)

server.register_blueprint(api, url_prefix='/api')

if __name__ == "__main__":
    server.run(host='0.0.0.0', port=5000)
