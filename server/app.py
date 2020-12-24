from flask import Flask, request
from flask_cors import CORS

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
  return torch.argmax(model.f(tensor))

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=["POST"])
def handle():
  print("printing now")
  data = np.array(request.json)
  # data = data[::8, ::8]
  data = np.rot90(data)
  data = np.flipud(data)
  tensor = torch.tensor([data]).float()
  tensor = (tensor.unsqueeze(1) - 33.3184) / 78.5675
  guess = get_pred(tensor)

  data = tensor.reshape(28, 28).numpy()
  #Rescale to 0-255 and convert to uint8
  rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
  im = Image.fromarray(rescaled)  
  im.save('test.png')
  return str(guess.item())

if __name__ == "__main__":
  app.debug = True
  app.run(host='0.0.0.0' , port="3001", threaded=True)
