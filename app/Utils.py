import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import torchvision
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from Models import ESPCN, SRGAN, Autoencoder, SAN, VDSR
import time

class GUIImageProcessor:
    def __init__(self, modelsDir):
        self.models = self.loadModels(modelsDir)

    def loadModels(self, modelsDir):
        models = {}
        for modelFile in os.listdir(modelsDir):
            if modelFile.endswith(".pth"):
                modelName = modelFile.split('.')[0]
                modelType = modelName.split('_')[1]
                match modelType:
                    case "espcn":
                        model = ESPCN(scale_factor=1)
                    case "srgan":
                        model = SRGAN()
                    case "autoencoder":
                        model = Autoencoder()
                    case "san":
                        model = SAN()
                    case "vdsr":
                        model = VDSR()
                    case _:
                        continue
                model.load_state_dict(torch.load(os.path.join(modelsDir, modelFile)))
                models[modelName] = model
        return models

    def processImage(self, imageInput: Image, modelName: str):
        model = self.models.get(modelName)
        if model is None:
            raise ValueError(f"No model named {modelName} is loaded.")
        
        transform = transforms.Compose([transforms.ToTensor()])
        model.eval()
        image = imageInput
        image = transform(image)
        with torch.no_grad():
            start = time.time()
            predictedOutput = model(image)
            end = time.time()
            print(f"Time taken: {end - start}")
        image = predictedOutput.detach().numpy()
        image = np.clip(image, 0, 1)
        image = np.transpose(image, (1, 2, 0))
        return image
    
    def visualizeTraining(self, modelName: str):
        fig = None
        file_processed = False
        colors = ['blue', 'orange', 'green']
        for fileName in os.listdir('../metrics'):
            if fileName.startswith(modelName) and fileName.endswith('.csv'):
                file_processed = True
                df = pd.read_csv(f'../metrics/{fileName}')

                metrics = ['mse', 'psnr', 'ssim']
                fig, axs = plt.subplots(1, len(metrics), figsize=(10 * len(metrics), 6))

                if 'kfold' in modelName:
                    for ax, metric, color in zip(axs, metrics, colors):
                        ax.plot(df[f'avg_{metric}'], label=f'avg_{metric}', color=color)
                        ax.fill_between(range(len(df)), df[f'avg_{metric}'] - df[f'std_{metric}'], df[f'avg_{metric}'] + df[f'std_{metric}'], alpha=0.1, color=color)
                        ax.legend()
                else:
                    for ax, metric, color in zip(axs, metrics, colors):
                        ax.plot(df[metric], label=metric, color=color)
                        ax.legend()
                        
        if not file_processed:
            print("No file detected in the metrics folder that fits the model name.")
        return fig

def processsImage(imageInput: Image, modelInput: nn.Module):
    transform = transforms.Compose([transforms.ToTensor()])
    model = modelInput
    model.eval()
    image = imageInput

    image = transform(image)
    with torch.no_grad():
        predicted_output = model(image)
    image = predicted_output.detach().numpy()
    image = np.transpose(image, (1, 2, 0))

    return image

def loadModels(modelsDir):
    models = {}
    for modelFile in os.listdir(modelsDir):
        if modelFile.endswith(".pth"):
            modelName = modelFile.split('.')[0]
            modelType = modelName.split('_')[1]
            match modelType:
                case "espcn":
                    model = ESPCN(scale_factor=1)
                case "srgan":
                    model = SRGAN()
                case "autoencoder":
                    model = Autoencoder()
            model.load_state_dict(torch.load(os.path.join(modelsDir, modelFile)))
            models[modelName] = model
    return models

def fileExists(folder, filename):
    file_path = os.path.join(folder, filename)
    return os.path.isfile(file_path)

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss