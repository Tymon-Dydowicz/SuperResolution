import torch
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as PSNR, structural_similarity as SSIM
import torch.nn as nn
from typing import List
from torchsummary import summary
from itertools import product
from Models import ESPCN
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold


class ModelManager():
    def __init__(self) -> None:
        pass

    def trainModel(self, model, train_loader,criterion, optimizer, numEpochs=10):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        metrics = {
            "loss": [],
            "psnr": [],
            "ssim": []
        }
        for epoch in tqdm(range(numEpochs)):
            model.train()
            runningLoss, totalPsnr, totalSsim = 0.0, 0.0, 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                runningLoss += loss.item()

                outputs_np = outputs.detach().cpu().numpy()
                targets_np = targets.detach().cpu().numpy()

                dataRange = 1.0 if targets_np.max() <= 1.0 else 255.0
                batchPSNR = np.mean([PSNR(t, o) for t, o in zip(targets_np, outputs_np)])
                batchSSIM = np.mean([SSIM(t, o, channel_axis=0, data_range=dataRange) for t, o in zip(targets_np, outputs_np)])

                totalPsnr += batchPSNR
                totalSsim += batchSSIM

            metrics["loss"].append(runningLoss / len(train_loader))
            metrics["psnr"].append(totalPsnr / len(train_loader))
            metrics["ssim"].append(totalSsim / len(train_loader))
        return metrics
    
    def evaluateModel(self, model, testLoader=None, metrics=["mse"]):
        metricValues = {
            "mse": 0.0,
            "psnr": 0.0,
            "ssim": 0.0
        }
        if testLoader is None:
            return metricValues
        model.eval()
        criterion = nn.MSELoss()
        with torch.no_grad():
            for inputs, targets in testLoader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                if "psrn" in metrics or "ssim" in metrics:
                    outputs_np = outputs.detach().cpu().numpy()
                    targets_np = targets.detach().cpu().numpy()

                if "psnr" in metrics:
                    batchPSNR = np.mean([PSNR(t, o) for t, o in zip(targets_np, outputs_np)])
                    metricValues["psnr"] += batchPSNR/len(testLoader)

                if "ssim" in metrics:
                    dataRange = 1.0 if targets_np.max() <= 1.0 else 255.0
                    batchSSIM = np.mean([SSIM(t, o, channel_axis=0, data_range=dataRange) for t, o in zip(targets_np, outputs_np)])
                    metricValues["ssim"] += batchSSIM/len(testLoader)

                if "mse" in metrics:
                    metricValues["mse"] += loss.item()/len(testLoader)

        return metricValues
    
    def describeModel(self, model) -> None:
        summary(model(), (3, 256, 256))

    def tuneHyperparameters(self, valTrainLoader, valTestLoader, learningRates, L2RegularizationValues, noOfConvBlocks, noOfChannels):
        bestHyperparameters = None
        bestValSSIM = float('-inf')

        for lr, l2Reg, noOfConvBlocks, noOfChannels in product(learningRates, L2RegularizationValues, noOfConvBlocks, noOfChannels):
            print(f"lr: {lr}, l2_reg: {l2Reg}, noOfConvBlocks: {noOfConvBlocks}, noOfChannels: {noOfChannels}")

            model = ESPCN(scale_factor=1, noOfConvBlocks=noOfConvBlocks, noOfChannels=noOfChannels)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2Reg)
            criterion = nn.MSELoss()

            self.trainModel(model, valTrainLoader, criterion, optimizer, numEpochs=5)

            valSSIM = self.evaluateModel(model, valTestLoader, ["ssim"])["ssim"]
            print(f"Validation Loss: {valSSIM}")

            if valSSIM > bestValSSIM:
                bestValSSIM = valSSIM
                bestHyperparameters = {"lr": lr, "l2Reg": l2Reg, "noOfConvBlocks": noOfConvBlocks, "noOfChannels": noOfChannels}



        print(f"ESPCN Best hyperparameters: {bestHyperparameters}")
        return bestHyperparameters
    
    def trainKFold(self, trainDataset, bestParams, noOfFolds=2, optimizer=torch.optim.Adam, lossFunction=nn.MSELoss(), model = ESPCN):
        trainingResults = []
        validationResults = []

        kf = KFold(n_splits=noOfFolds, shuffle=True, random_state=23)
        noOfSamples = len(trainDataset)

        for fold, (trainIndicies, valIndicies) in enumerate(kf.split(np.arange(noOfSamples))):
            print(f"Fold: {fold+1}/{noOfFolds}")

            trainLoader = DataLoader(Subset(trainDataset, trainIndicies), batch_size=1, shuffle=True)
            valLoader = DataLoader(Subset(trainDataset, valIndicies), batch_size=1, shuffle=True)

            modelFold = model()
            criterionFold = lossFunction
            optimizerFold = optimizer(modelFold.parameters(), lr=bestParams['lr'], weight_decay=bestParams['l2Reg'])

            metrics = self.trainModel(modelFold, trainLoader, criterionFold, optimizerFold, numEpochs=10)

            trainingResults.append(metrics)
            validationResults.append(self.evaluateModel(modelFold, valLoader, ["ssim"]))

        # Calculate the average and standard deviation of each metric in a given epoch across all folds
        avg_metrics = {metric: np.mean([fold_metrics[metric] for fold_metrics in trainingResults], axis=0) for metric in trainingResults[0]}
        std_metrics = {metric: np.std([fold_metrics[metric] for fold_metrics in trainingResults], axis=0) for metric in trainingResults[0]}
            
        return (avg_metrics, std_metrics), validationResults, modelFold