import torch
import torch.nn as nn
import pandas as pd
from ModelManager import ModelManager
from DataManager import DataManager
from Models import ESPCN, SAN, Autoencoder, SRGAN, VDSR
from Utils import VGGPerceptualLoss, fileExists, GUIImageProcessor
from GUI import generateGUI

def trainModels(models, modelManager, trainDataset, trainLoader, valLoader, retrain=False, KFOLD=True):
    LEARNINGRATES = [0.01, 0.001, 0.0001]
    L2REGULARIZATIONVALUES = [0.01, 0.001, 0.0001]
    NOOFCONVBLOCKS = [2, 3, 4]
    NOOFCHANNELS = [64, 128, 256]
    MODELS = models
    LOSSFUNCTIONS = {
        "mse": nn.MSELoss(),
        "mae": nn.L1Loss(),
        # "vgg": VGGPerceptualLoss()
    }
    OPTIMIZERS = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop
    }

    # RUNS FOR VERY VERY LONG TIME! FOR EFFIECIENCY USE PREVIOUSLY TUNED HYPERPARAMETERS
    #BESTPARAMS = modelManager.tuneHyperparameters(trainLoader, valLoader, LEARNINGRATES, L2REGULARIZATIONVALUES, NOOFCONVBLOCKS, NOOFCHANNELS)
    BESTPARAMS = {'lr': 0.001, 'l2Reg': 0.001, 'noOfConvBlocks': 2, 'noOfChannels': 64}

    for model in MODELS.values():
        modelManager.describeModel(model)
    for km, model in MODELS.items():
        for kc, criterion in LOSSFUNCTIONS.items():
            for ko, optimizer in OPTIMIZERS.items():
                print(f'Training {km} with {kc} loss and {ko} optimizer')
                if KFOLD:
                    print("Training with KFold")
                    modelFileName = 'model_' + km + '_kfold_' + kc + '_' + ko
                    if retrain:
                        metricsK, _, trainedModel = modelManager.trainKFold(trainDataset, BESTPARAMS, noOfFolds=5, optimizer=optimizer, lossFunction=criterion, model=model)
                    elif not retrain and not fileExists('../models', modelFileName + '.pth'):
                        metricsK, _, trainedModel = modelManager.trainKFold(trainDataset, BESTPARAMS, noOfFolds=5, optimizer=optimizer, lossFunction=criterion, model=model)
                    else:
                        print(f"Model {modelFileName} already exists")
                        continue
                    df = pd.DataFrame({
                        'avg_mse': metricsK[0]['loss'],
                        'std_mse': metricsK[1]['loss'],
                        'avg_psnr': metricsK[0]['psnr'],
                        'std_psnr': metricsK[1]['psnr'],
                        'avg_ssim': metricsK[0]['ssim'],
                        'std_ssim': metricsK[1]['ssim']
                    })
                    df.to_csv(f'../metrics/{modelFileName}_metrics.csv', index=False)
                    torch.save(trainedModel.state_dict(), '../models/' + modelFileName + '.pth')
                else:
                    print("Training without KFold")
                    modelFileName = 'model_' + km + '_' + kc + '_' + ko
                    if model == "ESCPN":
                        trainedModel = model(scale_factor=1, noOfConvBlocks=BESTPARAMS["noOfConvBlocks"], noOfChannels=BESTPARAMS["noOfChannels"])
                    else:
                        trainedModel = model()

                    if retrain:
                        metrics = modelManager.trainModel(trainedModel, trainLoader, criterion, optimizer(trainedModel.parameters(), lr=BESTPARAMS["lr"], weight_decay=BESTPARAMS["l2Reg"]), numEpochs = 10)
                    elif not retrain and not fileExists('../models', modelFileName + '.pth'):
                        metrics = modelManager.trainModel(trainedModel, trainLoader, criterion, optimizer(trainedModel.parameters(), lr=BESTPARAMS["lr"], weight_decay=BESTPARAMS["l2Reg"]), numEpochs = 10)
                    else:
                        print(f"Model {modelFileName} already exists")
                        continue
                    df = pd.DataFrame(metrics, columns=['MSE', 'PSNR', 'SSIM'])
                    df.to_csv(f'../metrics/{modelFileName}_metrics.csv', index=False)
                    torch.save(trainedModel.state_dict(), '../models/' + modelFileName + '.pth')

if __name__ == '__main__':
    DATADIRS = [
        '../data/ISR/raw_data', 
        '../data/Own'
    ]
    dataManager = DataManager()
    modelManager = ModelManager()
    models = { "autoencoder": Autoencoder, "espcn": ESPCN, "vdsr": VDSR} # Don't use SRGAN or SAN because their output is not correctly adjusted
    models = { "vdsrc": VDSR }
    imageProcessor = GUIImageProcessor('../models')

    train, val, test = dataManager.build(DATADIRS, augmented=False)
    trainDataset, trainLoader = train
    valDataset, valLoader = val
    testDataset, testLoader = test

    # trainModels(models, modelManager, trainDataset, trainLoader, valLoader, KFOLD=False, retrain=False)

    # for name, model in imageProcessor.models.items():
    #     metrics = modelManager.evaluateModel(model, testLoader, ['psnr', 'ssim', 'mse'])
    #     print(metrics)

    GUI = generateGUI(imageProcessor.models.keys(), imageProcessor.processImage, imageProcessor.visualizeTraining)
    GUI.launch()


    