import torch
import torch.nn as nn
import torch.optim as optim

from DataBuilder import DataBuilder
from ModelBuilder import ModelBuilder
from Trainer import Trainer
from Validator import Validator

dataDir = './Data'

batchSize = 256

modelName = 'resnet'
featureExtract = True

trainOnGpu = torch.cuda.is_available()

if not trainOnGpu:
    print('CUDA is not available. . Training on CPU, attempting...')
else:
    print('Training on GPU ...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


dataBuilder = DataBuilder(dataDir, batchSize)

dataloaders, trainingSize, classNames = dataBuilder.buildData()

numClasses = len(classNames)


modelBuilder = ModelBuilder(modelName, featureExtract, usePretrained=True)

modelTuning, inputSize = modelBuilder.initializeModel(numClasses)

modelTuning = modelTuning.to(device)


fileName = 'elGIS_MkI'

paramsToUpdate = []

print("Parameter training activated. Params to train:")
for name, param in modelTuning.named_parameters():
    if param.requires_grad:
        paramsToUpdate.append(param)
        print("\t", name)

optimizerTuning = optim.Adam(paramsToUpdate, lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizerTuning, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()


trainer = Trainer(
    modelTuning,
    dataloaders,
    criterion,
    optimizerTuning,
    scheduler,
    device,
    fileName
)

modelTuning, valAccuracyHistory, trainAccuracyHistory, validLosses, trainLosses, LRs = trainer.trainModel()


validator = Validator(
    modelTuning,
    dataloaders['valid'],
    classNames,
    device
)

validator.showPredictions()          
