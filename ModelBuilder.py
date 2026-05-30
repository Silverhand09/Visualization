import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class ModelBuilder:
    def __init__(self,  modelName='resnet', featureExtract=True, usePretrained=True):
        self.modelName = modelName
        self.featureExtract = featureExtract
        self.usePretrained = usePretrained

    def setParameterRequiresGrad(self, model):
        if self.featureExtract:
            for param in model.parameters():
                param.requires_grad = False

    def initializeModel(self, numClasses):
        if self.usePretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
    
        modelTuning = models.resnet50(weights=weights)
        self.setParameterRequiresGrad(modelTuning)

        numTuningFeatures = modelTuning.fc.in_features
        modelTuning.fc = nn.Linear(numTuningFeatures, numClasses)

        inputSize = 64

        return modelTuning, inputSize