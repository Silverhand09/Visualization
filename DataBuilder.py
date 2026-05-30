import os
import torch
from torchvision import datasets, transforms


class DataBuilder:
    def __init__(self, dataDir='./Data', batchSize=128):
        self.dataDir = dataDir
        self.batchSize = batchSize

        self.dataTransforms = {
            'training':
                transforms.Compose([
                    transforms.Resize([64, 64]),
                    transforms.RandomRotation(45),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomGrayscale(p=0.025),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    )
                ]),
            'valid':
                transforms.Compose([
                    transforms.Resize([64, 64]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    )
                ]),
        }

    def buildData(self):
        trainingDataset = datasets.ImageFolder(
            os.path.join(self.dataDir, 'Training'),
            self.dataTransforms['training']
        )

        validDataset = datasets.ImageFolder(
            os.path.join(self.dataDir, 'Valid'),
            self.dataTransforms['valid']
        )

        trainingLoader = torch.utils.data.DataLoader(
            trainingDataset,
            batch_size=self.batchSize,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        validLoader = torch.utils.data.DataLoader(
            validDataset,
            batch_size=self.batchSize,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        dataloaders = {
            'training': trainingLoader,
            'valid': validLoader
        }

        datasetSizes = {
            'training': len(trainingDataset),
            'valid': len(validDataset)
        }

        classNames = trainingDataset.classes

        return dataloaders, datasetSizes, classNames