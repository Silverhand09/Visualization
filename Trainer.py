import time
import copy
import torch


class Trainer:
    def __init__(self, model, dataloaders, criterion, optimizer, scheduler, device, fileName='bestModel.pt', numEpochs=10):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.fileName = fileName
        self.numEpochs = numEpochs

    def trainModel(self):
        since = time.time()

        bestAccuracy = 0

        self.model.to(self.device)

        valAccuracyHistory = []
        trainAccuracyHistory = []
        trainLosses = []
        validLosses = []

        LRs = [self.optimizer.param_groups[0]['lr']]
        bestModelWeights = copy.deepcopy(self.model.state_dict())

        for epoch in range(self.numEpochs):
            print('Epoch {}/{}'.format(epoch, self.numEpochs - 1))
            print('-' * 10)

            for phase in ['training', 'valid']:
                if phase == 'training':
                    self.model.train()
                else:
                    self.model.eval()

                runningLoss = 0.0
                runningCorrects = 0

                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    # Initialize
                    self.optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'training':
                        loss.backward()
                        self.optimizer.step()

                    # Loss calculation
                    runningLoss += loss.item() * inputs.size(0)
                    runningCorrects += torch.sum(preds == labels.data) # To estimate the result

                epochLoss = runningLoss / len(self.dataloaders[phase].dataset)
                epochAccuracy = runningCorrects.double() / len(self.dataloaders[phase].dataset)

                timeElapsed = time.time() - since # Time cost for one epoch
                print('Time elapsed {:.0f}m {:.0f}s'.format(timeElapsed // 60, timeElapsed % 60))
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epochLoss, epochAccuracy))

                # Get the best model
                if phase == 'valid' and epochAccuracy > bestAccuracy:
                    bestAccuracy = epochAccuracy
                    bestModelWeights = copy.deepcopy(self.model.state_dict())
                    state = {
                        'state_dictionary': self.model.state_dict(),
                        'best_accuracy': bestAccuracy,
                        'optimizer': self.optimizer.state_dict(),
                    }
                    torch.save(state, self.fileName)

                if phase == 'valid':
                    valAccuracyHistory.append(epochAccuracy)
                    validLosses.append(epochLoss)

                if phase == 'training':
                    trainAccuracyHistory.append(epochAccuracy)
                    trainLosses.append(epochLoss)

            print('Optimizer learning rate : {:.7f}'.format(self.optimizer.param_groups[0]['lr']))
            LRs.append(self.optimizer.param_groups[0]['lr'])
            print()
            self.scheduler.step()

        timeElapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(timeElapsed // 60, timeElapsed % 60))
        print('Best val Acc: {:4f}'.format(bestAccuracy))

        self.model.load_state_dict(bestModelWeights)

        return self.model, valAccuracyHistory, trainAccuracyHistory, validLosses, trainLosses, LRs