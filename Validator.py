import torch
import matplotlib.pyplot as plt
import numpy as np

class Validator:
    def __init__(self, model, dataloader, class_names, device):
        self.model = model
        self.dataloader = dataloader
        self.class_names = class_names
        self.device = device

    def imageConvert(self, tensor):

        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1, 2, 0)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        image = image * std + mean
        image = image.clip(0, 1)

        return image

    def getOneBatchPredictions(self):

        self.model.eval()

        dataiter = iter(self.dataloader)
        images, labels = next(dataiter)

        images = images.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
            _, preds = torch.max(outputs, 1)

        return images, labels, preds, outputs

    def showPredictions(self, rows=2, columns=4):

        images, labels, preds, outputs = self.getOneBatchPredictions()

        images = images.to("cpu")
        labels = labels.to("cpu")
        preds = preds.to("cpu")

        fig = plt.figure(figsize=(20, 10))

        for idx in range(rows * columns):
            ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])

            image = self.imageConvert(images[idx])
            ax.imshow(image)

            pred_name = self.class_names[preds[idx]]
            actual_name = self.class_names[labels[idx].item()]

            title_color = "green" if pred_name == actual_name else "red"

            ax.set_title(
                "{} ({})".format(pred_name, actual_name),
                color=title_color
            )

        plt.show()