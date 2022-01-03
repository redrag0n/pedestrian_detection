import torch
from engine import train_one_epoch, evaluate
import utils

import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.ops.boxes import nms, batched_nms

from pedestrian_dataset import PennFudanDataset, get_transform
from model import get_model_instance_segmentation, get_model, get_resnet_model


plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    plt.figure(figsize=(20, 60))
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


class Classifier:
    def __init__(self, num_classes=2, model=None):
        if model is not None:
            self.model = model
        else:
            #self.model = get_model(num_classes)
            self.model = get_resnet_model(num_classes)

    def log_results(self, dataset, output_dir, epoch, log_count=3):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.eval()
        images = [dataset[i][0].to(device) for i in range(log_count)]
        original_images = [dataset[i][1]['original'] for i in range(log_count)]
        true_boxes = [dataset[i][1]['boxes'].to(device) for i in range(log_count)]
        prediction = self.predict(images)
        predicted_boxes = [p['boxes']for p in prediction]

        boxed_images = [draw_bounding_boxes(original_images[i],
                                            torch.cat([true_boxes[i], predicted_boxes[i]], dim=0),
                                            colors=['blue'] * len(true_boxes[i]) + ['green'] * len(predicted_boxes[i]),
                                            width=2)
                        for i in range(log_count)]
        show(boxed_images)
        plt.savefig(f'{output_dir}/epoch{epoch}.png', dpi=300)
        plt.clf()
        plt.close()

    def train(self, data_dir, output_dir, num_epochs=10):
        # train on the GPU or on the CPU, if a GPU is not available
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        num_workers = 4 if torch.cuda.is_available() else 0
        # use our dataset and defined transformations
        dataset = PennFudanDataset(data_dir, get_transform(train=True))
        dataset_test = PennFudanDataset(data_dir, get_transform(train=False))
        dataset_log = PennFudanDataset(data_dir, get_transform(train=False), get_original_image=True)

        # split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
        dataset_log = torch.utils.data.Subset(dataset_log, indices[-50:])

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=num_workers,
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=num_workers,
            collate_fn=utils.collate_fn)

        # get the model using our helper function
        #model = get_model_instance_segmentation(num_classes)


        # move model to the right device
        self.model.to(device)

        # construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        self.log_results(dataset_log, output_dir, -1)
        for epoch in range(num_epochs):
            train_one_epoch(self.model, optimizer, data_loader, device, epoch, print_freq=10)

            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(self.model, data_loader_test, device=device)

            self.save(f'{output_dir}/epoch{epoch}')
            self.log_results(dataset_log, output_dir, epoch)


    def predict(self, images):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        if not isinstance(images, list):
            images = [images]
        return self.model(images)

    def save(self, output_path):
        torch.save(self.model, output_path)

    @staticmethod
    def load(input_path):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = torch.load(input_path, device)
        return Classifier(model=model)
