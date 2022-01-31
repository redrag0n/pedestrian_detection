import os.path

import matplotlib.pyplot as plt
import numpy as np
import captum
import torch
from torch import Tensor
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap

from classifier import Classifier
from pedestrian_dataset import PennFudanDataset, get_transform

if os.path.isdir('../PennFudanPed'):
    data_dir = '../PennFudanPed'
    epochs = 2
else:
    data_dir = 'PennFudanPed'
    epochs = 200


def main():
    classifier = Classifier.load('trained_models/epoch10')
    classifier.model.to()
    classifier.model.eval()
    dataset_test = PennFudanDataset(data_dir, get_transform(train=False), get_original_image=True)
    indices = torch.randperm(len(dataset_test)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    for i in range(len(dataset_test)):
        print(i)
        input_ = dataset_test[i]
        normalized_input = input_[0].unsqueeze(0)
        normalized_input = normalized_input.to(device)
        normalized_input.requires_grad = True

        def wrapper(inp):
            # print(classifier.model(inp))
            res = torch.stack([torch.sum(el['scores']) for el in classifier.model(inp)])
            # print(res)
            return res

        integrated_gradients = IntegratedGradients(wrapper)
        attributions_ig = integrated_gradients.attribute(normalized_input, n_steps=epochs, internal_batch_size=5)

        default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                         [(0, '#ffffff'),
                                                          (0.25, '#000000'),
                                                          (1, '#000000')], N=256)

        fig, axs = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                     np.transpose(input_[1]['original'].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                     method='heat_map',
                                     cmap=default_cmap,
                                     show_colorbar=True,
                                     sign='positive',
                                     outlier_perc=1)
        fig.savefig(f'trained_models/heatmap_{input_[1]["image_id"][0]}')


if __name__ == '__main__':
    main()

