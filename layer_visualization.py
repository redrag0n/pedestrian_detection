import torch
import matplotlib.pyplot as plt
from lucent.optvis import render
from lucent.modelzoo.util import get_model_layers

from classifier import Classifier


plt.rcParams["savefig.bbox"] = 'tight'


if __name__ == '__main__':
    layers = ['backbone_body_layer4_2_conv3', 'backbone_body_layer4_2_conv2',
              # 'backbone_body_layer4_2_conv1', 'backbone_body_layer4_1_conv3',
              # 'backbone_body_layer4_1_conv2', 'backbone_body_layer4_1_conv1',
              'backbone_body_layer4_0_conv3', 'backbone_body_layer4_0_conv2',
              'backbone_body_layer4_0_conv1', 'backbone_body_layer3_3_conv3',
              'backbone_body_layer3_2_conv2', 'backbone_body_layer3_1_conv1',
              'backbone_body_layer3_0_conv3', 'backbone_body_layer2_2_conv2',
              'backbone_body_layer1_2_conv2', 'backbone_body_layer1_0_conv3']
    nums = ['1', '10', '100', '1000', '200', '500']
    classifier = Classifier.load('trained_models/epoch10')
    classifier.model.eval()
    # print(get_model_layers(classifier.model))
    for layer in layers:
        for num in nums:
            print(f'layer: {layer} num: {num}')
            try:
                plt.figure(figsize=(20, 20))
                # _ = render.render_vis(classifier.model, f"{layer}:{num}", show_inline=False, thresholds=(1,),
                #                       save_image=True, image_name=f'trained_models/visualize{layer}-{num}.jpeg')
                _ = render.render_vis(classifier.model, f"{layer}:{num}", show_inline=False, thresholds=(512,),
                                      save_image=True, image_name=f'trained_models/visualize{layer}-{num}.jpeg')
                # plt.savefig(f'trained_models/visualize{layer}-{num}')
                plt.clf()
                plt.close()
            except BaseException as error:
                print('!!!!', error)
                continue


