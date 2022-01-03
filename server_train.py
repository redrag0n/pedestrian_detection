from classifier import Classifier


data_dir = 'PennFudanPed'
output_dir = 'trained_models'
epoch_count = 30

Classifier().train(data_dir, output_dir, epoch_count)
