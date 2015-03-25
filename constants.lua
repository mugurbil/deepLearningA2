require 'nn'

unlabeled_data_path = '/scratch/courses/DSGA1008/A2/binary/unlabeled_X.bin'
train_data_path = '/scratch/courses/DSGA1008/A2/binary/train_X.bin'
train_labels_path = '/scratch/courses/DSGA1008/A2/binary/train_y.bin'
test_data_path = '/scratch/courses/DSGA1008/A2/binary/test_X.bin'
train_features_path = '/scratch/ml4133/train_features.data'
trained_model_path = '/scratch/ml4133/model.data'

unlabeled_image_count = 100000
train_image_count = 500
test_image_count = 8000
num_categories = 10

image_width = 96
image_height = 96
image_channels = 3
receptive_field_size = 6
patch_size = image_channels*receptive_field_size*receptive_field_size

random_sample_count = 10000
num_centroids = 1600

criterion = nn.MSECriterion()

