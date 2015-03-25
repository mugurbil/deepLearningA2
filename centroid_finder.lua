require 'constants'
require 'nn'
require 'torch'
require 'math'
require 'xlua'
require 'kmeans'

unlabeled_data_path = '/scratch/courses/DSGA1008/A2/binary/unlabeled_X.bin'

-- Load training data
print("Loading training data")

data = torch.DiskFile(unlabeled_data_path, 'r', true)
data:binary():littleEndianEncoding()
tensor = torch.ByteTensor(unlabeled_image_count, image_channels, image_width, image_height)
data:readByte(tensor:storage())
tensor:float()

-- Random Patch Selection 
print("Picking random patches")
random_patches = torch.FloatTensor(random_sample_count, image_channels, receptive_field_size, receptive_field_size)
patch_count = 0
while patch_count < random_sample_count do
    image_index = math.random(1, unlabeled_image_count)
    image = tensor[image_index]
    patch_x = math.random(1, image_width - receptive_field_size + 1)
    patch_y = math.random(1, image_height - receptive_field_size + 1)
    patch = image[{{}, {patch_x, patch_x + receptive_field_size - 1}, {patch_y, patch_y + receptive_field_size - 1}}]
    patch_count = patch_count + 1
    random_patches[patch_count] = patch
    mean = random_patches[{patch_count, {}, {}, {}}]:mean()
    std = random_patches[{patch_count, {}, {}, {}}]:std()       
    random_patches[{patch_count, {}, {}, {}}]:add(-mean)
    if std ~= 0 then
        random_patches[{patch_count, {}, {}, {}}]:div(std)
    end
end

-- Whitening
print("Whitening")

patch_size = image_channels*receptive_field_size*receptive_field_size
random_patches_flattened = torch.reshape(random_patches, random_sample_count, patch_size)
means = torch.mean(random_patches_flattened, 1)
random_patches_flattened:add(means:expand(random_sample_count, patch_size))
covariances = torch.mm(random_patches_flattened:transpose(1,2), random_patches_flattened):div(random_sample_count)
Q, D, Q_T = torch.svd(covariances)
D_inv_sqrt = torch.pow(D, -1):sqrt():resize(patch_size, 1)
D_isQ_t = Q_T:cmul(D_inv_sqrt:expand(patch_size, patch_size))
W_zca = torch.mm(Q, D_isQ_t)
whitened = torch.mm(W_zca, random_patches_flattened:transpose(1,2)):transpose(1,2)

-- k-means
print("Computing centroids")

function dummy(x,y,z)
    return 0
end

centroids = kmeans(whitened:double(), num_centroids, 100, 100, callback, true)

-- Output
print("Saving results")

torch.save('centroids.data', centroids, 'ascii')
torch.save('whitening_matrix.data', W_zca, 'ascii')
torch.save('feature_means.data', means, 'ascii')