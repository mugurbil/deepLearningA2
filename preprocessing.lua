require 'constants'
require 'torch'

whitening_matrix = torch.load('whitening_matrix.data', 'ascii')
negative_feature_means = torch.load('feature_means.data', 'ascii'):reshape(patch_size):mul(-1)

stride = 4
num_patches = (image_width - receptive_field_size + 1)/stride*(image_height - receptive_field_size + 1)/stride
num_patches = math.floor(num_patches)

function patchify(image)    
    patches = torch.FloatTensor(num_patches, patch_size)
    i = 1
    y = 1
    x = 1
    while y + receptive_field_size - 1 < image_height do
	x = 1
	while x + receptive_field_size - 1 < image_width do
	    patch = image[{{}, {x, x + receptive_field_size -1}, {y, y + receptive_field_size - 1}}]
	    flattened_patch = torch.reshape(patch, image_channels*receptive_field_size*receptive_field_size, 1)
	    mean = flattened_patch:mean()
	    std = flattened_patch:std()
	    flattened_patch:add(-mean)
	    if std ~= 0 then
	       flattened_patch:div(std)
	    end
	    whitened_patch = torch.mm(whitening_matrix, flattened_patch:add(negative_feature_means))
	    patches[i] = whitened_patch
	    x = x + stride
	    i = i + 1
	    if i > num_patches then
	       goto done
	    end
        end
	y = y + stride
    end
    ::done::
    return patches
end

negative_centroids = torch.load('centroids.data', 'ascii'):mul(-1)

function extract_features(patches)
    num_patches = patches:size()[1]
    features = torch.DoubleTensor(num_patches, num_centroids)
    for i = 1, num_patches do
        patch = patches[i]
    	patch:resize(1, patch_size)
	patch = patch:expand(num_centroids, patch_size)
	patch = patch:double()
	sum = torch.add(negative_centroids, patch)
	sum:pow(2)
	z = torch.sum(sum, 2)
	z:sqrt()
	mean = z:mean()
	z:add(-mean)
	z:mul(-1)
	positives = torch.gt(z, 0):double()
	z:cmul(positives)
	z:resize(num_centroids)
	features[i] = z
    end
    return features
end

