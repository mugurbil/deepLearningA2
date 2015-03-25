require 'preprocessing'

print('Loading train data')
data = torch.DiskFile(train_data_path, 'r', true)
data:binary():littleEndianEncoding()
tensor = torch.ByteTensor(train_image_count, image_channels, image_width, image_height)
data:readByte(tensor:storage())
tensor = tensor:float()

function process_all()
    all_features = torch.DoubleTensor(train_image_count, num_patches, num_centroids)
    for i = 1, train_image_count do
        image = tensor[i]
        patches = patchify(image)
        image_features = extract_features(patches)
        all_features[i] = image_features
    end
    return all_features
end

print("Extracting Features")
all_features = process_all()

print("Saving")
torch.save(train_features_path, all_features, 'ascii')