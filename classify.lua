require 'preprocessing'
require 'torch'
require 'math'
require 'constants'
require 'xlua'

cmd = torch.CmdLine()
cmd:option('-start', 1)
cmd:option('-finish', test_image_count)
cmd:option('-label', 1)
cmd:option('-model', trained_model_path)
params = cmd:parse(arg)

print("Loading test data")
data = torch.DiskFile(test_data_path, 'r', true)
data:binary():littleEndianEncoding()
tensor = torch.ByteTensor(test_image_count, image_channels, image_width, image_height)
data:readByte(tensor:storage())
test_data = tensor:float()

print("Loading trained neural net from "..params.model)
net = torch.load(params.model, 'ascii')

print("Classifying")
guesses = torch.IntTensor(test_image_count)

for i = params.start, params.finish do
    image = test_data[i]
    patches = patchify(image)
    features = extract_features(patches)
    features:resize(1, features:size()[1], features:size()[2]) --batch size 1
    classification = net:forward(features)

    best_index = 1
    best_delta = math.huge
    for guess = 1, num_categories do
    	--print("Guessing "..guess)
    	guess_as_vector = torch.DoubleTensor(num_categories):fill(0)
	guess_as_vector[guess] = 1
    	delta = criterion:forward(classification, guess_as_vector) 
	--print("Error "..delta)
	if delta < best_delta then
	   best_delta = delta
	   best_index = guess
	end
	--print("Best guess so far "..best_index)
    end
    guesses[i] = best_index
end

print("Saving classifications")
torch.save("classifications"..params.label..".data", guesses, 'ascii')