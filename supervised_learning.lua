require 'torch'
require 'nn'
require 'constants'
require 'math'

print("Loading labels")
data = torch.DiskFile(train_labels_path, 'r', true)
data:binary():littleEndianEncoding()
tensor = torch.ByteTensor(train_image_count)
data:readByte(tensor:storage())
tensor = tensor:int()
labels = torch.FloatTensor(train_image_count, num_categories):fill(0)
for i = 1, train_image_count do
    labels[i][tensor[i]] = 1
end

print("Loading preprocessed training data")
training_data = torch.load(train_features_path, 'ascii')
flat_size = training_data:size()[2] * training_data:size()[3]

print("Building neural net")
hidden_layer_count = 200
net = nn.Sequential()
net:add(nn.Reshape(flat_size, true)) --reshape with batch mode
net:add(nn.Linear(flat_size, hidden_layer_count))
net:add(nn.Tanh())
net:add(nn.Linear(hidden_layer_count, num_categories))

print("Training")
learning_rate = .01
batch_size = 10

epoch = 0

while true do
      epoch = epoch + 1
      print("Epoch "..epoch)
      processed_count = 0
      while processed_count < train_image_count do
            batch_lower = processed_count + 1
            batch_upper = math.min(train_image_count, batch_lower + batch_size -1)
            batch = training_data[{{batch_lower, batch_upper}, {}, {}}]
            batch_labels = labels[{{batch_lower, batch_upper}}]:double()
            criterion:forward(net:forward(batch), batch_labels)
	    nancheck = torch.ne(net.output, net.output):byte()
	    if torch.any(nancheck) then
	       print("nancheck failed!")
	       print(nancheck)
	       goto nancheck_fail
	    end
      	    net:zeroGradParameters()
            net:backward(batch, criterion:backward(net.output, batch_labels))
            net:updateParameters(learning_rate)
            processed_count = processed_count + batch_size
     end
     print("Saving Model")
     torch.save(trained_model_path, net, 'ascii')
end

::nancheck_fail::

