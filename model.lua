-- deep learing assignment 2
-- m&m : mehmet ugurbil, mark liu
-- model for learning stl-10
-- 3/10/2015
print("Starting module.lua")
require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
require 'optim'

-- -- get the preprocessed data
-- www = 'http://cims.nyu.edu/~ml4133/'
-- --os.execute('wget ' .. www .. train_file)
save = 'results'
-- getting the data and then shaping it 
print("Loading training file")
trsize = 500
numPatch = 517
train_file = 'train_features.data'
loaded = torch.load(train_file, 'ascii')
-- labelsFile = torch.load('/scratch/courses/DSGA1008/A2/binary/train_y.bin','byte')
print("Loaded training file")
train_labels_path = '/scratch/courses/DSGA1008/A2/binary/train_y.bin'
print("Loading labels")
data = torch.DiskFile(train_labels_path, 'r', true)
data:binary():littleEndianEncoding()
tensor = torch.ByteTensor(trsize)
data:readByte(tensor:storage())
labelsFile = tensor:double()

-- inLabels = torch.Tensor(trsize)
-- loaded = torch.rand(trsize,numPatch,1600)

labels = torch.Tensor(numPatch,trsize):zero()
for i = 1, numPatch do
  labels[i] = labelsFile
end
labels = labels:resize(trsize*numPatch)
net = nn.Reshape(trsize*numPatch,1,40,40)
resized = net:forward(loaded)
trainData = {
   data = resized,
   labels = labels,
   size = function() return trsize end
}
----------------------------------------------------------------------
print '==> construct model'
model = nn.Sequential()

-- stage 1 : zero padding -> filter bank -> squashing -> max pooling -> normalization
model:add(nn.SpatialZeroPadding(2, 2, 2, 2))
model:add(nn.SpatialConvolutionMM(1, 64, 5, 5, 2, 2))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))

-- stage 2 : filter bank -> squashing -> max pooling -> normalization
model:add(nn.SpatialConvolutionMM(64, 128, 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(6,6))

-- stage 3 : dropout -> view -> linear -> squashing
model:add(nn.Dropout(0.5))
model:add(nn.View(128))
model:add(nn.Linear(128, 128))
model:add(nn.ReLU())

-- stage 4 : linear -> log softmax
model:add(nn.Linear(128, 10))
model:add(nn.LogSoftMax())

print '==> here is the model:'
print(model)

----------------------------------------------------------------------
print '==> define loss'

-- negative log likelihood loss function
criterion = nn.ClassNLLCriterion()

print '==> here is the loss function:'
print(criterion)
----------------------------------------------------------------------
--training
----------------------------------------------------------------------
saveFile = 'results'
-- CUDA
-- model:cuda()
-- criterion:cuda()

----------------------------------------------------------------------
print '==> defining some tools'

-- classes
classes = {'1','2','3','4','5','6','7','8','9','0'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(save, 'train.log'))
testLogger = optim.Logger(paths.concat(save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'

optimState = {
  learningRate = 0.01,
  weightDecay = 0.0,
  momentum = 0.9,
  learningRateDecay = 1e-7
}
optimMethod = optim.sgd

----------------------------------------------------------------------

print '==> defining training procedure'

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   batchSize = 1
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1,trainData:size(),batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+batchSize-1,trainData:size()) do
         -- load new sample
         local input = trainData.data[shuffle[i]]
         local target = trainData.labels[shuffle[i]]
         input = input:double()
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          -- update confusion
                          confusion:add(output, targets[i])
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      optimMethod(feval, parameters, optimState)

   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}

   -- save/log current net
   local filename = paths.concat(saveFile, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end
----------------------------------------------------------------------

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(2)
torch.manualSeed(1)

-- for i = 1,10 do
-- 	train()
-- end

print("Training")
learning_rate = .001
decay = .0000001
batch_size = 10

epoch = 0

while true do
      epoch = epoch + 1
      print("Epoch "..epoch)
      processed_count = 0
      while processed_count < train_image_count do
            xlua.progress(processed_count, train_image_count)
            batch_lower = processed_count + 1
            batch_upper = math.min(train_image_count, batch_lower + batch_size -1)
            batch = training_data[{{batch_lower, batch_upper}, {}, {}}]
            batch_labels = labels[{{batch_lower, batch_upper}}]:double()
            criterion:forward(net:forward(batch), batch_labels)
            net:zeroGradParameters()
            net:backward(batch, criterion:backward(net.output, batch_labels))
            net:updateParameters(learning_rate)
            processed_count = processed_count + batch_size
            learning_rate = learning_rate * decay
     end
     print("Saving Model")
     torch.save(trained_model_path, net, 'ascii')
end

