-- deep learing assignment 2
-- m&m : mehmet ugurbil, mark liu
-- model for learning stl-10
-- 3/10/2015

require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers


-- get the preprocessed data
www = 'http://cims.nyu.edu/~ml4133/'
train_file = 'train_features.data'
os.execute('wget ' .. www .. train_file)

trsize = 500
data = torch.load(train_file, 'ascii')
trainData = {
   --data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return trsize end
}
----------------------------------------------------------------------
print '==> construct model'
model = nn.Sequential()

-- stage 1 : zero padding -> filter bank -> squashing -> max pooling -> normalization
model:add(nn.Reshape(trsize*517,1,40,40))
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
model:cuda()
criterion:cuda()

----------------------------------------------------------------------
print '==> defining some tools'

-- classes
classes = {'1','2','3','4','5','6','7','8','9','0'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

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
  weightDecay = 0,
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
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,trainData:size(),128 do
      -- disp progress
      xlua.progress(t, trainData:size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+128-1,trainData:size()) do
         -- load new sample
         local input = trainData.data[shuffle[i]]
         local target = trainData.labels[shuffle[i]]
         input = input:cuda()
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

require 'cunn'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(2)
torch.manualSeed(1)

for i = 1,1 do
	train()
end

