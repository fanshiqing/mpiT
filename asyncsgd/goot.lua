-------------------------------------------------------------------
-- Author: Sixin Zhang (zsx@cims.nyu.edu)
-- Author: Shiqing Fan (sqfan6@gmail.com)
-- Ref: https://github.com/torch/demos/blob/master/train-on-cifar/train-on-cifar.lua
-------------------------------------------------------------------
local opt = opt or {}
local state = state or {}
local optname = opt.name or 'sgd'
local full = opt.full or nil -- use full dataset or not
local lr = opt.lr or 1e-2
local mom = opt.mom or 0.99
local mb = opt.mb or 128
local mva = opt.mva or 0
local su = opt.su or 1
local maxep = opt.maxepoch or 1000
local saveep = opt.saveep or 10 -- save model every saveep epoch
local data_root = opt.data_root or
   io.popen('echo $HOME'):read() .. '/data/torch7/mnist10'
local gpuid = opt.gpuid or -1
local rank = opt.rank or -1
local pclient = opt.pc or nil
-------------------------------------------------------------------
require 'sys'
local tm = {}
tm.feval = 0
tm.sync = 0
-------------------------------------------------------------------
require 'os'
local seed = opt.seed or os.time()
torch.manualSeed(seed) -- remember to set cutorch.manualSeed if needed
-------------------------------------------------------------------
require 'nn'
local function buildModel()
   model = nn.Sequential()
--   model:add(nn.Reshape(3,32,32))
   model:add(nn.Reshape(3,28,28))
   model:add(nn.SpatialConvolution(3,64,5,5)) -- 64*24*24
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(2,2,2,2)) -- 64*12*12

   model:add(nn.SpatialConvolution(64,128,5,5)) -- 128*8*8
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(2,2,2,2)) -- 128*4*4

   model:add(nn.SpatialConvolution(128,64,3,3)) -- 64*2*2
   model:add(nn.ReLU())

--   model:add(nn.Reshape(64*3*3))
--   model:add(nn.Linear(64*3*3,256)) -- fully connected layer
   model:add(nn.Reshape(64*2*2)) -- reshape tensor with size 64*2*2 into 1D vector with length 64*2*2
   model:add(nn.Linear(64*2*2,256)) -- fully connected layer
   model:add(nn.ReLU())
   model:add(nn.Dropout(0.5))

   model:add(nn.Linear(256,10))
   model:add(nn.LogSoftMax())

   return model;
end
--- model ---
local model = buildModel()
--- verbose
if rank == 1 then
   print('<cifar> using model:')
   print(model)
end
--- Define loss function
criterion = nn.CrossEntropyCriterion()
state.theta,state.grad = model:getParameters() -- param and grad

-------------------------------------------------------------------
-- remember to reset data_root
--------------------------------------------------
--train_data = torch.load(train_bin) -- training data
--test_data = torch.load(test_bin)   -- test data
--local dim = train_data['data']:size(2)*
--            train_data['data']:size(3)*
--	    train_data['data']:size(4)
--local trsize = train_data['data']:size(1)
--local ttsize = test_data['data']:size(1)
--train_data.data:resize(trsize,dim)
--test_data.data:resize(ttsize,dim)
trsize = nil
tesize = nil
if full then
   trsize = 50000
   tesize = 10000
else
   trsize = 2000
   tesize = 1000
end

trainData = {
   data = torch.Tensor(50000, 3*32*32),
   labels = torch.Tensor(50000),
   size = function() return trsize end
}
for i = 0,4 do
   subset = torch.load('../cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t():float()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end
trainData.labels = trainData.labels + 1

subset = torch.load('../cifar-10-batches-t7/test_batch.t7', 'ascii')
testData = {
   data = subset.data:t():float(),
   labels = subset.labels[1]:float(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

--- resize dataset (if using small version)
trainData.data = trainData.data[{ {1,trsize} }]
trainData.labels = trainData.labels[{ {1,trsize} }]
testData.data = testData.data[{ {1,tesize} }]
testData.labels = testData.labels[{ {1,tesize} }]

--- Normalise to [0,1] ---
trainData.data = trainData.data:div(255)
testData.data = testData.data:div(255)

----------------------------------------------------------------------
print '===> defining some tools'

-- classes
classes = {'1','2','3','4','5','6','7','8','9','0'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- reshape to 4-D tensor ---
trainData.data = trainData.data:reshape(trsize,3,32,32)
testData.data = testData.data:reshape(tesize,3,32,32)
--
--trainData.labels = trainData.labels:float()
--testData.labels = testData.labels:float()
-------------------------------------------------------------------
require 'optim'
require 'xlua'    -- xlua provides useful tools, like progress bars
local opti
if optname == 'sgd' then
   print('sgd')
   opti = optim.msgd
   state.optim = {
      lr = lr,
      mommax = mom,      
   }
elseif optname == 'downpour' then
   print('downpower')
   opti = optim.downpour
   state.optim = {
      lr = lr,
      pclient = pclient,
      su = su,      
   }
elseif optname == 'eamsgd' then
   print('eamsgd')
   opti = optim.eamsgd
   state.optim = {
      lr = lr,
      pclient = pclient,
      su = su,
      mva = mva,
      mom = mom,
      rank = rank,
   }
end
-------------------------------------------------------------------
print('i am ' .. rank .. ' ready to run')
if pclient then
   pclient:start(state.theta,state.grad)
   assert(rank == pclient.rank)
   print('pc ' .. rank .. ' started')
end
-------------------------------------------------------------------
require 'image' -- image.crop for training & testing

--- training ---
local w = 32 -- raw weight and height of cifar-10 image
local h = 32
local tw = 28 -- after random/center corp image size in training/testing
local th = 28
local function train()
   sys.tic()
   local avg_err = 0
   local iter = 0
   for epoch = 1,maxep do
      --- shuffle at each epoch ---
      shuffle = torch.randperm(trsize)
      print('===> doing epoch on training data:')
      print("===> online epoch #" .. epoch .. '/' .. maxep .. ' [batchSize = ' .. mb .. ']')
      for t = 1,trsize,mb do
         -- disp progress
         xlua.progress(t, trsize)
         -- prepare mini batch
         local mbs = math.min(trsize-t+1,mb)
         local inputs = {} -- clear last mini-batch
         local targets = {}
         for j = t,t+mbs-1 do
            -- load new sample
            local input = trainData.data[shuffle[j]]
            local x1, y1 = torch.random(0, w - tw), torch.random(0, h - th)
            input = image.crop(input, x1, y1, x1 + tw, y1 + th) -- random batch
            assert(input:size(2) == tw and input:size(3) == th, 'wrong crop size')

            local target = trainData.labels[shuffle[j]]
            table.insert(inputs, input)
            table.insert(targets, target)
         end
         --- create closure to evaluate f(X) and df/dX
         local feval = function(x)
            local time_feval = sys.clock()
            -- get new parameters
            if x ~= state.theta then
               print('copy theta!!')
               state.theta:copy(x)
            end
            -- reset gradients
            state.grad:zero()
            -- f is the average of all criterions
            local f = 0
            --- evaluate function for complete mini batch
            for i = 1, #inputs do
               -- estimate f
               local output = model:forward(inputs[i])
               local err = criterion:forward(output, targets[i])
               f = f + err

               -- estimate df/dw
               local df_do = criterion:backward(output, targets[i])
               model:backward(inputs[i], df_do)

               -- update confusion
               confusion:add(output, targets[i])
            end

            -- normalize gradients and f(X)
            state.grad:div(#inputs)
            f = f/#inputs

            avg_err = avg_err + f
            tm.feval = tm.feval + (sys.clock() - time_feval)
            -- return f and df/dX
            return f,state.grad
         end

         --- optimize on current mini-batch
         local x,fx
         x,fx = opti(feval, state.theta, state.optim)
         -- increase iteration count
         iter = iter + 1
      end
      -- print confusion matrix
      print(confusion)

      -- update logger/plot
      trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
      if opt.plot then
         trainLogger:style{['% mean class accuracy (train set)'] = '-'}
         trainLogger:plot()
      end

      --- save model every
      if epoch % saveep == 0 then
         print('saveing model at epoth ' .. epoch)
         modelname = 'model-epoch-' .. epoch .. '.net'
         local filename = paths.concat(opt.save, modelname)
         os.execute('mkdir -p ' .. sys.dirname(filename))
         print('==> saving model to '..filename)
         torch.save(filename, model)
      end
      confusion:zero()
      print(io.popen('hostname -s'):read(),sys.toc(),rank,
        'avg_err at epoch ' .. epoch .. ' is ' .. avg_err / iter)
   end
end
-------------------------------------------------------------------
--- testing ---
--- report accuracy on the test data
local function test()
   -- local vars
   local time = sys.clock()

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('===> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      -- get new sample
      local input = testData.data[t]:float()
      local w1 = math.ceil((w - tw)/2)
      local h1 = math.ceil((h - th)/2)
      input = image.crop(input, w1, h1, w1 + tw, h1 + th) -- center patch
      local target = testData.labels[t]

      -- test sample
      local pred = model:forward(input)
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n===> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- next iteration:
   confusion:zero()
end
-------------------------------------------------------------------

train()

if pclient then
   pclient:stop()
end
local time = sys.toc()
print(rank,'total training time is', time)
print(rank,'total function eval time is', tm.feval)
print("\n===> time to learn 1 sample = " .. (time/trsize*1000) .. 'ms')
if state.optim.dusync then
   tm.sync = state.optim.dusync
end
print(rank,'total sync time is', tm.sync)
