-------------------------------------------------------------------
-- Author: Shiqing Fan (sqfan6@gmail.com)
-------------------------------------------------------------------
require 'nn'

model = nn.Sequential()
--model:add(nn.Reshape(3,28,28))

model:add(nn.SpatialConvolution(3,64,5,5)) -- 64*24*24
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2)) -- 64*12*12

model:add(nn.SpatialConvolution(64,128,5,5)) -- 128*8*8
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2)) -- 128*4*4

model:add(nn.SpatialConvolution(128,64,3,3)) -- 64*2*2
model:add(nn.ReLU())

model:add(nn.View(64*2*2)) -- reshape tensor with size 64*2*2 into 1D vector with length 64*2*2
model:add(nn.Linear(64*2*2,256)) -- fully connected layer
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))

model:add(nn.Linear(256,10))
model:add(nn.LogSoftMax())

return model;
