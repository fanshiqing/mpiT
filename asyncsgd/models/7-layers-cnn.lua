-------------------------------------------------------------------
-- Author: Shiqing Fan (sqfan6@gmail.com)
-------------------------------------------------------------------
require 'nn'

model = nn.Sequential()

model:add(nn.SpatialConvolution(3,64,5,5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))

model:add(nn.SpatialConvolution(64,128,5,5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))

model:add(nn.SpatialConvolution(128,64,3,3))
model:add(nn.ReLU())

model:add(nn.View(64*2*2))
model:add(nn.Linear(64*2*2,256))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))

model:add(nn.Linear(256,10))
model:add(nn.LogSoftMax())

return model;
