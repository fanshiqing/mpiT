-- cpu launch
-- Author: Sixin Zhang (zsx@cims.nyu.edu)
-- luajit claunch.lua
dofile('init.lua')

opt = {}
opt.gpuid = -1
torch.setdefaulttensortype('torch.FloatTensor')

opt.name = 'eamsgd'
opt.data_root='/home/sqfan/data/torch7/mnist10'
opt.lr = 1e-2           -- learning rate
opt.mva = opt.lr * 1e-4 -- moving rate
opt.su = 1              -- comm period
opt.mom = 0.99          -- momentum
opt.maxepoch = 100


dofile('goot.lua')
