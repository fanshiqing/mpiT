-- mpi launch
-- Author: Sixin Zhang (zsx@cims.nyu.edu)
-- Author: Shiqing Fan (sqfan6@gmail.com)
-- mpirun -n 12 luajit mlaunch.lua

--[[
--
   This script is used to launch mpi.  The user's script goes at the bottom, where
   goot.lua has been referenced.   Edit this file only to change the variables
   noted.

   MPI is configured so this script will be running separately on each available
   core on each available machine in the cluster - so the 'ranks' below will range
   from 0-7 if there are two machines with 4 CPU cores each.  I don't have any
   GPUs, so I can't speak to how those are presented.

   It is assumed the EAMSGD optimizer is desired.
 ]]
local oncuda = false

local AGPU = nil
if oncuda then
   require 'cutorch'
   AGPU = {1,2,3,4,5,6} -- use the first 6 gpus on each machine
end

dofile('init.lua')
mpiT.Init()

local world = mpiT.COMM_WORLD
local rank = mpiT.get_rank(world) -- mpi rank
local size = mpiT.get_size(world) -- num of mpi processes
local gpuid = -1

local conf = {}
conf.rank = rank
conf.world = world
conf.sranks = {} -- server ranks
conf.cranks = {} -- client ranks
for i = 0,size-1 do
   if math.fmod(i,2)==0 then -- if node rank is even, it's a server
      table.insert(conf.sranks,i)
   else
      table.insert(conf.cranks,i)
   end
end

opt = {}
opt.data_root = io.popen('echo $HOME'):read() .. '/intel/mpiT/asyncsgd/cifar-10-batches-t7'
--[[
opt.name = 'downpour'
opt.lr = 1e-4
opt.su = 1
]]
opt.name = 'eamsgd'
opt.su = 2 -- communication period
opt.p = size/2
opt.mva = 0.9/opt.p -- this is \beta when there are opt.p workers

opt.lr = 1e-2
opt.mom = 0.99

opt.maxepoch = 100

opt.save = 'results' -- results save path
opt.full = true -- use full dataset
opt.plot = true -- live plot!!
if math.fmod(rank,2)==0 then
   -- server (always on CPU!!!)
   print('[server] rank',rank,'use cpu')
   torch.setdefaulttensortype('torch.FloatTensor')
   --- new param server class
   local ps = pServer(conf)
   ps:start()
else
   if AGPU then
      assert(false)
--      -- if GPU is enabled, param clients are all on GPU!!!
--      require 'cunn'
--      local gpus = cutorch.getDeviceCount()
--      gpuid = AGPU[(rank%(size/2)) % gpus + 1]
--      cutorch.setDevice(gpuid)
--      print('[client] rank ' .. rank .. ' use gpu ' .. gpuid)
--      torch.setdefaulttensortype('torch.CudaTensor')
   else
      print('[client] rank ' .. rank .. ' use cpu')
      torch.setdefaulttensortype('torch.FloatTensor')
   end
   -- setup
   opt.gpuid = gpuid
   --- new param client class
   opt.pc = pClient(conf)
   opt.rank = rank
   -- go
   dofile('goot.lua')
end

-- clean up the MPI communication channels.
mpiT.Finalize()
