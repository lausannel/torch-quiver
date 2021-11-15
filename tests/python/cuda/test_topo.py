import quiver
import torch

world_size = torch.cuda.device_count()
device_list = list(range(world_size))

numa_topo = quiver.p2pCliqueTopo(device_list)
numa_topo.info()
