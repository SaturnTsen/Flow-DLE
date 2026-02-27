from .depth_estimation import *
from .vis_utils import *

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed+np.random.randint(0, 1000000))