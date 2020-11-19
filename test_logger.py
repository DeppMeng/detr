from util.logger import setup_logger
import torch.distributed as dist
from test_logger_2 import test_proc



logger = setup_logger(output='output/test_logger', distributed_rank=0, name="DETR", phase="train")
logger.info("hello")
test_proc()