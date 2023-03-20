import logging
import os

from rich.logging import RichHandler
from torch import distributed
from torch.utils.data import DataLoader
from tqdm.rich import tqdm

# logging part
log_level = logging.INFO
FORMAT = "%(message)s"
logging.basicConfig(
    level=log_level, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger(os.path.basename(__file__))


class TrainLogging:
    def __init__(
        self,
        local_rank=0,
        log_dir=None,
        experiment_name=None,
        logger="tensorboard",
        **kwargs,
    ) -> None:
        from torch.utils.tensorboard import SummaryWriter

        log_dir = os.path.join(log_dir, experiment_name)
        if not os.path.exists(log_dir) and local_rank == 0:
            os.makedirs(log_dir)
        elif local_rank == 0:
            for i in range(1000):
                new_dir = os.path.join(log_dir, f"version_{i}")
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                    break
            log_dir = new_dir
        if logger == "tensorboard":
            self.logger = SummaryWriter(log_dir=log_dir)
        elif logger == "wandb":
            raise NotImplementedError
        else:
            raise NotImplementedError
        self.metrics = {}  # type: ignore

    def add_images(self, key, img, step):
        self.logger.add_images(key, img, step)

    def add_scalar(self, key, scalar, step):
        self.logger.add_scalar(key, scalar, step)

    def log_metric(self, key, scalar):
        if key not in self.metrics:
            self.metrics[key] = AverageMeter()
        self.metrics[key].update(scalar)

    def eval_metric(self, key):
        return self.metrics[key].eval()

    def flush(self):
        self.logger.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, avg_size=100):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0
        self.loss_bottom = 0

        self.avg_size = avg_size
        self.losses = []  # kept latest losses
        self.reset()

    def reset(self):
        pass

    def update(self, val, n=1):
        if not isinstance(val, float):  # in case it is tensor
            val = val.item()
        self.val = val
        if len(self.losses) > self.avg_size:
            self.losses.pop()
        self.losses.insert(0, val)

    def eval(self):
        if len(self.losses) == 0:
            return 1000
        return sum(self.losses) / len(self.losses)
        # self.avg = sum(self.losses)/len(self.losses)


def get_current_time():
    from datetime import datetime, timedelta

    eight_hours_from_now = datetime.now() + timedelta(hours=8)
    dt_string = eight_hours_from_now.strftime("date:%d-%m-%Y_Time:%H:%M")
    return dt_string
