# scheduler that is during warmup steps at initial lr and then goes to final lr
from torch.optim.lr_scheduler import LambdaLR
class ChillScheduler(LambdaLR):
    pass


# scheduler that goes from initial lr during warmup to final lr and then stays there


class CooldownScheduler(LambdaLR):
    pass
    