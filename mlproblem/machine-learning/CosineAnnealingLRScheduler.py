import math


class CosineAnnealingLRScheduler:
    def __init__(self, initial_lr, T_max, min_lr):
        self.initial_lr = float(initial_lr)
        self.T_max = int(T_max)
        self.min_lr = float(min_lr)

    def get_lr(self, epoch):
        epoch = min(epoch, self.T_max)
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                1 + math.cos(math.pi * epoch / self.T_max)
        )
        return round(lr, 4)


scheduler = CosineAnnealingLRScheduler(initial_lr=0.1, T_max=10, min_lr=0.001)

print(f"{scheduler.get_lr(epoch=0):.4f}")  # 0.1000
print(f"{scheduler.get_lr(epoch=2):.4f}")  # 0.0905
print(f"{scheduler.get_lr(epoch=5):.4f}")  # 0.0505
print(f"{scheduler.get_lr(epoch=7):.4f}")  # 0.0214
print(f"{scheduler.get_lr(epoch=10):.4f}")  # 0.0010
