from datetime import datetime

from memory_profiler import profile
import torch


fp = open(f'{datetime.now()}.log', 'w+')


@profile(stream=fp)
def generate_data():
    large_image_set = torch.rand(500000, 3, 24, 24)
    large_image_set_targets = torch.rand(500000, 1000)

    small_image_set = torch.rand(50, 3, 24, 24)
    small_image_set_targets = torch.rand(50, 1000)

    generated_numbers = [i for i in range(1, 100000)]

    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    c = [3] * (2 * 10 ** 8)


if __name__ == '__main__':
    generate_data()
