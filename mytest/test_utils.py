import numpy as np


def print_stat_info(data):
    print(
        data.max(),
        data.min(),
        np.percentile(data, 1),
        np.percentile(data, 5),
        np.percentile(data, 25),
        np.percentile(data, 50),
        np.percentile(data, 75),
        np.percentile(data, 95),
        np.percentile(data, 99),
    )
