import numpy as np
# import matplotlib.pyplot as plt


def load_demand_records():
    np.random.seed(0)
    demand_hist = []
    for i in range(52):
        for j in range(4):
            random_demand = np.random.normal(3, 1.5)
            if random_demand < 0:
                random_demand = 0
            random_demand = np.round(random_demand)
            demand_hist.append(random_demand)
        random_demand = np.random.normal(6, 1)
        if random_demand < 0:
            random_demand = 0
        random_demand = np.round(random_demand)
        demand_hist.append(random_demand)
        for j in range(2):
            random_demand = np.random.normal(12, 2)
            if random_demand < 0:
                random_demand = 0
            random_demand = np.round(random_demand)
            demand_hist.append(random_demand)
    # plt.hist(demand_hist)

    return demand_hist


def convert_day_to_month_fraction():
    return None
