import numpy as np
import matplotlib.pyplot as plt


def load_demand_records(seed=None):
    """Load the demand records with a specified seed."""
    if seed is None:
        seed = np.random.randint(0, 100)
    np.random.seed(seed)
    demand_hist = []

    # Define demand patterns for each season
    patterns = {
        'Winter': {'mean': 3, 'std': 1},
        'Spring': {'mean': 8, 'std': 1},
        'Summer': {'mean': 15, 'std': 1},
        'Fall': {'mean': 7, 'std': 1}
    }

    # Generate demand records for each season
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        for _ in range(91):  # 91 days for each season
            random_demand = np.random.normal(patterns[season]['mean'], patterns[season]['std'])
            random_demand = max(0, np.round(random_demand))  # Ensure demand is non-negative
            demand_hist.append(random_demand)

    return demand_hist


# # Generate demand record for 365 days
# demand_record = load_demand_records()
#
# # Plotting
# plt.figure(figsize=(15, 6))
# plt.plot(demand_record)
# plt.title("Demand Record for One Year (365 days)")
# plt.xlabel("Day")
# plt.ylabel("Demand")
# plt.grid(True)
# plt.show()
