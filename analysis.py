import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, chisquare


def load_data(filename):
    values = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if not line.startswith('count (1000ms): '):
                print('discarding abnormal reading[', line, ']')
                continue

            values.append(int(line.split(':')[1].strip()))

    return np.array(values)

# def plot_histogram(data):
#
#     counts, bins, _ = plt.hist(data, bins='auto', alpha=0.6, label="Observed")
#
#     mean, std = np.mean(data), np.std(data)
#
#     # Poisson fit
#     poisson_x = np.arange(int(min(data)), int(max(data)) + 1)
#     poisson_y = poisson.pmf(poisson_x, mean) * len(data)
#     plt.plot(poisson_x, poisson_y, 'r-', label="Poisson fit")
#
#     # Gaussian fit
#     gauss_x = np.linspace(min(data), max(data), 100)
#     gauss_y = norm.pdf(gauss_x, mean, std) * len(data) * (bins[1] - bins[0])
#     plt.plot(gauss_x, gauss_y, 'g-', label="Gaussian fit")
#
#     # Error bars
#     errors = np.sqrt(counts)
#     plt.errorbar((bins[:-1] + bins[1:]) / 2, counts, yerr=errors, fmt='o', label="Error bars")
#
#     # Chi-square test
#     expected_counts = poisson.pmf(bins[:-1], mean) * len(data)
#     # chi2, p_value = chisquare(counts, expected_counts)
#     chi2, p_value = 0, 0
#     plt.title(f"Histogram (Mode 0: Counts per Interval)\nChi-square: {chi2:.2f}, p-value: {p_value:.3f}")
#
#     plt.xlabel("Counts per interval")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.show()


filename = 'data/count_2025_02_04-15_55_24.txt'
import sys
if len(sys.argv) > 1:
    filename = sys.argv[1]

data = load_data(filename)
print(len(data), 'data points loaded')

num_bins = np.max(data) + 1
x = np.unique(data)

# split the data into num_splits distinct sub-ranges, estimating the uncertainty
# on each bin by the spread of the data in that sub-range

num_splits = int(len(data) ** 0.5) # not sure if this is mathematically
                                   # justified, but since we're estimating the
                                   # uncertainty *of the mean*, results don't
                                   # change with the number of splits
print('sub-sections:', num_splits)

observations = []

for replica in np.array_split(data, num_splits):

    # get the counts of values in the replica
    counts = np.unique(replica, return_counts=True)
    # print(', '.join(map(lambda x: f'{x:02}', counts[0])))
    # print(', '.join(map(lambda x: f'{x:02}', counts[1])))

    # set 0 for any missing values (present in x, but not in replica)
    for i in range(len(x)):
        if x[i] not in counts[0]:
            counts = (np.insert(counts[0], i, x[i]), np.insert(counts[1], i, 0))

    normalised_counts = (counts[0], counts[1] / sum(counts[1]))
    observations.append(normalised_counts[1])

fig, axs = plt.subplots(2)

# draw black dotted line at 0
axs[1].axhline(0, color='black', linestyle='--', linewidth=1)

def mean_and_uncertainty(confidence):
    observed_mean = np.mean(observations, axis=0)
    observed_mean_uncertainty = np.std(observations, axis=0) / np.sqrt(num_splits)
    observed_mean_uncertainty *= norm.ppf(0.5 + confidence / 2) - norm.ppf(0.5 - confidence / 2)
    return observed_mean, observed_mean_uncertainty

# plot the observed data, with error bars derived from the spread per sub-range
confidences = [0.68, 0.95]
colours = ['#2d494f', '#5f7c82']
for i in range(len(confidences) - 1, -1, -1):
    confidence = confidences[i]
    observed_mean, observed_mean_uncertainty = mean_and_uncertainty(confidence)

    bar_width = 2 + 8 * (1 - confidence)

    axs[0].errorbar(x, observed_mean,
                 yerr=observed_mean_uncertainty,
                 elinewidth=bar_width,
                 capsize=2,
                 linestyle='None',
                 color=colours[i],
                 label=f"Observed Data ({confidence:.0%} confidence)")

residuals_zero, residuals_one = mean_and_uncertainty(0.68)

# fit a poisson distribution to the data, with mean taken from the total data
poisson_x = x
poisson_y = poisson.pmf(poisson_x, np.mean(data))
params = {'label': "Poisson fit", 'color': '#f2a900', 'linewidth': 1.5}
axs[0].plot(poisson_x, poisson_y, **params)
axs[1].plot(poisson_x, (poisson_y - residuals_zero) / residuals_one, marker='o', markersize=3, **params)

# fit a gaussian distribution to the data, with mean and std taken from the total data
mean, std = np.mean(data), np.std(data)
# gauss_x = np.linspace(0, num_bins, 1000)
gauss_x = x
gauss_y = norm.pdf(gauss_x, mean, std)
params = {'label': "Gaussian fit", 'color': '#de5823', 'linewidth': 1.5}
axs[0].plot(gauss_x, gauss_y, **params)
axs[1].plot(gauss_x, (gauss_y - residuals_zero) / residuals_one, marker='o', markersize=3, **params)


axs[1].set_xlabel("Clicks per 1000ms")
axs[0].set_ylabel("Probability [0-1]")
axs[1].set_ylabel("Curve Residuals (σ)")

# set x ticks to be integers
for ax in axs:
    ax.set_xticks(x)
    ax.set_xticklabels(map(str, x))

axs[1].set_ylim(-5, 5)
axs[1].set_yticklabels(map(lambda x: f"{int(x)}σ", axs[1].get_yticks()))

axs[0].legend()

# plt.title(f"Clicks per 1000ms\n{len(data)} data points split into {num_splits} replicas with {len(data) // num_splits} points each")
axs[0].set_title(f"Clicks per 1000ms\n{len(data)} data points split into {num_splits} replicas with {len(data) // num_splits} points each",
    fontsize=10)

plt.show()



# NOTES FROM Talking with Prof. Ryan
# - Error bars are probably too big, check that out
# - Don't normalise data, keep batch size and number of batches ("Replicas") in the report
# - Actually fit both Poisson and Gaussian distributions, give them a fighting chance
# - Don't use datapoints for which we don't have a count
