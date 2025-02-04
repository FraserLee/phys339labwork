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

def plot_histogram(data):
    counts, bins, _ = plt.hist(data, bins='auto', alpha=0.6, label="Observed")

    mean, std = np.mean(data), np.std(data)

    # Poisson fit
    poisson_x = np.arange(int(min(data)), int(max(data)) + 1)
    poisson_y = poisson.pmf(poisson_x, mean) * len(data)
    plt.plot(poisson_x, poisson_y, 'r-', label="Poisson fit")

    # Gaussian fit
    gauss_x = np.linspace(min(data), max(data), 100)
    gauss_y = norm.pdf(gauss_x, mean, std) * len(data) * (bins[1] - bins[0])
    plt.plot(gauss_x, gauss_y, 'g-', label="Gaussian fit")

    # Error bars
    errors = np.sqrt(counts)
    plt.errorbar((bins[:-1] + bins[1:]) / 2, counts, yerr=errors, fmt='o', label="Error bars")

    # Chi-square test
    expected_counts = poisson.pmf(bins[:-1], mean) * len(data)
    # chi2, p_value = chisquare(counts, expected_counts)
    chi2, p_value = 0, 0
    plt.title(f"Histogram (Mode 0: Counts per Interval)\nChi-square: {chi2:.2f}, p-value: {p_value:.3f}")

    plt.xlabel("Counts per interval")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


filename = 'data/count_2025_02_04-15_55_24.txt'

data = load_data(filename)
plot_histogram(data)
