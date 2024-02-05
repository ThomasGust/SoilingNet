import os
from dataset import get_labels
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def generate_label_bar_plot(num_classes):
    labels, _ = get_labels(num_classes)

    unique = np.unique(np.array(labels))
    
    data = []

    for u in unique:
        data.append(labels.count(u))

    plt.bar(range(0, num_classes), data)
    plt.xticks(range(0, num_classes))
    plt.title(f"Classification Dataset Distribution ({num_classes})")
    plt.xlabel("Soiling Severity")
    plt.ylabel("Num Samples")
    plt.savefig(f"DatasetDistribution{num_classes}.png")
    plt.close()

def generate_irradiance_bar_plot(num_classes):
    _, labels = get_labels(num_classes)

    unique = np.unique(np.array(labels))
    
    data = [0 for i in range(num_classes)]

    for i, u in enumerate(unique):
        data[i] = labels.count(u)

    print(len(data))
    plt.bar(range(0, num_classes), data)
    plt.xticks(range(0, num_classes))
    plt.title(f"Classification Dataset Distribution ({num_classes})")
    plt.xlabel("Soiling Severity")
    plt.ylabel("Num Samples")
    plt.savefig(f"DatasetIrradianceDistribution{num_classes}.png")
    plt.close()

if __name__ == "__main__":
    generate_irradiance_bar_plot(4)
    generate_irradiance_bar_plot(8)
    generate_irradiance_bar_plot(12)
    generate_irradiance_bar_plot(16)