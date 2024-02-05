import matplotlib.pyplot as plt

def learning_rate(initial, decay, epochs):
    rates = []

    prev = 1
    for e in range(epochs):
        rate = initial*decay*prev
        prev = rate
        rates.append(rate)
    
    epoch_range = [e for e in range(epochs)]

    plt.plot(epoch_range, rates, 'g', label='LEARNING RATES')
    plt.title("Learning Rates")
    plt.xlabel("Epoch")
    plt.ylabel('Learning Rate')
    plt.savefig("LEARNINGRATES.png")

if __name__ == "__main__":
    learning_rate(0.07, 0.985, 100)