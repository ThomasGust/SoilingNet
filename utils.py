import numpy as np
import matplotlib.pyplot as plt

def get_confusion_matrix(predictions, labels, per=True):
    K = len(np.unique(labels))
    result = np.zeros((K, K))

    for i in range(len(labels)):
        result[labels[i]][predictions[i]] += 1
    
    if per:
        result /= result.astype(np.float).sum(axis=0)
        
        """
        for i in range(np.shape(result)[0]):
            for ii in range(np.shape(result)[1]):
                result[i][ii] = f'{result[i][ii]:.2f}'
        """
    
    return result

def plot_confusion_matrix(conf_matrix, c, s):
    c_dict = {4:'x-large', 8:'large', 12:'medium', 16:'x-small'}
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, alpha=0.7)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=f'{str(conf_matrix[i, j]*100):.4}%', va='center', ha='center', size=c_dict[c])
    
    plt.xlabel(f'Confusion Matrix Classifier Model-{c}-{s}', fontsize=18)
    plt.ylabel('Labels', fontsize=18)
    plt.title('Predictions', fontsize=18)
    plt.savefig(f'figures\\confusion_matrices\\CONFUSIONMATRIX_{c}_{s}.png')
    plt.close()

