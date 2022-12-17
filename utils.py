import numpy as np
import matplotlib.pyplot as plt

def get_confusion_matrix(predictions, labels, per=True):
    K = len(np.unique(labels))
    result = np.zeros((K, K))

    for i in range(len(labels)):
        result[labels[i]][predictions[i]] += 1
    
    if per:
        result /= result.astype(np.float).sum(axis=1)
        
        for i in range(np.shape(result)[0]):
            for ii in range(np.shape(result)[1]):
                result[i][ii] = f'{result[i][ii]:.2f}'
    
    return result

def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, alpha=0.7)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Confusion Matrix', fontsize=18)
    plt.ylabel('Labels', fontsize=18)
    plt.title('Predictions', fontsize=18)
    plt.savefig('CONFUSIONMATRIX.png')
    plt.show()

