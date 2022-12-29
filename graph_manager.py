import matplotlib.pyplot as plt
import pickle as pkl
import os


model_names = [("FCN32-20", "UNet-20", "SegNet-20"), ("FCN32-50", "UNet-50", "SegNet-50"), ("FCN32-70", "UNet-70", "SegNet-70")]
stats = "training_stats"
clr_dict = {0:"r", 1:"g", 2:"b"}

def compile_graphs():
    for name in model_names:
        red, green, blue = name

        names = [red, green, blue]
        
        for i, n in enumerate(names):

            with open(os.path.join(stats, n, "accuracies.pkl"), "rb") as f:
                accuracies = pkl.load(f)
            
            with open(os.path.join(stats, n, "losses.pkl"), "rb") as f:
                losses = pkl.load(f)
            
            plt.plot(range(0, len(losses)), losses, clr_dict[i], label=f'Loss {n}')
            plt.title(f'Training Loss {n}')

            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(f'figures\\losses\\trainingLoss{n}.png')
            plt.close()

            plt.plot(range(0, len(accuracies)), accuracies, clr_dict[i], label=f'Accuracy {n}')
            plt.title(f'Training Accuracy {n}')

            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.savefig(f'figures\\accuracies\\TrainingAccuracy{n}.png')
            plt.close()
    
if __name__ == "__main__":
    compile_graphs()