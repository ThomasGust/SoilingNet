import matplotlib.pyplot as plt
import pickle as pkl
import os


#TODO add functionality for iou metric when I add that to training loop
def compile_graphs_segmentation():
    model_names = [("FCN32-20", "UNet-20", "SegNet-20"), ("FCN32-50", "UNet-50", "SegNet-50"), ("FCN32-70", "UNet-70", "SegNet-70")]
    stats = "training_stats"
    clr_dict = {0:"r", 1:"g", 2:"b"}

    for ii, name in enumerate(model_names):
        red, green, blue = name

        names = [red, green, blue]
        
        for i, n in enumerate(names):

            with open(os.path.join(stats, n, "accuracies.pkl"), "rb") as f:
                accuracies = pkl.load(f)
            
            with open(os.path.join(stats, n, "losses.pkl"), "rb") as f:
                losses = pkl.load(f)

            with open(os.path.join(stats, n, "ious.pkl"), "rb") as f:
                jaccard_indices = pkl.load(f)
            
            plt.plot(range(0, len(losses)), losses, clr_dict[i], label=f'Loss {n}')
            plt.legend(loc='upper right')
            plt.title(f'Training Loss {n}')

            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(f'figures\\losses\\trainingLoss{n}.png')
            plt.close()

            plt.plot(range(0, len(accuracies)), accuracies, clr_dict[i], label=f'Accuracy {n}')
            plt.legend(loc='lower right')
            plt.title(f'Training Accuracy {n}')

            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.savefig(f'figures\\accuracies\\TrainingAccuracy{n}.png')
            plt.close()

            plt.plot(range(0, len(accuracies)), jaccard_indices, clr_dict[i], label=f'Accuracy {n}')
            plt.legend(loc='lower right')
            plt.title(f'Jaccard Index {n}')

            plt.xlabel("Epoch")
            plt.ylabel("Jaccard Index")
            plt.savefig(f'figures\\accuracies\\TrainingJaccardIndex{n}.png')
            plt.close()
        


        for i, n in enumerate(names):
            with open(os.path.join(stats, n, "accuracies.pkl"), "rb") as f:
                accuracies = pkl.load(f)
            
            plt.plot(range(0, len(accuracies)), accuracies, clr_dict[i], label=f"Accuracy {n}")
        
        plt.legend(loc='lower right')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.savefig(f"figures\\accuracies\\TrainingAccuracyCombined{ii}.png")
        plt.close()

        for i, n in enumerate(names):
            with open(os.path.join(stats, n, "losses.pkl"), "rb") as f:
                losses = pkl.load(f)
            
            plt.plot(range(0, len(losses)), losses, clr_dict[i], label=f"Loss {n}")
        
        plt.legend(loc='upper right')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.savefig(f"figures\\losses\\TrainingLossCombined{ii}.png")
        plt.close()

        for i, n in enumerate(names):
            with open(os.path.join(stats, n, "ious.pkl"), "rb") as f:
                accuracies = pkl.load(f)
            
            plt.plot(range(0, len(accuracies)), accuracies, clr_dict[i], label=f"Jaccard Index {n}")
        
        plt.legend(loc='lower right')
        plt.xlabel("Epoch")
        plt.ylabel("Jaccard Index")

        plt.savefig(f"figures\\accuracies\\TrainingJaccardIndexCombined{ii}.png")
        plt.close()

        
def compile_graphs_classification():

    classes_dict = {4:'r', 8:'g', 12:'b', 16:'y'}
    combined_dict = {0:0, 5:1, 7:2}


    classes = [4, 8, 12, 16]
    splits = [0]

    configs = []
    
    epochs_range = range(0, 100)

    for c in classes:
        for s in splits:
            configs.append((c, s))
    

    pth = "training_stats"

    combined_list = []

    for i, (c, s) in enumerate(configs):
        hist_pth = os.path.join(pth, f"ClassifierModel_{c}_{s}","hist.pkl")


        with open(hist_pth, "rb") as f:
            hist = pkl.load(f)
        
        loss_hist, acc_hist, time_hist, test_accuracy = hist[0], hist[1], hist[2], hist[3]
        
        color = classes_dict[c]
        combined_list.append(combined_dict[s])

        plt.plot(epochs_range, loss_hist, color, label=f"Loss Classifier-{c}-{s}")
        plt.legend(loc='upper right')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.savefig(f"figures\\classification_losses\\ClassifierTrainingLoss{c}{s}.png")
        plt.close()

        plt.plot(epochs_range, acc_hist, color, label=f"Accuracy Classifier-{c}-{s}")
        plt.legend(loc='lower right')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

        plt.savefig(f"figures\\classification_accuracies\\ClassifierTrainingAccuracy{c}{s}.png")
        plt.close()
    
    combined_configs = [[] for c in classes]

    for i, config in enumerate(configs):
        combined_configs[combined_list[i]].append(config)

    for i, config_list in enumerate(combined_configs):
        for c, s in config_list:

            with open(f'training_stats\\ClassifierModel_{c}_{s}\\hist.pkl', "rb") as f:
                hist = pkl.load(f)
            
            loss_hist, acc_hist, time_hist, test_accuracy = hist[0], hist[1], hist[2], hist[3]

            plt.plot(epochs_range, loss_hist, classes_dict[c], label=f"Loss Classifier-{c}-{s}")
        plt.legend(loc='upper right')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"figures\\classification_losses\\CombinedClassifierTrainingLoss{i}.png")
        plt.close()

        for c, s in config_list:

            with open(f'training_stats\\ClassifierModel_{c}_{s}\\hist.pkl', "rb") as f:
                hist = pkl.load(f)
            
            loss_hist, acc_hist, time_hist, test_accuracy = hist[0], hist[1], hist[2], hist[3]

            plt.plot(epochs_range, acc_hist, classes_dict[c], label=f"Accuracy Classifier-{c}-{s}")
        
        plt.legend(loc='lower right')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")


        plt.savefig(f"figures\\classification_accuracies\\CombinedClassifierTrainingAccuracy{i}.png")
        plt.close()




if __name__ == "__main__":
    compile_graphs_classification()