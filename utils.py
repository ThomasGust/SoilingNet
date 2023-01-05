import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle as pkl
import torch
import torchmetrics
from torch.utils.data import DataLoader
from dataset import DynamicSolarPanelSoilingDataset
from torchvision import transforms
from kerasegmentation import resnetsegnet, resnetunet, fcn32
from sklearn.model_selection import train_test_split
import shutil
import random
from dataset import load_data, augment_data

def get_confusion_matrix(predictions, labels, per=True):
    K = len(np.unique(labels))
    result = np.zeros((K, K))

    for i in range(len(labels)):
        result[labels[i]][predictions[i]] += 1
    
    if per:
        result /= result.astype(np.float).sum(axis=0)
    
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

def add_reference(clean_reference_path='clean_reference_.png'):
    ref = cv2.imread(clean_reference_path)
    ref = cv2.resize(ref, dsize=(192, 192))

    for path in os.listdir("Dataset\\labels"):
        img = cv2.imread(os.path.join("Dataset", "labels", path))
        new_img = ref + img
        cv2.imwrite(os.path.join("Dataset", "labels", path), new_img)

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

def get_labels(n_classes, every):
    labels = []


    file_names = os.listdir("PanelImages")
    
    for name in file_names:
        smooth_label = float(name.split("_L_")[1].split("_I_")[0])
        hard_label = int(round(smooth_label*(n_classes-1)))
        labels.append(hard_label)

    return labels[::every]

def build_confusion_matrix(c, s, ds, every, device):
    model = torch.load(f'classifier_checkpoints\\ClassifierModel_{c}_{s}\\MODEL99.pt').to(device)

    dataloader = DataLoader(ds, 64)

    predictions = []
    _predictions = []
    labels = get_labels(c, every)

    with torch.no_grad():
        for i, (img, true, irradiance) in enumerate(dataloader):
            img = img.to(device)
            irradiance = irradiance.to(device)

            preds = model(img, irradiance)
            _preds = torch.softmax(preds, dim=1)
            preds = torch.argmax(_preds, dim=1)

            preds = preds.cpu().tolist()

            _preds = _preds.cpu().tolist()
            for p in _preds:
                _predictions.append(p)
            for p in preds:
                predictions.append(p)



    predictions = np.array(predictions)
    labels = np.array(labels)
    _predictions = np.array(_predictions)

    top_one_accuracy = torchmetrics.Accuracy(num_classes=c)(torch.tensor(predictions), torch.tensor(labels))
    top_two_accuracy = torchmetrics.Accuracy(num_classes=c, top_k=2)(torch.tensor(_predictions), torch.tensor(labels))

    print(top_one_accuracy)
    print(top_two_accuracy)

    cf = get_confusion_matrix(predictions, labels, per=True)
    plot_confusion_matrix(cf, c, s)

def generate_configs():
    cs = [4, 8, 12, 16]
    ss = [0]

    configs = []

    for c in cs:
        for s in ss:
            configs.append((c, s))
    
    return configs

def generate_confusion_matrices():
    every = 1
    device = torch.device('cuda:0')
    ds = DynamicSolarPanelSoilingDataset(4, "PanelImages", segmentation_model=None, every=every, format='PNG', transform=transforms.ToTensor())

    configs = generate_configs()

    for i, config in enumerate(configs):
        c, s = config
        build_confusion_matrix(c, s, ds, every, device)

        print(f"Generate confusion matrix for config {i+1}")

def put_pallete(img, path):
    t = torch.tensor(img)
    img = transforms.ToPILImage()(t.byte())
    img.putpalette([0,   0,   0,
                    255, 0, 0,
                    0, 255,   0,
                    0,   0, 255,
                    255,255,0,
                    0,255,255,
                    255,0,255,
                    255,255,255]) 
    filename = f"{path}.png"
    img.save(filename)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    return int(hours), int(mins), sec

def load_completed_models():
    resnetsegnet.load_weights("segmenters_checkpoints\\segnet_20\\SEGNET.99")
    resnetunet.load_weights("segmenters_checkpoints\\unet_20\\UNET.99")
    fcn32.load_weights("segmenters_checkpoints\\fcn32_20\\FCN32.99")

def tts(path, splits=[0.2, 0.5, 0.7]):
    if not os.path.exists("SplitDatasets"):
        os.mkdir("SplitDatasets")
    
    img_names = os.listdir(os.path.join(path, 'images'))
    label_names = os.listdir(os.path.join(path, "labels"))

    for i, s in enumerate(splits):

        X_train, X_test, y_train, y_test = train_test_split(img_names, label_names, random_state=314, test_size=s,shuffle=True)
        per = int(s*100)

        if os.path.exists(f"SplitDatasets\\Dataset{per}"):
            shutil.rmtree(f"SplitDatasets\\Dataset{per}")
        
        os.mkdir(f"SplitDatasets\\Dataset{per}")
        os.mkdir(f"SplitDatasets\\Dataset{per}\\train")
        os.mkdir(f"SplitDatasets\\Dataset{per}\\train\\images")
        os.mkdir(f"SplitDatasets\\Dataset{per}\\test")
        os.mkdir(f"SplitDatasets\\Dataset{per}\\test\\images")
        os.mkdir(f"SplitDatasets\\Dataset{per}\\train\\labels")
        os.mkdir(f"SplitDatasets\\Dataset{per}\\test\\labels")

        #train_dataset = training_datasets[i]
        test_dataset = zip(X_test, y_test)
        train_dataset = zip(X_train, y_train)
    
        for sample in train_dataset:
            img_name, label_name = sample

            img_path_src = os.path.join(path, "images", img_name)
            label_path_src = os.path.join(path, "labels", label_name)

            img_path_dst = os.path.join(f"SplitDatasets\\Dataset{per}\\train\\images", img_name)
            label_path_dst = os.path.join(f"SplitDatasets\\Dataset{per}\\train\\labels", label_name)

            img_img = cv2.imread(img_path_src)
            img_label = cv2.imread(label_path_src, cv2.IMREAD_UNCHANGED)

            cv2.imwrite(img_path_dst, img_img)
            cv2.imwrite(label_path_dst, img_label)
            print(np.unique(img_label))
            #shutil.copyfile(img_path_src, img_path_dst)
            #shutil.copyfile(label_path_src, label_path_dst)
        
        for sample in test_dataset:
            img_name, label_name = sample
            img_path_src = os.path.join(path, "images", img_name)
            label_path_src = os.path.join(path, "labels", label_name)
            img_path_dst = os.path.join(f"SplitDatasets\\Dataset{per}\\test\\images", img_name)
            label_path_dst = os.path.join(f"SplitDatasets\\Dataset{per}\\test\\labels", label_name)
            img_img = cv2.imread(img_path_src)
            img_label = cv2.imread(label_path_src, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(img_path_dst, img_img)
            cv2.imwrite(label_path_dst, img_label)

def random_colorize(i, mod, n):
    num = random.randint(0, len(os.listdir("PanelImages")))

    img_path = os.path.join("PanelImages", os.listdir("PanelImages")[num])
    src_img = cv2.imread(img_path)
    mask = mod.predict_segmentation(img_path)
    print(np.unique(mask))
    cv2.imwrite(f"{n}IMG{i}.png", src_img)
    
    put_pallete(mask, f"{n}OUT{i}")

def apply_data_augmentation():
    shutil.rmtree("AugmentedDataset\\images")
    shutil.rmtree("AugmentedDataset\\labels")
    os.mkdir("AugmentedDataset\\images")
    os.mkdir("AugmentedDataset\\labels")
    images, masks = load_data("Dataset")
    augment_data(images, masks, "AugmentedDataset")

def predict_test_data(model):
    for i in range(4):
        img = model.predict_segmentation(inp=f"examples\\inputs\\test{i+1}.png", out_fname=f'out{i+1}.png')
        plt.imshow(img)
        plt.show()
        put_pallete(img, f"out{i+1}")

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