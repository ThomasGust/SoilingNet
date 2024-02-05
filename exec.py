import numpy as np
from kerasegmentation import fcn32, put_pallete, resnetsegnet, resnetunet
import matplotlib.pyplot as plt

resnetsegnet.load_weights('segmenters_checkpoints\\segnet_20\\SEGNET.50')


for i in range(4):
    img = resnetsegnet.predict_segmentation(inp=f"examples\\inputs\\test{i+1}.png", out_fname=f'out{i+1}.png')
    plt.imshow(img)
    plt.show()
    put_pallete(img, f"out{i+1}")
