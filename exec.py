import numpy as np
from kerasegmentation import fcn32, put_pallete

fcn32.load_weights('segmenters_checkpoints\\fcn32_20\\FCN32.0')


for i in range(4):
    img = fcn32.predict_segmentation(inp=f"examples\\inputs\\test{i+1}.png", out_fname=f'out{i+1}.png')
    print(np.unique(img))
    put_pallete(img, f"out{i+1}")