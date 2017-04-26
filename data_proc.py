import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lab2rgb import lab2rgb

def data_proc():
    input_scan = pd.read_table('F:\Deep_learning_Nanodegree\workspace\color_trans_01\Input_scanned.txt',delim_whitespace=True, header=None)
    input_scan = input_scan.get_values()
    input_scan_rgb = input_scan[:,1:4]

    output_std = pd.read_table('F:\Deep_learning_Nanodegree\workspace\color_trans_01/target_standard.txt',delim_whitespace=True, header=None)
    output_std = output_std.get_values()
    output_std_lab = output_std[:,1:4]
    output_std_rgb = np.zeros_like(input_scan_rgb)

    # print(output_std_rgb)
    for i in range(1, 13):
        for j in range(1, 23):
            output_std_rgb[(j-1)*12+i-1,:] = lab2rgb(output_std_lab[(i-1)*22+j-1,:])

    return input_scan_rgb,output_std_rgb























