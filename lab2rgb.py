from colormath.color_objects import LabColor, XYZColor,sRGBColor
from colormath.color_conversions import convert_color
import numpy as np

def lab2rgb(lab_color):

    lab = LabColor(lab_color[0], lab_color[1], lab_color[2])
    xyz = convert_color(lab, sRGBColor)

    # print(xyz)

    a = np.array(xyz.get_upscaled_value_tuple())
    # print(a,a[0])
    return a