import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

def k_nearest_neighbor(points, new_point, k=1):
    dist = np.array([])    
    for i in range(len(points)):
        x = points.iloc[i].long_hair
        y = points.iloc[i].forehead_width_cm

        d = np.sqrt((new_point - x)**2)
        dist = np.append(dist, [{"dist": d, "class": points.iloc[i].gender}])
    
    return dist

data = pd.read_csv("gender_classification_v7.csv")
training_data = data[:150]


print(k_nearest_neighbor(training_data, data.iloc[155].long_hair, k=5))