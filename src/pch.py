import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataframe(dataset_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    dataset_path = os.path.join(parent_dir, f'./datasets/{dataset_name}')
    return pd.read_csv(dataset_path)
