import os
from pathlib import Path
import numpy as np
import pandas as pd
import glob

path = # use your path
#all_files = glob.glob(os.path.join(path , "/*.csv"))
files = Path(path).glob('*.csv')  # .rglob to get subdirectories
from sklearn.preprocessing import MinMaxScaler

datasets = list()
for f in files:
    data = pd.read_csv(f, index_col=6)
    # .stem is method for pathlib objects to get the filename w/o the extension
    #data['file'] = f.stem
    datasets.append(data)

