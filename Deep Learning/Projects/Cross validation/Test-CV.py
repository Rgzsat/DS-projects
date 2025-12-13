import os
from pathlib import Path
import numpy as np
import pandas as pd
import glob

path = # use your path
#all_files = glob.glob(os.path.join(path , "/*.csv"))
files = Path(path).glob('*.csv')  # .rglob to get subdirectories
from sklearn.preprocessing import MinMaxScaler

