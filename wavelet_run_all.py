import os
import sys

import pandas as pd

df = pd.read_excel(sys.argv[1], index_col=[0,1])

for idx, row in df.iterrows():
    os.system(
        f"python wavelet_analysis.py ./settings/{idx[0][:4]}{idx[0][5:7]}{idx[0][8:10]}_{row.hour}.json {idx[1]}")
