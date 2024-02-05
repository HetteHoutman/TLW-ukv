import os
import sys

import pandas as pd

df = pd.read_excel(sys.argv[1], index_col=[0,1,2], parse_dates=[0])
leadtime = sys.argv[2]

for idx, row in df.iterrows():
    os.system(
        f"python wavelet_analysis.py {idx[0].date()}_{idx[2]} {leadtime} {idx[1]}")
