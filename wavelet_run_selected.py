import numpy as np
import os
import sys

import pandas as pd

df = pd.read_excel(sys.argv[1], index_col=[0,1,2], parse_dates=[0])
try:
    leadtime = sys.argv[2]
except IndexError:
    leadtime = 0
    print("No leadtime given, setting to 0")

for idx, row in df.iterrows():
    if idx[2] is not np.nan:
        regions = idx[2].split('/')
        hour = int(idx[1])

        for region in regions:
            if row['selected'] == 'x':
                command = f"python wavelet_analysis.py {idx[0].date()}_{hour:02d} {leadtime} {region}"
                os.system(command)