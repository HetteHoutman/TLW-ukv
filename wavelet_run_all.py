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
            # only 2023 for now
            if idx[0].year == 2023 and region != 'none':
                # only run if not already run
                command = f"python wavelet_analysis.py {idx[0].date()}_{hour:02d} {leadtime} {region}"
                dir_name = f'./plots/{idx[0].date()}_{hour:02d}/{region}/'
                if os.path.isdir(dir_name):
                    if not os.listdir(dir_name):
                        print(f"Directory {dir_name} is empty")
                        os.system(command)
                    else:
                        pass
                else:
                    print(f"Directory {dir_name} does not exist")
                    os.system(command)
