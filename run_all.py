import os
import sys

import pandas as pd

df = pd.read_excel(sys.argv[1], header=1).reset_index(drop=True)

for idx, row in df.iterrows():
    os.system(f"python fourier_analysis.py ../tephi_plot/settings/{row.date.year}{row.date.month:02d}{row.date.day:02d}_{row.hour}.json {row.region}")
