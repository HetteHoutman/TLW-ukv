import sys
import tephi
import matplotlib.pyplot as plt
import pandas as pd

txtfile = sys.argv[1]

station = txtfile[-20:-15]
year = txtfile[-14:-12]
month = txtfile[-12:-10]
day = txtfile[-10:-8]
time = txtfile[-7:-4]

print(f'Showing sounding from station {station} on {year}/{month}/{day} at {time}.')

data = pd.read_csv(txtfile, skiprows=[0,2,3], delim_whitespace=True)

tephi.MIN_THETA = -40

tpg = tephi.Tephigram(anchor=[(1050, -40), (200, -40)])
temp = tpg.plot(data[['PRES', 'TEMP']])
tpg.plot(data[['PRES', 'DWPT']])
temp.barbs(data[['SKNT', 'DRCT', 'PRES']], color= 'black', pivot='middle')

plt.tight_layout()
plt.savefig(f'plots/tephi_from_txt_{station}_{year}{month}{day}_{time}.png', dpi=300)
plt.show()