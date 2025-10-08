import csv
import matplotlib.pyplot as plt
import numpy as np
from reg_model.linereg import LineReg
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent
DATA_PATH = BASE_PATH / 'bmw_saless.csv'


file = 'bmw_saless.csv'

sales, price = [],[]

try:
    with open(DATA_PATH, newline="") as csvfile:
        data = csv.DictReader(csvfile)
        for row in data:
            price.append(row['Price_USD'])
            sales.append(row['Sales_Volume'])
        print('import successfull')
except Exception as e:
    print(f'not working: {e}')

data = np.column_stack(
    [np.array(price, dtype=np.float64), 
     np.array(sales, dtype = np.float64)] ) # (50,000x2) matrix instead of 
#                                                          pandas table.
np.random.seed(32131)

samp = np.random.choice(data[:,0], size=5000)
fig, (ax1,ax2) = plt.subplots(2,1)

ax1.hist(samp, bins=300, label='500 bins', color='red', linewidth=0.5)
ax1.set_ylabel('Frequencies')
ax1.set_xlabel('Price')
ax1.legend()

ax2.scatter(data[:,0][:100], data[:,1][:100], label='Sales w.r.t Price', color='red')

ax2.set_ylabel('Sales')
ax2.set_xlabel('Price')
#ax2.set_ylim(0,100000)
#ax2.set_xlim(0,100000)

ax2.legend()

model = LineReg(data)
params = model.linear_ests()
resid = model.resid()
anova = model.anova()
summ = model.summary()
# mean of residuals 
# -2.6542693376541137e-13




