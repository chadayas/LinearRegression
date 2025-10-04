import csv
import matplotlib.pyplot as plt


file = 'bmw_saless.csv'

sales, price = [],[]


with open('bmw_saless.csv', newline="") as csvfile:
    data = csv.DictReader(csvfile)
    for row in data:
       price.append(row['Price_USD'])
       sales.append(row['Sales_Volume'])

data = np.column_stack(
    [np.array(price, dtype=np.float64), 
     np.array(sales, dtype = np.float64)] ) # (50,000x2) matrix instead of 
#                                                          pandas table.












model = linereg(data)
params = model.linear_ests()
resid = model.resid()
anova = model.anova()
summ = model.summary()

print(model.compute_rstat())

