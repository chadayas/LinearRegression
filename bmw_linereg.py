import numpy as np 
import csv
import json
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
     np.array(sales, dtype = np.float64)] ) # (50,000x2) matrix instead of pandas table.
'''
plt.scatter(data[:,0][:100] , data[:, 1][:100],)
plt.xlabel('price')
plt.ylabel('sales')
plt.show()
'''

class linereg():
    def __init__(self, data):
        self.data = data
    def linear_ests(self):
       # Our unfitted model is Y_i = Beta_0 + Beta_1X_i + Ep_i
       # proof of b_1 and b_0 is in the README.md 
        self.n = self.data.shape[0]
        self.X_i, self.Y_i = self.data[:, 0] ,  self.data[:, 1]
        
        self.b_1 = ((np.sum(self.X_i *self.Y_i) - np.mean(self.Y_i)*np.sum(self.X_i))
                    /(np.sum(self.X_i**2) - np.mean(self.X_i) * np.sum(self.X_i))
                     )
        self.b_0 = np.mean(self.Y_i) - self.b_1*np.mean(self.X_i)
        self.params = json.dumps({'intercept' : f'{self.b_0:.10f}',
                       'slope (price)': f'{self.b_1:.10f}' 
                       }, indent=4)
        return self.params
    
    def resid(self):
        # find resids to get MSE which is our estimator for 
        # sigma^2  and s^2 = MSE. 
        df = self.n - 2
        self.Y_hat = self.b_0 + self.b_1 * self.X_i
        
        self.e_i = self.Y_i - (self.Y_hat)
        self.MSE = self.e_i**2 / df  

        # point esitmate of population standard deviation
        self.samp_std = np.sqrt(self.MSE)     
        self.SSR = np.sum((self.Y_hat - np.mean(self.Y_i))**2)
        self.SSE = np.sum(self.e_i**2)
        self.SSTo = self.SSR + self.SSE

    def conf_int(self, alpha=0.05):
       # We may be interested in making inferences on our slope
       # or our intercept. proof of variance of slope in README.md
       # H_0 :  \beta_1 = 0 , H_a: \beta_1 > 0
        self.var_b1 = self.MSE/np.sum((self.X_i - np.mean(self.X_i))**2) 
        self.std_b1 = np.sqrt(self.var_b1)

        if alpha == 0.05:
            # making a probability statement whether our slope
            res = f""""""
   
    def anova(self):
        
        anova_table = {
            
    'source'     : ['sum of squares', 'df', 'mse', 'F-stat'],
    'regression' : [self.SSR.item(), 1, self.SSR.item(), (self.SSR/self.SSE).item()],
    'error'      : [self.SSE.item(), self.n - 2, (self.SSE/(self.n - 2)).item(), ''],
    'total'      : [self.SSTo.item(), 1 + self.n-2, '', '']
        }
        for key, val in anova_table.items():
            print(key,val)

    def compute_rstat(self, r_sq=True):
        # computes r^2 by default but when r_sq if false
        # returns corr coffecicent r
        r_squared = self.SSR/self.SSTo
        if r_sq == True:
            return r_squared
        elif r_sq == False:
            return np.sqrt(r_squared)

    def summary(self):
        summary = f''

model = linereg(data)
params = model.linear_ests()
resid = model.resid()
anova = model.anova()
print(params)
print('\n', anova)
