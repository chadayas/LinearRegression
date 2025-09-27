import numpy as np 
import csv

file = 'bmw_saless.csv'

sales, price = [],[]
with open('bmw_saless.csv', newline="") as csvfile:
    data = csv.DictReader(csvfile)
    for row in data:
       price.append(row['Price_USD'])
       sales.append(row['Sales_Volume'])

data = np.column_stack(
    [np.array(price, dtype=np.float32), 
     np.array(sales, dtype = np.float32)] )



class linereg(object):
    def __init__(self, data):
        self.data = data
    def linear_ests(self):
       # Our unfitted model is Y_i = Beta_0 + Beta_1X_i + Ep_i
       # proof of b_1 and b_0 is in the README.md 
        n = len(self.data.shape[0])
        self.X_i, self.Y_i = self.data[:, 0] ,  self.data[:, 1]
        
        self.b_1 = ((np.sum(self.X_i *self.Y_i) - np.mean(self.Y_i)*np.sum(self.X_i))
                    /(np.sum(X_i**2) - np.mean(self.X_i) * np.sum(self.X_i))
                     )
        self.b_0 = np.mean(self.Y_i) - self.b_1*np.mean(self.X_i)
        self.params = {'intercept' : f'{self.b_0:.4f}'
                'slope (price)': f'{self.b_1:.4f}'}
        return self.params
    
    def predict(self, vals):
        self.Y_hat = self.b_0 + self.b_1*vals
        return self.Y_hat

    def resid(self):
        # find resids to get MSE which is our estimator for 
        # sigma^2  and s^2 = MSE. 
        n = len(self.data.shape[0])
        df = n - 2
        
        self.e_i = self.Y_i - (self.Y_hat)
        self.MSE = self.e_i**2 / df  
        # point esitmate of population standard deviation
        self.samp_std = np.sqrt(self.MSE)     

   def conf_int(self, alpha=0.05)
       # We may be interestead in making inferences on our slope
       # or our intercept. proof of variance of slope in README.md
        self.var_b1 = self.MSE/ 
        self.error_b1 =1 
        
