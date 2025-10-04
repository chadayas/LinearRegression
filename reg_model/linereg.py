import numpy as np 
import csv
import json
from scipy.stats import t
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
        self.MSE = np.sum(self.e_i**2) / df  

        # point esitmate of population standard deviation
        self.samp_std = np.sqrt(self.MSE)     
        self.SSR = np.sum((self.Y_hat - np.mean(self.Y_i))**2)
        self.SSE = np.sum(self.e_i**2)
        self.SSTo = self.SSR + self.SSE

    def summary(self):
        s_xx = np.sum((self.X_i - np.mean(self.X_i))**2)
        self.var_b1 = self.MSE/s_xx        
        self.std_b1 = np.sqrt(self.var_b1)
        
        self.var_b0 = self.MSE*((1/self.n) + (np.mean(self.X_i)**2 / s_xx) )
        self.std_b0 = np.sqrt(self.var_b0)
         
        t_val_b0, t_val_b1 = self.b_0/self.std_b0, self.b_1/self.std_b1
        topics = {'Estimate' :{'intercept': f'{self.b_0.item():.7f}', 
                                'slope (b_1)': f'{self.b_1.item():.7f}'
                               },
                  'Std. Error':{'intercept':f'{self.std_b0:.7f}', 
                                'slope (b_1)': f'{self.std_b1:.7f}'
                                },
                   't-value' : {'intercept': f'{t_val_b0:.7f}', 
                                'slope (b_1)': f'{t_val_b1:.7f}'
                                }
                   } 
        summary = json.dumps(topics, indent=4)
        return summary

    def conf_int(self, alpha=0.05):
        
       # We may be interested in making inferences on our slope
       # or our intercept. proof of variance of slope in README.md
       # H_0 :  \beta_1 = 0 , H_a: \beta_1 > 0
        crit_val = t.ppf(1 - alpha/2, df=self.n-2) 
            # making a probability statement of what possible values
            # our slope is between with a level of confidence.
        lower = self.b1 - self.std_b1*crit_val
        upper = self.b1 + self.std_b1*crit_val
        prob_s = f"{lower} <= beta_1 <= {upper}"
        return prob_s 
    
    def anova(self):
        dict_summary = {'Regression': {'SSR' :self.SSR.item(), 
                                       'df': 1, 
                                       'MSR': self.SSR.item()},
                        'Error': {'SSE': self.SSE.item(), 
                                  'df': self.n - 2, 
                                  'MSE': self.MSE.item()},
                        'Total': {'SSTo': self.SSTo.item(), 
                                  'df total':self.n - 1 },
                        'F-stat' : (self.SSR/self.MSE).item()
        }
        
        anova_table = json.dumps(dict_summary, indent=4)
        
        return anova_table
    
    def compute_rstat(self, r_sq=True):
        # computes r^2 by default but when r_sq if false
        # returns corr coffecicent r
        r_squared = self.SSR/self.SSTo
        if r_sq == True:
            return round(float(r_squared), 10)
        elif r_sq == False:
            return np.sqrt(r_squared)

 


