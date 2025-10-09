import numpy as np 
import json
from scipy.stats import t

class LineReg():
    def __init__(self, data):
       
        # Best for data to be a matrix and since this is for slr a [n x 2] matrix
        # is ideal and required
        self.data = data
    
    def fit(self): 
       
    # All math logic for code. Function for calculating the necessary variables

        self.n = self.data.shape[0]
        # array indexing to get X and Y vals
        self.X_i, self.Y_i = self.data[:, 0] ,  self.data[:, 1]
        
        self.b_1 = ((np.sum(self.X_i *self.Y_i) 
                    - np.mean(self.Y_i)*np.sum(self.X_i))
                    / (np.sum(self.X_i**2) - np.mean(self.X_i) 
                    * np.sum(self.X_i))
                     )
        self.b_0 = np.mean(self.Y_i) - self.b_1*np.mean(self.X_i)
        
        self.df = self.n - 2
        self.Y_hat = self.b_0 + self.b_1*self.X_i
        self.e_i = self.Y_i - (self.Y_hat)
        self.MSE = (np.sum(self.e_i**2) / self.df ).astype(float) 

        # point esitmate of population standard deviation
        self.samp_std = np.sqrt(self.MSE).astype(float) 
        
        self.SSR = np.sum((self.Y_hat - np.mean(self.Y_i))**2).astype(float) 
        self.SSE = np.sum(self.e_i**2).astype(float) 
        self.SSTo = self.SSR + self.SSE
        
        s_xx = np.sum((self.X_i - np.mean(self.X_i))**2)
        
        self.var_b1 = self.MSE/s_xx        
        self.std_b1 = np.sqrt(self.var_b1)
        
        self.var_b0 = self.MSE*((1/self.n) + (np.mean(self.X_i)**2 / 
                        s_xx) )
        self.std_b0 = np.sqrt(self.var_b0)
 
        self.r_squared = self.SSR/self.SSTo
    

        self.t_val_b0, self.t_val_b1 = self.b_0/self.std_b0, self.b_1/self.std_b1
    
    def get_params(self):
       # Our unfitted model is Y_i = Beta_0 + Beta_1X_i + Ep_i
       # proof of b_1 and b_0 is in the README.md 
       self.params = json.dumps({'intercept' : f'{self.b_0:.10f}',
                                'slope (price)': f'{self.b_1:.10f}' 
                                }, 
                                indent=4)
       return self.params
    
    def predict(self, X):
        # Make predictions with linear estimates
        y_hat = self.b_0 + self.b_1 * X
        return y_hat
       
    def summary(self):
        # Supposed to recreate summary() function in R.
        # In json format
        topics = {'Estimate' :{'intercept': f'{self.b_0.item():.7f}', 
                                'slope (b_1)': f'{self.b_1.item():.7f}'
                               },
                  'Std. Error':{'intercept':f'{self.std_b0:.7f}', 
                                'slope (b_1)': f'{self.std_b1:.7f}'
                                },
                   't-value' : {'intercept': f'{self.t_val_b0:.7f}', 
                                'slope (b_1)': f'{self.t_val_b1:.7f}'
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
       # ANOVA table in json format.

        dict_summary = {'Regression': {'SSR' :self.SSR.item(), 
                                       'df': 1, 
                                       'MSR': self.SSR.item()},
                        'Error': {'SSE': self.SSE.item(), 
                                  'df': self.df, 
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
        if r_sq == True:
            return round(float(self.r_squared), 10)
        elif r_sq == False:
            return np.sqrt(self.r_squared)

 


