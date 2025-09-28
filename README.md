# Purpose
We have a dataset of BMW car sales, with 11 variables, but for right now we will fit a linear regression model on it
with a predictor variable being price and our response variable being sales. 
Not anything important just have a exam on monday, lets have some fun.

# Theory 
We have a SLR; simple linear regression model defined below

$$Y_i = \beta_0 + \beta_1X_i +\epsilon_i$$

Where $Y_i$ is our response variable, $\beta_1$ and $\beta_0$ is both our parameters and $\epsilon_i$ is the random error term.
We can start off with our assummptions for SLR, which are $\epsilon_i \overset{i.i.d}{\sim} N(0, \sigma^2)$.
## Least squares method
The least squares methods is the devaitions from of $Y_i$ from its mean $\mathbb{E}[Y_i]$. Remeber $Y_i$ is normally distrubited
random variable such that $Y_i \sim N(\beta_0 + \beta_1X_i, \sigma^2)$. Remeber; $\mathbb{E}[Y_i] = \beta_0 + \beta_1X_i$ since 
$\mathbb{E}[\epsilon_i] = 0$ from our assumption. We can write our least squares equation $Q$ as this.

$$Q = \sum_{i=1}^n (Y_i - (\beta_0 +\beta_1X_i ))^2$$

The goal is too minimize $Q$ in order to keep our model relatively stable, a high value of $Q$ states higher deviations away from
the mean, which could hinder predictions.

## Our friend Mr.Partial Derivative
Take the PD of $\beta_1$ and $\beta_0$

$$\frac{\partial Q}{\partial \beta_0} = \sum_{i=1}^{n} -2(Y_i - \beta_0 - \beta_1X_i)$$ 

$$\frac{\partial Q}{\partial \beta_1} = \sum_{i=1}^{n} -2X_i(Y_i - \beta_0 - \beta_1X_i)$$
                                                                                      

