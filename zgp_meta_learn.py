'''
Date Preprocessing
'''

import pandas as pd
import numpy as np
import torch

data = pd.read_csv('wdata_buffer_pittsburg_Jun1Sept1_random_action.csv', index_col=0) # read data

# for each column, calculate and record the mean and standard deviation, then normalize the data
standardization = {}
for col in data.columns:
    standardization[col] = {'mean': data[col].mean(), 'std': data[col].std()}
    data[col] = (data[col] - data[col].mean()) / data[col].std()

X = data
Y = data['zone temperature'].shift(-1) # shift the zone temperature column up by 1 row

# drop the last row of X and Y, since the last row of X has no corresponding Y
X = X.drop(X.index[-1])
Y = Y.drop(Y.index[-1])

# tensorize X and Y
X = torch.tensor(X.values, dtype=torch.float32)
Y = torch.tensor(Y.values, dtype=torch.float32)


'''
Gaussian Process
'''

import gpytorch

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPModel(train_x=X, train_y=Y, likelihood=likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

losses = []
lengthscale = []
scale = []
noise = []
training_iter = 200
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(X)
    # Calc loss and backprop gradients
    loss = -mll(output, Y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    losses.append(loss.item())
    lengthscale.append(model.covar_module.base_kernel.lengthscale.item())
    scale.append(model.covar_module.outputscale.item())
    noise.append(model.likelihood.noise.item())
    optimizer.step()

# plot the loss curve
import matplotlib.pyplot as plt
plt.plot(losses)
plt.title('loss')
plt.show()
plt.savefig('zimages/loss.png')

# plot the lengthscale curve
plt.plot(lengthscale)
plt.title('lengthscale l')
plt.show()
plt.savefig('zimages/lengthscale.png')

# plot the scale curve
plt.plot(scale)
plt.title('scale sigma')
plt.show()
plt.savefig('zimages/scale.png')

# plot the noise curve
plt.plot(noise)
plt.title('noise sigma_epsilon')
plt.show()
plt.savefig('zimages/noise.png')

