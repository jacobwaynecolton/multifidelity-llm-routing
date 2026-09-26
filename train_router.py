import torch
import gpytorch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
# loading in the numpy arrays for the embeddings and their associated labels 
X_raw = np.load("slm_embeddings.npy")
y_raw = np.load("execution_labels.npy")

# Compress embeddings using pca, currently compressing to 10 dimensions from 960 
pca = PCA(n_components=10)
# transforming x to be 10 dimensions for when I feed it to the gp
X_pca = pca.fit_transform(X_raw)

# converting from a numpy array to a tensor to use with pytorch&gpytorch
X_tensor = torch.tensor(X_pca, dtype=torch.float32)
y_tensor = torch.tensor(y_raw, dtype=torch.float32)

#re-splitting the data, current seed = 42
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

class SVGPModel(gpytorch.models.ApproximateGP):

    def __init__(self, inducing_points):
        # using cholesky vvariational distribution so when the matrices are constructed, they can be updated
        # without some sort of rounding error occurring during the gpu math
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        # turning on learning for the inducing points so they can be moved during the optimization 
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        
        super().__init__(variational_strategy)
        # setting the mean module as constant mean
        self.mean_module = gpytorch.means.ConstantMean()
        # setting the covariance module as an rbf kernel wrapped in a scale kernel
        # the scale kernel will uncompress the results after the rbf kernel squashes them down to between 0 & 1
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=10))

    # Setting up forward pass to return a multivariate normal distribution with the mean & covariance
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initially setting the inducing points as the first 100 from x_train, but they will be updated
inducing_points = X_train[:100, :]

model = SVGPModel(inducing_points)
# setting the likelihood to bernoulli so the output gets squashed down to a probability 
likelihood = gpytorch.likelihoods.BernoulliLikelihood()

# dynamic device coding just in case
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Moving everything to the same hardware to prevent that mismatch crash
model, likelihood = model.to(device), likelihood.to(device)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Using the adam optimizer --adjusts its speed dynamically. need to learn if there are better optimizers
optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': likelihood.parameters()}], lr=0.01)

# Using elbo marginal log likelihood, lower bound to be pushed as high as it can? 
# clarifying the number of data elements as the size of y_train
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train.size(0))

# had a gpu out-of-memory crash, using dataloader to drop the embeddings
# into the gpu in chunks of 64 to avoid that occurring once more
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Activating training mode
model.train()
likelihood.train()

# Iterating for 100 epochs
print("Training GP...")
for epoch in range(100):
    # Iterating through each batch of data
    for x_batch, y_batch in train_loader:
        # resetting the gradients to avoid momentum carry over
        optimizer.zero_grad()
        # generating the output
        output = model(x_batch)
        # generating the loss (elbo wants to maximize by pushing the likihoold up, but the optimizer default is to minimize, so we must flipt he fign for mll)
        loss = -mll(output, y_batch)
        # autograd -- figureing out exactly which direction every anchor point needs to go
        loss.backward()
        # update the inducing points
        optimizer.step()
        

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# switching to evaluation mode
model.eval()
likelihood.eval()

# turning off the generation of the computational graph --gradient tracking
with torch.no_grad():
    # test the model on the test set
    test_output = model(X_test)
    preds = likelihood(test_output).mean
    # 1 if > 0.5, then compare to the vector from the y_test, then take the mean
    acc = ((preds > 0.5).float() == y_test).float().mean()
    print(f"Test Accuracy: {acc.item() * 100:.2f}%")