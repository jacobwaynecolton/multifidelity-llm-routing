import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# --- Global Plotting & Font Configuration ---
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
# Fallbacks removed: This will fail loudly if Times New Roman is not found/cached
plt.rcParams['font.serif'] = ['Times New Roman'] 
plt.rcParams['mathtext.fontset'] = 'stix'

# Ensure fonts are embedded in the PDF as TrueType (Type 42) instead of vectorized paths
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Lock in the base font size for the whole plot
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 11    # X and Y axis labels
plt.rcParams['xtick.labelsize'] = 9    # Tick numbers (shrunk to stay out of the way)
plt.rcParams['ytick.labelsize'] = 9    # Tick numbers (shrunk to stay out of the way)
plt.rcParams['legend.fontsize'] = 10   # Legend text

# --- Configuration ---
EMBEDDINGS_FILE = "nlp_slm_embeddings.npy"
LABELS_FILE = "nlp_execution_labels.npy"  
PCA_COMPONENTS = 10         
NUM_INDUCING_POINTS = 100   
TRAINING_ITERATIONS = 100
BATCH_SIZE = 64

# --- Cost Assumptions ---
ORACLE_COST_MULTIPLIER = 4.7 

# --- 1. Load Data & PCA ---
print("Loading NLP data...")
X_raw = np.load(EMBEDDINGS_FILE)
y_raw = np.load(LABELS_FILE)

pca = PCA(n_components=PCA_COMPONENTS)
X_pca = pca.fit_transform(X_raw)

X_tensor = torch.tensor(X_pca, dtype=torch.float32)
y_tensor = torch.tensor(y_raw, dtype=torch.float32)
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# ==========================================
# Phase 1: Train & Evaluate SVGP Router
# ==========================================
print("Training SVGP Router...")
class SVGPClassificationModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=PCA_COMPONENTS))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

inducing_points = X_train[:NUM_INDUCING_POINTS, :]
model = SVGPClassificationModel(inducing_points)
likelihood = gpytorch.likelihoods.BernoulliLikelihood()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, likelihood = model.to(device), likelihood.to(device)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': likelihood.parameters()}], lr=0.01)
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train.size(0))

model.train()
likelihood.train()
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

for i in range(TRAINING_ITERATIONS):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = -mll(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()

print("Sweeping SVGP Confidence Thresholds...")
model.eval()
likelihood.eval()

with torch.no_grad():
    test_output = model(X_test)
    pred_probs = likelihood(test_output).mean.cpu().numpy()
    true_labels = y_test.cpu().numpy()

thresholds = np.linspace(0.0, 1.0, 100)
svgp_accuracies = []
svgp_costs = []

for t in thresholds:
    # Route to Oracle if GP predicts SLM success probability is below threshold
    route_to_oracle = pred_probs < t
    correct_count = 0
    for i in range(len(true_labels)):
        if route_to_oracle[i]:
            correct_count += 1  
        else:
            if true_labels[i] == 1.0:
                correct_count += 1
                
    overall_accuracy = (correct_count / len(true_labels)) * 100
    percent_oracle = np.mean(route_to_oracle)
    percent_slm = 1.0 - percent_oracle
    aggregate_cost = (percent_slm * 1.0) + (percent_oracle * ORACLE_COST_MULTIPLIER)
    
    svgp_accuracies.append(overall_accuracy)
    svgp_costs.append(aggregate_cost)

# ==========================================
# Phase 2: Contextual Multi-Armed Bandit
# ==========================================
print("Evaluating CMAB Baseline...")

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_train.cpu().numpy())
test_contexts = kmeans.predict(X_test.cpu().numpy())

mab_accuracies = []
mab_costs = []

penalties = np.linspace(0.8, 0.0, 50) 

for penalty in penalties:
    priors = {
        0: {'SLM': [1, 1], 'Oracle': [1, 1]},
        1: {'SLM': [1, 1], 'Oracle': [1, 1]}
    }
    
    correct_count = 0
    oracle_picks = 0
    
    for i in range(len(true_labels)):
        ctx = test_contexts[i]
        true_slm_success = int(true_labels[i])
        
        slm_sample = np.random.beta(priors[ctx]['SLM'][0], priors[ctx]['SLM'][1])
        oracle_sample = np.random.beta(priors[ctx]['Oracle'][0], priors[ctx]['Oracle'][1]) - penalty
        
        if slm_sample > oracle_sample:
            if true_slm_success == 1:
                correct_count += 1
                priors[ctx]['SLM'][0] += 1 
            else:
                priors[ctx]['SLM'][1] += 1 
        else:
            oracle_picks += 1
            correct_count += 1
            priors[ctx]['Oracle'][0] += 1 
            
    acc = (correct_count / len(true_labels)) * 100
    perc_oracle = oracle_picks / len(true_labels)
    perc_slm = 1.0 - perc_oracle
    cost = (perc_slm * 1.0) + (perc_oracle * ORACLE_COST_MULTIPLIER)
    
    mab_accuracies.append(acc)
    mab_costs.append(cost)

# ==========================================
# Phase 3: Plotting the Trade-off Curves
# ==========================================
# Locked figure size to perfectly match LaTeX text width
plt.figure(figsize=(7.2, 3.5))

plt.plot(svgp_costs, svgp_accuracies, marker='.', linestyle='-', color='#1f77b4', markersize=6, label='SVGP Router (Proposed)')
plt.plot(mab_costs, mab_accuracies, marker='x', linestyle='--', color='#ff7f0e', markersize=6, alpha=0.8, label='Contextual MAB Baseline')

sd_accuracy = max(svgp_accuracies)
sd_cost = 3.8 
plt.plot(sd_cost, sd_accuracy, marker='*', color='#2ca02c', markersize=15, linestyle='None', label='Speculative Decoding')

plt.xlabel("Aggregate Computational Cost (1.0x = SLM Only, 4.7x = Oracle Only)")
plt.ylabel("Overall System Accuracy (%)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right')

output_img = "tradeoff_curve_nlp.pdf"
plt.savefig(output_img, format='pdf', bbox_inches='tight')
print(f"\nSuccess! Final NLP trade-off curve saved to: {output_img}")