using MLDatasets

include("../src/sampler_gaussian.jl")
include("../src/utils.jl")

# number of samples to use for training
ns = 10000

X, Y = MNIST.traindata()
X = reshape(X, 784, size(X, 3))[:, 1:ns]

d, ns = size(X)
c = size(Y, 1)

# hyperparameters
nh = 100 # number of hidden units
sigma = 0.001 # weights std initialization
n_epochs = 1000
lr = 1e-4 # learning rate
batch_size = 100
randomize = true
n_gibbs = 1
ratio = 0.8 # fraction of missing variables
n_mon = 20 # visualize training state every n_mon epochs

# centering variables
m = mean(X, dims = 2);
X = X .- m;

# rescaling variance
δ = 0.5
σs = [std(filter(x -> x != 0, X[i, :])) for i in 1:size(X, 1)]
map!(s -> isnan(s) ? 1 : s, σs, σs)
map!(s -> s < δ ? δ : s, σs, σs)
X = X ./ σs

# introducing missing values
Xtrue = deepcopy(X)

X = Array{Union{Missing, Float64}, 2}(X)

for i in 1:size(X, 2)
  mask = missing_mask(d, ratio)
  X[mask, i] = [missing for i = 1:length(mask)]
end

# declaring a RBM with gaussian visible units and binary (+- 1) hidden units
rbm = RBM(Float64, Distributions.Normal, Boltzmann.IsingSpin, Boltzmann.IsingActivation, d, nh; sigma = sigma)

# getting the reporter to visualize training state
include("mnist_report.jl")
plots_list = [ws, dW, PL, SVs, features, reconstructions]
lossy = map(x -> ismissing(x) ? 0 : x, X)
vr = VisualReporter(rbm, n_mon, plots_list, pre = pre, init = Dict(:X => X, :Xtrue => Xtrue, :lossy => lossy, :dW_prev => zeros(nh, size(X, 1))))

# fitting the RBM
fit(rbm, X;
  n_epochs = n_epochs,
  lr = lr,
  batch_size = batch_size,
  randomize = randomize,
  sampler = persistent_contdiv,
  update = update_simple!,
  reporter = vr,
  n_gibbs = n_gibbs
)

# saving the model and the training visualization
try mkdir("log") catch end

filename = "log/mnist_h$(nh)_r$(ratio)_lr$(lr)_bs$(batch_size)_ng$(n_gibbs)_e$(n_epochs)"
save("$filename.jld", "w", rbm.W, "vbias", rbm.vbias, "hbias", rbm.hbias, "ratio", ratio)
mp4(vr.anim, "$filename.mp4", fps=5)
