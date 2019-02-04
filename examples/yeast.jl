using Boltzmann
using Distributions
using JLD

include("../src/sampler_gaussian.jl")
include("../src/data_load.jl")
include("../src/utils.jl")

X, labels = load_yeast()
Xtest, ltest = load_yeast(set = "test")

d, ns = size(X)

# hyperparameters
nh = 500 # number of hidden units
sigma = 0.001 # weights std initialization
n_epochs = 1
lr = 1e-4 # learning rate
batch_size = 100
randomize = true
n_gibbs = 100
ratio = 0.2 # fraction of missing variables
n_mon = 30 # monitoring every n_mon epochs

# centering variables and normalizing variance
m = mean(X, dims = 2)
s = mapslices(std, X, dims = 2)
X = (X .- m) ./ s

mt = mean(Xtest, dims = 2)
st = mapslices(std, Xtest, dims = 2)
Xtest = (Xtest .- mt) ./ st

# introducing missing values
Xm = Array{Union{typeof(X[1, 1]), Missing}, 2}(X)

for i in 1:size(Xm, 2)
  mask = missing_mask(d, ratio)
  Xm[mask, i] = [missing for i = 1:length(mask)]
end

rbm = RBM(Float64, Distributions.Normal, Boltzmann.IsingSpin, Boltzmann.IsingActivation, d, nh; sigma = sigma)

# getting the reporter to visualize training state
include("yeast_report.jl")
plots_list = [weights, PL, dW, SVs, re2, gre2]
vr = VisualReporter(rbm, n_mon, plots_list, pre = pre, init=Dict(:X => Xm, :Xt => X, :Xtest => Xtest, :persistent_chain => Xm[:, 1:batch_size], :dW_prev => zeros(nh, size(X, 1)), :ratio => 0.5))

# fitting the RBM
fit(rbm, Xm;
  n_epochs = n_epochs,
  lr = lr,
  batch_size = batch_size,
  randomize = randomize,
  scorer = Boltzmann.pseudo_likelihood,
  sampler = persistent_contdiv,
  update = update_simple!,
  reporter = vr,
  X = Xm,
  n_gibbs = n_gibbs
)

# saving the model and the training visualization
try mkdir("../log") catch end
filename = "../log/yeast_h$(nh)_r$(ratio)_lr$(lr)_bs$(batch_size)_ng$(n_gibbs)_e$(n_epochs)"
save("$filename.jld", "w", rbm.W, "vbias", rbm.vbias, "hbias", rbm.hbias, "ratio", ratio)
mp4(vr.anim, "$filename.mp4", fps=5)
gif(vr.anim, "$filename.gif")
