using Images
using Plots
using Statistics
using LinearAlgebra
using Distributions
using JLD

pyplot()

include("sampler_gaussian.jl")
include("../rbm/visual-reporter/reporter.jl")
include("data_load.jl")
include("utils.jl")

x, labels = load_yeast()
X = x
Xtest, ltest = load_yeast(set = "test")

d, ns = size(X)
nh = 500
sigma = 0.001
n_epochs = 25000
lr = 5e-4
batch_size = 100
randomize = true
n_gibbs = 100
ratio = 0.2
n_mon = 10 # monitoring every n_mon epochs

m = mean(X, dims = 2)
s = mapslices(std, X, dims = 2)
X = (X .- m) ./ s

mt = mean(Xtest, dims = 2)
st = mapslices(std, Xtest, dims = 2)
Xtest = (Xtest .- mt) ./ st

Xm = Array{Union{typeof(X[1, 1]), Missing}, 2}(X)

for i in 1:size(Xm, 2)
  mask = missing_mask(d, ratio)
  Xm[mask, i] = [missing for i = 1:length(mask)]
end

include("yeast_report.jl")

rbm = RBM(Float64, Distributions.Normal, Boltzmann.IsingSpin, Boltzmann.IsingActivation, d, nh; sigma = sigma)

# getting the reporter
plots_list = [weights, PL, dW, SVs, re, re2, gre, gre2, rmse10, rmse50, rmse90]
vr = VisualReporter(rbm, n_mon, plots_list, pre = pre, init=Dict(:X => Xm, :Xt => X, :Xtest => Xtest, :persistent_chain => Xm[:, 1:batch_size], :dW_prev => zeros(nh, size(X, 1)), :missing => missing_mask(d, 0.5)))

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

filename = "log/yeast_$(ratio)_$(lr)_$(batch_size)_$(n_gibbs)_$(n_epochs)"

save("$filename.jld", "w", rbm.W, "vbias", rbm.vbias, "hbias", rbm.hbias, "ratio", ratio)

mp4(vr.anim, "$filename.mp4", fps=5)
gif(vr.anim, "$filename.gif")
