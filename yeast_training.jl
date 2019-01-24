using Images
using Plots
using Statistics
using LinearAlgebra
using Distributions
using JLD

pyplot()

include("sampler_mean.jl")
include("../rbm/visual-reporter/reporter.jl")
include("data_load.jl")
include("utils.jl")

x, labels = load_yeast()
X = x
Xtest, ltest = load_yeast(set = "test")
#X = vcat(x, labels)

d, ns = size(X)
nh = 100
sigma = 0.01
n_epochs = 2500
lr = 1e-4
batch_size = 50
randomize = true
n_gibbs = 100
ratio = 0.2 #, 0.05, 0.1, 0.2]
n_mon = 5 # monitoring every n_mon epochs

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

pre = Dict(
  :in => [:rbm],
  :preprocessor => rbm -> (r = svd(rbm.W); (r.U, r.S, r.V)),
  :out => [:U, :s, :V]
)

# declaring wanted plots
weights = Dict(
  :ys => [:W],
  :transforms => [x->x[:]],
  :title => "Weights",
  :seriestype => :histogram,
  :leg => false,
  :nbins => 100
)

PL = Dict(
  :ys => [(:rbm, :X)],
  :transforms => [Boltzmann.pseudo_likelihood],
  :title => "Pseudolikelihood",
  :incremental => true,
  :leg => false
)

dW = Dict(
  :ys => [:dW_prev],
  :transforms => [x -> norm(x)],
  :title => "Gradient norm",
  :incremental => true,
  :leg => false
)

svs = zeros(min(d, nh))'

SVs = Dict(
  :ys => [:s],
  :transforms => [s -> (global svs = vcat(svs, s'); svs)],
  :title => "Singular Values",
  :leg => false
)

SV = Dict(
  :ys => [:s for i=1:1],
  :transforms => [x -> x[i] for i=1:1],
  :incremental => true,
  :title => "Singular values",
  :leg => false
)

features = Dict(
  :ys => [:W],
  :transforms => [W -> reshape_mnist(W')],
  :title => "Features",
  :ticks => nothing
)

#mask = missing_mask(d, ratio)

re = Dict(
  :ys => [(:rbm, :Xt, :missing)],
  :transforms => [(rbm, X, mask) -> vcat(rRE(rbm, X, mask), rRE(rbm, X, mask; n_gibbs = 100))],
  :title => "Reconstruction error",
  :incremental => true,
  :lab => ["CD1" "CD100"]
)

re2 = Dict(
  :ys => [(:rbm, :Xt, :missing)],
  :transforms => [(rbm, X, mask) -> rRE(rbm, X, mask; n_gibbs = 100)],
  :title => "Reconstruction error",
  :incremental => true,
  :lab => "CD100"
)

gen_activations(rbm, X; n_gibbs = 1) = Boltzmann.vis_means(rbm, Boltzmann.gibbs(rbm, X; n_times = 1)[4])

sam = rand(1:size(X, 2), 1)

function reconstruction(rbm, X, mask; n_gibbs = 1)
  s = X[:, sam]
  lossy = get_lossy(s, mask)
  #gen_pin = gen_activations(rbm, s) 
  gen_pin = generate(rbm, lossy, mask; n_gibbs = n_gibbs)
  #obs = setdiff(1:size(X, 1), mask)
  #gen_pin[obs] = s[obs]
  (s, gen_pin)
end

gre = Dict(
  :ys => [(:rbm, :Xt, :missing)],
  :transforms => [(rbm, X, mask) -> reconstruction(rbm, X, mask)],
  :title => "Reconstruction",
  :linetype => :scatter
)

gre2 = Dict(
  :ys => [(:rbm, :Xt, :missing)],
  :transforms => [(rbm, X, mask) -> reconstruction(rbm, X, mask; n_gibbs = 100)],
  :title => "Recon - CD100",
  :linetype => :scatter
)

xrange = [0.1, 0.5, 0.9]

rmse = Dict(
  :ys => [(:rbm, :Xt, :Xtest)],
  :transforms => [(rbm, Xt, Xtest) -> (repeat(xrange, inner = (1,2)), [rRE(rbm, x, missing_mask(d, r), n_gibbs = 100) for r in xrange, x in [Xt, Xtest]])],
  :title => "RMSE",
  :lab => ["Train" "Test"]
)

function project(V, s, X, idxs = [1, 2]) #i = 1, j = 2)
  p = [[dot(V[:, i], X[:, k]) for k in 1:size(X, 2)] for i in idxs]

  # dirty way to build a tuple on  the fly...
  (p[1:end-1]..., p[end])
end

function get_color(label)
  label *= 0.2
  RGB(sum(label[1:5]), sum(label[6:10]), sum(label[11:end]))
end

function color_labels(labels)
  get_color.([labels[:, j] for j in 1:size(labels, 2)])
end

scat3d = Dict(
  :ys => [(:V, :s, :Xt)],
  :transforms => [(V, s, X) -> project(V, s, X, [1, 2, 3])],
  :title => "V1 - V2 - V3",
  :linetype => :scatter,
  :ms => 2,
  :msw => 0,
  :c => color_labels(labels)
)

scatter = Dict(
  :ys => [(:V, :s, :Xt)],
  :transforms => [project],
  :title => "V1 - V2",
  :linetype => :scatter,
  :ms => 2,
  :msw => 0,
  :c => color_labels(labels)
)

scatter2 = Dict(
  :ys => [(:V, :s, :Xt)],
  :transforms => [(V, s, X) -> project(V, s, X, [1, 3])],
  :title => "V1 - V3",
  :linetype => :scatter,
  :ms => 2,
  :msw => 0,
  :c => color_labels(labels)

)

scatter3 = Dict(
  :ys => [(:V, :s, :Xt)],
  :transforms => [(V, s, X) -> project(V, s, X, [2, 3])],
  :transforms => [project],
  :title => "V2 - V3",
  :linetype => :scatter,
  :ms => 2,
  :msw => 0,
  :c => color_labels(labels)

)

rbm = RBM(Float64, Distributions.Normal, Boltzmann.IsingSpin, Boltzmann.IsingActivation, d, nh; sigma = sigma)
#rbm = GRBM(d, nh; sigma = sigma)#, X = X)

# getting the reporter
plots_list = [weights, PL, dW, SVs, re, re2, gre, gre2, rmse, scat3d] #, scatter, scatter2, scatter3]
vr = VisualReporter(rbm, n_mon, plots_list, pre = pre, init=Dict(:X => Xm, :Xt => X, :Xtest => Xtest, :persistent_chain => Xm[:, 1:batch_size], :dW_prev => zeros(nh, size(X, 1)), :missing => missing_mask(d, 0.5)))

fit(rbm, Xm;
  n_epochs = n_epochs,
  lr=lr,
  batch_size=batch_size,
  randomize=randomize,
  scorer=Boltzmann.pseudo_likelihood,
  sampler = persistent_contdiv,
  update = update_simple!,
  reporter=vr,
  X = Xm,
  #ratio = ratio,
  n_gibbs = n_gibbs
)

filename = "log/yeast_$(ratio)_$(lr)_$(batch_size)_$(n_gibbs)_$(n_epochs)"

save("$filename.jld", "w", rbm.W, "vbias", rbm.vbias, "hbias", rbm.hbias, "ratio", ratio)

mp4(vr.anim, "$filename.mp4", fps=5)
gif(vr.anim, "$filename.gif")
