using MLDatasets
using Images
using Plots
using Statistics
using Distributions
using LinearAlgebra

pyplot()

include("sampler_mean.jl")
include("utils.jl")
include("../rbm/visual-reporter/reporter.jl")

# we need to reshape weights/samples for visualization purposes
function reshape_mnist(samples; c=10, r=10, h=28, w=28)
  f = zeros(r*h,c*w)
  for i=1:r, j=1:c
    f[(i-1)*h+1:i*h,(j-1)*w+1:j*w] = reshape(samples[:,(i-1)*c+j],h,w)'
  end
  w_min = minimum(samples)
  w_max = maximum(samples)
  scale = x -> (x-w_min)/(w_max-w_min)
  map!(scale,f,f)
  colorview(Gray,f)
end

ns = 10000
d = 784
nh = 100
#sigma = 0.001
n_epochs = 200
lr = 0.00001
batch_size = 50
randomize = true
n_gibbs = 1

sigmas = [0.01]
lrs = [0.0005] #5e-4, 1e-4, 5e-5, 1e-5]
bss = [50] #30, 20]
gs = [1] #, 5]

ratios = [0.2] #, 0.05, 0.1, 0.2]

X, Y = MNIST.traindata()
X = reshape(X, 784, size(X, 3))[:, 1:ns]
X = 255 * X
#m = mean(X, 2)
#s = mapslices(std, X, 2)
#println(s)
#s[s .< 0.0001] = 0.0001
#println(s)
##X = 2X - 1
#X = (X .- m) ./ s
#X[isnan.(X)] = 0

nz = []
for i=1:size(X,1)
  push!(nz,findall(x->x!=0,X[i,:]))
end

# Get normalized variables
m = mean(X, dims = 2);
X = X.-m;
σ = zeros(size(X,1))
for i=1:size(X,1)
  if(!isempty(nz[i]))
    σ[i] = std(X[i,nz[i]])
  end
end
# σ = std(X,2);
idxs = findall(x->x<50,σ)
for i in idxs
  σ[i] = 50
end
X = X ./ σ;
idxs = findall(x->isnan(x),X)
for i in idxs
  X[i] = 0;
end

p = plot(reshape_mnist(X[:, 1:100]))
savefig(p, "samples.png")

Xm = Array{Union{typeof(X[1, 1]), Missing}, 2}(X)

for i in 1:size(Xm, 2)
  mask = missing_mask(d, ratios[1])
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
  :title => "Gradient",
  :incremental => true,
  :leg => false
)

svs = zeros(nh)'

SVs = Dict(
  :ys => [:s],
  :transforms => [s -> (global svs = vcat(svs, s'); svs)],
  :title => "Singular Values",
  :leg => false
)

SV = Dict(
  :ys => [:s for i=1:50],
  :transforms => [x -> x[i] for i=1:50],
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

reconstructions = Dict(
  :ys => [(:rbm, :Xt)],
  :transforms => [(rbm, X) -> reshape_mnist(hcat(X[:, 1:30], rbm.W' * rbm.W * X[:, 1:30]), r = 6)], #reshape_mnist(generate(rbm, X[:,1:100], n_gibbs=1))],
  :title => "Reconstructions",
  :ticks => nothing
)

#function RE(rbm, X)
#  idxs = rand(1:size(X, 2), 100)
#  gen = generate(rbm, X[:,idxs], n_gibbs=1)
#
#  mean([norm(gen[:, i] - X[:, i]) for i = 1:100])
#end
#
#re = Dict(
#  :ys => [(:rbm, :X)],
#  :transforms => [(rbm, X) -> RE(rbm, X)],
#  :title => "Reconstruction error",
#  :incremental => true,
#  :leg => false
#)

re = Dict(
  :ys => [(:rbm, :Xt, :missing)],
  :transforms => [(rbm, X, mask) -> RE(rbm, X, mask; n_gibbs = 1)],
  :title => "Reconstruction error",
  :incremental => true,
  :lab => "CD1"
)

re2 = Dict(
  :ys => [(:rbm, :Xt, :missing)],
  :transforms => [(rbm, X, mask) -> RE(rbm, X, mask; n_gibbs = 1000)],
  :title => "Reconstruction error",
  :incremental => true,
  :lab => "CD1000"
)

#gen_activations(rbm, X; n_gibbs = 1) = Boltzmann.vis_means(rbm, Boltzmann.gibbs(rbm, X; n_times = 1)[4])

sam = rand(1:size(X, 2), 1)

function reconstruction(rbm, X, mask)
  s = X[:, sam]
  lossy = get_lossy(s, mask)
  #gen_pin = gen_activations(rbm, s) 
  gen_pin = generate(rbm, lossy, mask)
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

params = [(sigma, lr, bs, ng, ratio) for sigma in sigmas, lr in lrs, bs in bss, ng in gs, ratio in ratios]

for p in params

sigma, lr, batch_size, n_gibbs, ratio = p

chain = Dict(
  :ys => [:persistent_chain],
  :transforms => [chain -> reshape_mnist(chain, r = div(batch_size, 10))],
  :title => "Persistent chain",
  :ticks => nothing
)

println(batch_size)
rbm = RBM(Float64, Distributions.Normal, Boltzmann.IsingSpin, Boltzmann.IsingActivation, d, nh; sigma = sigma)
#rbm = GRBM(d, nh; sigma = sigma)#, X = X)

# getting the reporter
vr = VisualReporter(rbm, div(ns, batch_size) - 1, [weights, PL, dW, re, SVs, features, reconstructions, gre], pre = pre, init=Dict(:Xt => X, :X => Xm, :persistent_chain => X[:, 1:batch_size], :dW_prev => zeros(nh, size(X, 1)), :missing => missing_mask(d, ratio)))
#, SVs, features, reconstructions, chain
#try

fit(rbm, Xm;
  n_epochs=n_epochs,
  lr=lr,
  batch_size=batch_size,
  randomize=randomize,
  scorer=Boltzmann.pseudo_likelihood,
  sampler = persistent_contdiv,
  update = update_simple!,
  reporter=vr,
  X = Xm,
  ratio = ratio,
  n_gibbs = n_gibbs,
  dW_prev = zeros(nh, size(X, 1))
)

filename = "log/gaussian_$(ratio)_$(lr)_$(sigma)_$(batch_size)_$(n_gibbs)"

mp4(vr.anim, "$filename.mp4", fps=2)
gif(vr.anim, "$filename.gif")

ps = [vr.plots[i][:plot] for i = 1:length(vr.plots)]
savefig(plot(ps...), "$filename.png")

#catch e
#  println("Crash:")# on $lr_$sigma_$batch_size_$n_gibbs")
#  println("$e")
#end

end
