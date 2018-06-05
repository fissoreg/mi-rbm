using MLDatasets
using Images
using Plots

pyplot()

include("sampler.jl")
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
n_epochs = 100
#lr = 5e-3
#batch_size = 10
randomize = true
#n_gibbs = 3

sigmas = [0.01, 0.001]
lrs = [5e-4, 1e-4, 5e-5, 1e-5]
bss = [30, 20, 50]
gs = [1, 3, 5]

ratio = 0.2

X, Y = MNIST.traindata()
X = reshape(X, 784, size(X, 3))[:, 1:ns]

X = 2X - 1

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
  
features = Dict(
  :ys => [:W],
  :transforms => [W -> reshape_mnist(W')],
  :title => "Features",
  :ticks => nothing
)

reconstructions = Dict(
  :ys => [(:rbm, :X)],
  :transforms => [(rbm, X) -> reshape_mnist(generate(rbm, X[:,1:100], n_gibbs=1000))],
  :title => "Reconstructions",
  :ticks => nothing
)

  
params = [(sigma, lr, bs, ng) for sigma in sigmas, lr in lrs, bs in bss, ng in gs]

for p in params

sigma, lr, batch_size, n_gibbs = p

chain = Dict(
  :ys => [:persistent_chain],
  :transforms => [chain -> reshape_mnist(chain, r = div(batch_size, 10))],
  :title => "Persistent chain",
  :ticks => nothing
)

println(batch_size)
rbm = IsingRBM(d, nh; sigma = sigma, X = X)

# getting the reporter
vr = VisualReporter(rbm, div(ns, batch_size) - 1, [weights, PL, features, reconstructions, chain], init=Dict(:X => X, :persistent_chain => X[:, 1:batch_size]))

try

fit(rbm, X;
  n_epochs=n_epochs,
  lr=lr,
  batch_size=batch_size,
  randomize=randomize,
  scorer=Boltzmann.pseudo_likelihood,
  sampler = persistent_contdiv,
  reporter=vr,
  init=Dict(:X => X, :ratio => ratio, :n_gibbs => n_gibbs)
)

mp4(vr.anim, "$(ratio)_$(lr)_$(sigma)_$(batch_size)_$(n_gibbs).mp4", fps=2)
gif(vr.anim, "$(ratio)_$(lr)_$(sigma)_$(batch_size)_$(n_gibbs).gif")

catch
  println("Crash")# on $lr_$sigma_$batch_size_$n_gibbs")
end

end
