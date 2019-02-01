using Images
using Plots
using Statistics
using LinearAlgebra
using Distributions

pyplot()

include("visual-reporter/reporter.jl")

macro transpose(t, exp)
  :($t ? $exp' : $exp)
end

# we need to reshape weights/samples for visualization purposes
function reshape_mnist(samples; c=10, r=10, h=28, w=28, t = true)
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

pre = Dict(
  :in => [:rbm],
  :preprocessor => rbm -> (r = svd(rbm.W); (r.U, r.S, r.V)),
  :out => [:U, :s, :V]
)

# declaring wanted plots
ws = Dict(
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
  :transforms => [x -> x[:]],
  :title => "dW",
  :seriestype => :histogram,
  :leg => false,
  :nbins => 100
)

dW_norm = Dict(
  :ys => [:dW_prev],
  :transforms => [x -> norm(x)],
  :title => "Gradient norm",
  :incremental => true,
  :leg => false
)

svs = zeros(min(d+c, nh))'

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
  :ys => [:rbm],
  :transforms => [rbm -> reshape_mnist(rbm.W')],
  :title => "Features",
  :ticks => nothing
)

function generate(rbm, X)
  X = Array{Float64, 2}(X)
  h_samples, _ = sample_hiddens(rbm, X)
  v_means = vis_means(rbm, h_samples)
end

reconstructions = Dict(
  :ys => [(:rbm, :X, :Xtrue, :lossy)],
  :transforms => [(rbm, X, Xtrue, lossy) -> (idxs = rand(1:ns, 30); reshape_mnist(hcat(Xtrue[1:d, idxs], lossy[1:d, idxs], generate(rbm, X[:, idxs])[1:d, :]), r = 9))], #reshape_mnist(generate(rbm, X[:,1:100], n_gibbs=1))],
  :title => "Reconstructions",
  :ticks => nothing
)

xbias = Dict(
  :ys => [:rbm],
  :transforms => [rbm -> reshape_mnist(rbm.vbias[1:d], r = 1, c = 1)],
  :title => "xbias",
  :ticks => nothing
)

lbias = Dict(
  :ys => [:rbm],
  :transforms => [rbm -> reshape_mnist(repeat(rbm.vbias[d + 1:end], inner = (100, 1)), h = 10, w = 100, r = 1, c = 1)],
  :title => "lbias",
  :ticks => nothing
)


modes = Dict(
  :ys => [:V],
  :transforms => [V -> reshape_mnist(V[1:d, 1:10], r = 1)],
  :title => "PCA",
  :ticks => nothing
)

lmodes = Dict(
  :ys => [:V],
  :transforms => [V -> reshape_mnist(repeat(V[d+1 : end, 1:10]', inner = (100, 1)), h = 10, w = 100, r = 10, c = 1)],
  :title => "PCA - Labels",
  :ticks => nothing
)

#mask = missing_mask(d, ratio)

re = Dict(
  :ys => [(:rbm, :Xt, :missing)],
  :transforms => [(rbm, X, mask) -> vcat(RE(rbm, X, mask), RE(rbm, X, mask; n_gibbs = 1000))],
  :title => "Reconstruction error",
  :incremental => true,
  :lab => ["CD1" "CD1000"]
)

re2 = Dict(
  :ys => [(:rbm, :Xt, :missing)],
  :transforms => [(rbm, X, mask) -> RE(rbm, X, mask; n_gibbs = 1000)],
  :title => "Reconstruction error",
  :incremental => true,
  :lab => "CD1000"
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
  :transforms => [(rbm, X, mask) -> reconstruction(rbm, X, mask; n_gibbs = 10000)],
  :title => "Recon - CD10000",
  :linetype => :scatter
)

function project(V, s, X, idxs = [1, 2]) #i = 1, j = 2)
  p = [[dot(V[:, i], X[:, k]) for k in 1:size(X, 2)] for i in idxs]

  # dirty way to build a tuple on  the fly...
  (p[1:end-1]..., p[end])
end

function get_color(label)
  l = get_class(label) * 0.1
  RGB(l, l, l)
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
  :c => color_labels(Y)
)

scat = Dict(
  :ys => [(:V, :s, :Xt)],
  :transforms => [project],
  :title => "V1 - V2",
  :linetype => :scatter,
  :ms => 2,
  :msw => 0,
  :c => color_labels(Y)
)

scatter2 = Dict(
  :ys => [(:V, :s, :Xt)],
  :transforms => [(V, s, X) -> project(V, s, X, [1, 3])],
  :title => "V1 - V3",
  :linetype => :scatter,
  :ms => 2,
  :msw => 0,
  :c => color_labels(Y)

)

scatter3 = Dict(
  :ys => [(:V, :s, :Xt)],
  :transforms => [(V, s, X) -> project(V, s, X, [2, 3])],
  :transforms => [project],
  :title => "V2 - V3",
  :linetype => :scatter,
  :ms => 2,
  :msw => 0,
  :c => color_labels(Y)

)

function classify(rbm, X, c)
  #p = rbm.W' * (rbm.W * X[1:784, : ] .+ rbm.hbias) .+ rbm.vbias
  h_samples, h_means = sample_hiddens(rbm, Array{Float64, 2}(X))
  p = vis_means(rbm, h_samples)
  labels = [cidx[1] for cidx in argmax(p[end-c:end, : ], dims = 1)]
end

function class_error(rbm, X, c)
  n = size(X, 2)
  labels = classify(rbm, X, c)
  errors = findall(i -> get_class(X[end-c:end, i]) != labels[i], 1:n)

  length(errors) / size(X, 2)
end

ce = Dict(
  :ys => [(:rbm, :X, :c)],
  :transforms => [class_error],
  :title => "Classification error",
  :incremental => true
)
