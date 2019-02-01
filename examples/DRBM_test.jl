using MLDatasets

include("DRBM.jl")
include("sampler_disc.jl")

# `c` is the number of classes
function k_hot_encoding(labels, c)
  kh = zeros(c, length(labels))
  for j in 1:length(labels), i in labels[j]
    kh[i, j] = 1
  end

  kh
end

get_class(label) = findfirst(x -> x == 1, label)

ns = 10000

X, Y = MNIST.traindata()

X = reshape(X, 784, size(X, 3))[:, 1:ns]
Y = Y[1:ns] .+ 1
Y = k_hot_encoding(Y, 10)

#X = 2X .- 1
#Y = 2Y .- 1

d, ns = size(X)
c = size(Y, 1)

X = vcat(X, Y)
X = Array{Union{Missing, Float64}, 2}(X)

nh = 6000
sigma = 0.0001
n_epochs = 10000
lr = 1e-4
batch_size = 100
randomize = true
n_gibbs = 1
ratio = 0 #, 0.05, 0.1, 0.2]
n_mon = 10

include("mnist_report.jl")

rbm = DRBM(d, nh, c; X = X)

# getting the reporter
plots_list = [ws, dW, SVs, features, reconstructions, lbias, modes, lmodes, ce] #, re, ce, features, gre, gre2]
vr = VisualReporter(rbm, n_mon, plots_list, pre = pre, init=Dict(:X => X, :Xt => X, :persistent_chain => X[:, 1:batch_size], :dW_prev => zeros(nh, size(X, 1)), :missing => missing_mask(d, ratio), :c => 10))

fit(rbm, X;
  n_epochs = n_epochs,
  lr=lr,
  batch_size=batch_size,
  randomize=randomize,
  scorer = (rbm, vis) -> 0,
  gradient = gradient_classic,
  sampler = persistent_contdiv,
  update = update_simple!,
  reporter = vr,
  X = X,
  #ratio = ratio,
  n_gibbs = n_gibbs
)

filename = "log/mnist_h$(nh)_r$(ratio)_lr$(lr)_bs$(batch_size)_ng$(n_gibbs)_e$(n_epochs)"

save("$filename.jld", "w", rbm.W, "vbias", rbm.vbias, "hbias", rbm.hbias, "ratio", ratio)

mp4(vr.anim, "$filename.mp4", fps=5)
