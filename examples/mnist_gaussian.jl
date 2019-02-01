using MLDatasets

include("../src/sampler_mean.jl")
include("../src/utils.jl")

ns = 1000

X, Y = MNIST.traindata()
X = reshape(X, 784, size(X, 3))[:, 1:ns]

d, ns = size(X)
c = size(Y, 1)

nh = 100
sigma = 0.01
n_epochs = 300
lr = 1e-4
batch_size = 50
randomize = true
n_gibbs = 1
ratio = 0.5
n_mon = 1

X = 255*X
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


#m = mean(X, dims = 2)
#s = mapslices(std, X, dims = 2)
#X = (X .- m) ./ s
#map!(x -> isnan(x) ? 0 : x, X, X)

Xtrue = deepcopy(X)

X = Array{Union{Missing, Float64}, 2}(X)

for i in 1:size(X, 2)
  mask = missing_mask(d, ratio)
  X[mask, i] = [missing for i = 1:length(mask)]
end

include("mnist_report.jl")

rbm = RBM(Float64, Distributions.Normal, Boltzmann.IsingSpin, Boltzmann.IsingActivation, d, nh; sigma = sigma)

# getting the reporter
plots_list = [ws, dW, PL, SVs, features, reconstructions]
lossy = map(x -> ismissing(x) ? 0 : x, X)
vr = VisualReporter(rbm, n_mon, plots_list, pre = pre, init = Dict(:X => X, :Xtrue => Xtrue, :lossy => lossy, :dW_prev => zeros(nh, size(X, 1))))

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

try
  mkdir("log")
end

filename = "log/mnist_h$(nh)_r$(ratio)_lr$(lr)_bs$(batch_size)_ng$(n_gibbs)_e$(n_epochs)"
save("$filename.jld", "w", rbm.W, "vbias", rbm.vbias, "hbias", rbm.hbias, "ratio", ratio)
mp4(vr.anim, "$filename.mp4", fps=5)
