using Boltzmann
import Boltzmann.generate
using Statistics
using LinearAlgebra
using Images

# `c` is the number of classes
function k_hot_encoding(labels, c)
  kh = zeros(c, length(labels))
  for j in 1:length(labels), i in labels[j]
    kh[i, j] = 1
  end

  kh
end

get_class(label) = findfirst(x -> x == 1, label)

function missing_mask(d, ratio)
  sample(1:d, floor(Int, d * ratio), replace = false)
end

function get_lossy(X, ratio::Float64)
  map(x -> rand() < ratio ? missing : x, X)
end

function get_lossy(X, mask)
  lossy = deepcopy(X)

  ##### WTF?!?!?!?! ################################
  # WHY DOESN'T THIS WORK?!?!?!?!
  #throw_away = repeat(mask, outer = (1, size(X, 2)))
  #lossy[throw_away] = zero(throw_away)
  ##################################################
  for j in 1:size(lossy, 2), i in 1:length(mask)
    lossy[mask[i], j] = 0.0
  end

  lossy
end

gen_activations(rbm, X; n_gibbs = 1) = Boltzmann.vis_means(rbm, Boltzmann.bs(rbm, X; n_times = 1)[4])

function generate(rbm::RBM{T}, X::Array{Union{T, Missing}, 2}; n_gibbs = 1) where T

  obs = findall(x -> !ismissing(x), X)
  lossy = map(x -> ismissing(x) ? 0.0 : x, X)

  for i = 1:(n_gibbs - 1)
    lossy = rbm.W' * tanh.(rbm.W * lossy .+ rbm.hbias) .+ rbm.vbias
    lossy[obs] = X[obs]
  end

  lossy = rbm.W' * tanh.(rbm.W * lossy .+ rbm.hbias) .+ rbm.vbias
  lossy[obs] = X[obs]
  
  lossy

end

function RE(rbm, X, mask; n_gibbs = 1)
  gen = generate(rbm, X, mask, n_gibbs = n_gibbs)
  mean((norm(gen[:, i] - X[:, i]) for i = 1:size(X, 2))) / sqrt(length(mask))
end

function rRE(rbm, X, Xm; n_gibbs = 1)
  gen = generate(rbm, Xm; n_gibbs = n_gibbs)
  #mean((norm(((gen[:, i] - X[:, i]) .+ m) .* s) for i = 1:size(X, 2))) / sqrt(length(mask))
  r = length(findall(x -> ismissing(x), Xm[:, 1]))
  mean((norm(((gen[:, i] - X[:, i])) * 0.1) for i = 1:size(X, 2))) / sqrt(r)
end
