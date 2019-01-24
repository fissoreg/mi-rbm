using Boltzmann
import Boltzmann.generate
using Statistics
using LinearAlgebra
using Images

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

function missing_mask(d, ratio)
  sample(1:d, floor(Int, d * ratio), replace = false)
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

function generate(rbm, X, mask; n_gibbs = 1)
  lossy = get_lossy(X, mask)
  obs = setdiff(1:size(X, 1), mask)
  
  keep = repeat(obs, outer = (1, size(X, 2)))

  for i = 1:(n_gibbs - 1)
    #lossy = gen_activations(rbm, lossy)
    #lossy, _ = get_positive_term(rbm, lossy, reshape(mask, length(mask), 1); n_gibbs = n_gibbs)
    #lossy, _ = get_positive_term(rbm, X; n_gibbs = n_gibbs)
    #lossy = generate(rbm, lossy, n_gibbs = 1)  

    lossy = rbm.W' * tanh.(rbm.W * lossy .+ rbm.hbias) .+ rbm.vbias
    
    for j in 1:size(lossy, 2), k in 1:length(obs)
      lossy[obs[k], j] = X[obs[k], j]
    end 
  end

  lossy = rbm.W' * tanh.(rbm.W * lossy .+ rbm.hbias) .+ rbm.vbias
  for j in 1:size(lossy, 2), k in 1:length(obs)
    lossy[obs[k], j] = X[obs[k], j]
  end

  lossy
end

function RE(rbm, X, mask; n_gibbs = 1)
  gen = generate(rbm, X, mask, n_gibbs = n_gibbs)
  mean((norm(gen[:, i] - X[:, i]) for i = 1:size(X, 2))) / sqrt(length(mask))
end

# renormalized RE
function rRE(rbm, X, mask; n_gibbs = 1)
  gen = generate(rbm, X, mask, n_gibbs = n_gibbs)
  #mean((norm(((gen[:, i] - X[:, i]) .+ m) .* s) for i = 1:size(X, 2))) / sqrt(length(mask))
  mean((norm(((gen[:, i] - X[:, i])) * 0.1) for i = 1:size(X, 2))) / sqrt(length(mask))
end
