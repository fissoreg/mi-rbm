using Boltzmann
using StatsBase

using LinearAlgebra
using LinearAlgebra.BLAS

using Boltzmann: @get_or_create, @get_array

import Boltzmann: sample

function sample_hiddens(rbm::AbstractRBM{T,V,H}, vis::Array{T, 2}) where {T,V,H}
  means = rbm.activation(rbm.W * vis .+ rbm.hbias)

  return Boltzmann.sample(H, means), means
end

function sample_visibles(rbm::AbstractRBM{T,V,H}, hid::Array{T, 2}) where {T,V,H}
  signal = rbm.W' * hid  .+ rbm.vbias
  mean = signal #rbm.activation(signal)
  return Boltzmann.sample(V, mean), mean
end

function vis_means(rbm::DiscriminativeRBM{T,V,H,Softmax}, hid::Array{T, 2}) where {T,V,H}
  _, d = size(G(rbm))
  signal = rbm.W' * hid .+ rbm.vbias
  #signal[1:d, :] .+= rbm.vbias

  signal
end

# softmax sampling with "tower" method
# TODO: implement the Gumbel-max trick
function Boltzmann.sample(::Type{Softmax}, means::Array{T, 2}) where T
  means = exp.(means)
  norms = dropdims(sum(means, dims = 1), dims = 1)
  norms .*= rand(length(norms))

  for i = 1:length(norms)
    acc = 0
    for j = 1:size(means, 1)
      acc += means[j, i]
      if acc >= norms[i]
        norms[i] = j
        break;
      end
    end
  end

  s = zeros(size(means))
  s[Int.(norms)] = ones(size(norms))

  s
end

function sample_visibles(rbm::DiscriminativeRBM{T,V,H,L}, hid::Array{T, 2}) where {T,V,H,L}
  _, d = size(G(rbm))
  means = vis_means(rbm, hid)
  means[1:d, :] = rbm.activation(means[1:d, :])

  samples = Matrix(means)

  samples[1:d, :] = Boltzmann.sample(V, means[1:d, :])
  samples[d + 1 : end, :] = sample(L, means[1 + d : end, :])

  return samples, means
end

function get_positive_term(rbm::AbstractRBM{T,V,H}, vis::Array{Union{T, Missing}, 2};
			   n_gibbs = 1) where {T,V,H}

  obs = findall(x -> !ismissing(x), vis)
  miss = findall(x -> ismissing(x), vis)

  v_sample = map(x -> ismissing(x) ? 0.0 : x, vis)

  v_mean = []
  h_mean = []

  for i = 1:n_gibbs
    h_sample, h_mean = sample_hiddens(rbm, v_sample)
    v_sample, v_mean = sample_visibles(rbm, h_sample)
    v_sample[obs] = vis[obs]
  end

  v_mean[obs] = vis[obs]

  v_mean, h_mean
end

function get_negative_term(rbm::AbstractRBM{T,V,H}, vis::Array{T, 2};
			   n_gibbs = 1) where {T,V,H}
  h_sample, h_neg = sample_hiddens(rbm, vis)
  v_neg = []

  for i = 1:n_gibbs
    v_sample, v_neg = sample_visibles(rbm, h_sample)
    h_sample, h_neg = sample_hiddens(rbm, v_sample)
  end

  v_neg, h_neg
end

function persistent_contdiv(rbm::AbstractRBM{T,V,H}, vis::Array{Union{T, Missing}, 2}, ctx::Dict) where {T,V,H}

  n_gibbs = Boltzmann.@get(ctx, :n_gibbs, 1)
  persistent_chain = Boltzmann.@get_array(ctx, :persistent_chain, size(vis), Boltzmann.sample(V, randn(size(vis))))

  if size(persistent_chain) != size(vis)
    println("persistent_chain not initialized")
    # persistent_chain not initialized or batch size changed
    # re-initialize
    persistent_chain = Boltzmann.sample(V, randn(size(vis)))
  end
  
  v_pos, h_pos = get_positive_term(rbm, vis; n_gibbs = n_gibbs) 
  # take negative samples from "fantasy particles"
  #_, _, v_neg, h_neg = Boltzmann.gibbs(rbm, persistent_chain, n_times=n_gibbs)
  v_neg, h_neg = get_negative_term(rbm, persistent_chain; n_gibbs = n_gibbs)
  copyto!(ctx[:persistent_chain], v_neg)

  return Array{T, 2}(v_pos), Array{T, 2}(h_pos), v_neg, h_neg
end

function gradient_classic(rbm::AbstractRBM{T,V,H}, vis, ctx::Dict) where {T,V,H}
    sampler = @get_or_create(ctx, :sampler, persistent_contdiv)
    v_pos, h_pos, v_neg, h_neg = sampler(rbm, vis, ctx)
    dW = @get_array(ctx, :dW_buf, size(rbm.W), similar(rbm.W))
    n_obs = size(vis, 2)
    # same as: dW = ((h_pos * v_pos') - (h_neg * v_neg')) / n_obs
    gemm!('N', 'T', T(1 / n_obs), h_neg, v_neg, T(0.0), dW)
    gemm!('N', 'T', T(1 / n_obs), h_pos, v_pos, T(-1.0), dW)
    # gradient for vbias and hbias
    db = dropdims(sum(v_pos, dims = 2) - sum(v_neg, dims = 2), dims = 2) ./ n_obs
    dc = dropdims(sum(h_pos, dims = 2) - sum(h_neg, dims = 2), dims = 2) ./ n_obs

    db[rbm.d+1:end] = zeros(length(db) - rbm.d)
    return dW, db, dc #db[1:size(G(rbm), 2)]
end

function update_simple!(rbm::AbstractRBM, X::Boltzmann.Mat, dtheta::Tuple, ctx::Dict)
    # apply gradient updaters. note, that updaters all have
    # the same signature and are thus composable
    Boltzmann.grad_apply_learning_rate!(rbm, X, dtheta, ctx)
    # add gradient to the weight matrix
    Boltzmann.update_weights!(rbm, dtheta, ctx)
end
