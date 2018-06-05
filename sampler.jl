using Boltzmann
using StatsBase

function sample_hiddens(rbm::AbstractRBM{T,V,H}, v_miss::Array{T, 1},
			obs_bias::Array{T, 1}, missing::Array{Int, 1}) where {T,V,H}
  signal = rbm.W[:, missing] * v_miss .+ rbm.hbias .+ obs_bias
  mean = rbm.activation(signal)

  return Boltzmann.sample(H, reshape(mean, :, 1))
end

function sample_visibles(rbm::AbstractRBM{T,V,H}, h_pos::Array{T, 1},
			 missing::Array{Int, 1}) where {T,V,H}
  signal = rbm.W[:, missing]' * h_pos .+ rbm.vbias[missing]
  mean = rbm.activation(signal)

  return Boltzmann.sample(V, reshape(mean, :, 1))
end

function get_positive_term(rbm::AbstractRBM{T,V,H}, vis::Array{T, 2},
			   missing::Array{Int, 2};
			   n_gibbs = 1) where {T,V,H}
  h_pos = Array{typeof(vis[1]), 2}(size(rbm.W, 1), size(vis, 2))
  v_miss = Array{typeof(vis[1]), 2}(size(missing)...)

  for m = 1:size(missing,2), i = 1:n_gibbs
    miss = missing[:, m]
    observed = setdiff(1:length(miss), miss)

    v_obs = vis[observed, m]
    v_miss[:, m] = vis[miss, m]

    obs_bias = rbm.W[:, observed] * v_obs

    h_pos[:, m] = sample_hiddens(rbm, v_miss[:, m], obs_bias, miss)
    v_miss[:, m] = sample_visibles(rbm, h_pos[:, m], miss)
  end

  v_pos = deepcopy(vis)
  for i = 1:size(missing, 2)
    v_pos[missing[:, i], i] = v_miss[:, i]
  end

  v_pos, h_pos
end

function persistent_contdiv(rbm::AbstractRBM{T,V,H}, vis::Array{T, 2}, ctx::Dict) where {T,V,H}
  d = size(vis, 1)
  ratio = Boltzmann.@get(ctx, :ratio, 0.2)
  missing = hcat((sample(1:d, Int(floor(d * ratio)), replace = false) for i = 1:size(vis, 2))...)
  # corrupting data
  vis[missing, :] = 2 * rand(length(missing), size(vis,2)) - 1

  n_gibbs = Boltzmann.@get(ctx, :n_gibbs, 1)
  persistent_chain = Boltzmann.@get_array(ctx, :persistent_chain, size(vis), vis)
  if size(persistent_chain) != size(vis)
    println("persistent_chain not initialized")
    # persistent_chain not initialized or batch size changed
    # re-initialize
    persistent_chain = Boltzmann.sample(V, rand(size(vis)))
  end
  # NOTE: this is the only line that really changes from Boltzmann.persistent_contdiv!!!
  v_pos, h_pos = get_positive_term(rbm, vis, missing; n_gibbs = n_gibbs)
  # take negative samples from "fantasy particles"
  _, _, v_neg, h_neg = Boltzmann.gibbs(rbm, persistent_chain, n_times=n_gibbs)
  copy!(ctx[:persistent_chain], v_neg)
  return v_pos, h_pos, v_neg, h_neg
end
