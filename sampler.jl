using Boltzmann
using StatsBase

function sample_hiddens(rbm::AbstractRBM{T,V,H}, v_miss::Array{T, 2},
			obs_bias::Array{T, 2}, missing::Array{Int, 1}) where {T,V,H}
  signal = rbm.W[:, missing] * v_miss .+ rbm.hbias .+ obs_bias
  mean = rbm.activation(signal)

  return Boltzmann.sample(H, mean)
end

function sample_visibles(rbm::AbstractRBM{T,V,H}, h_pos::Array{T, 2},
			 missing::Array{Int, 1}) where {T,V,H}
  signal = rbm.W[:, missing]' * h_pos .+ rbm.vbias[missing]
  mean = rbm.activation(signal)

  return Boltzmann.sample(V, mean)
end

function get_positive_term(rbm::AbstractRBM{T,V,H}, vis::Array{T, 2},
			   missing::Array{Int, 1};
			   n_gibbs = 1) where {T,V,H}
  observed = setdiff(1:length(missing), missing)

  v_obs = vis[observed, :]
  v_miss = vis[missing, :]

  obs_bias = rbm.W[:, observed] * v_obs

  h_pos = []
  for i = 1:n_gibbs
    h_pos = sample_hiddens(rbm, v_miss, obs_bias, missing)
    v_miss = sample_visibles(rbm, h_pos, missing)
  end

  v_pos = deepcopy(vis)
  v_pos[missing, :] = v_miss

  v_pos, h_pos
end

function persistent_contdiv(rbm::AbstractRBM{T,V,H}, vis::Array{T, 2}, ctx::Dict) where {T,V,H}
  d = size(vis, 1)
  ratio = Boltzmann.@get(ctx, :ratio, 0.2)
  missing = sample(1:d, Int(floor(d * ratio)), replace = false)
  # corrupting data
  vis[missing, :] = rand(length(missing), size(vis,2))

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
