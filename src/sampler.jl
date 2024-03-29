using Boltzmann
using StatsBase

function sample_hiddens(rbm::AbstractRBM{T,V,H}, v_miss::Array{T, 1},
			obs_bias::Array{T, 1}, missing::Array{Int, 1}) where {T,V,H}
  signal = rbm.W[:, missing] * v_miss ./ size(rbm.W, 2) .+ rbm.hbias .+ obs_bias
  mean = rbm.activation(signal)

  return Boltzmann.sample(H, reshape(mean, :, 1)), mean
end

function sample_visibles(rbm::AbstractRBM{T,V,H}, h_pos::Array{T, 1},
			 missing::Array{Int, 1}) where {T,V,H}
  signal = rbm.W[:, missing]' * h_pos ./ size(rbm.W, 2) .+ rbm.vbias[missing]
  mean = rbm.activation(signal)
  return Boltzmann.sample(V, reshape(mean, :, 1)), mean
end

function get_positive_term(rbm::AbstractRBM{T,V,H}, vis::Array{T, 2},
			   missing::Array{Int, 2};
			   n_gibbs = 1) where {T,V,H}

  h_pos = Array{typeof(vis[1]), 2}(undef, size(rbm.W, 1), size(vis, 2))
  v_miss = Array{typeof(vis[1]), 2}(undef, size(missing)...)

  for m = 1:size(missing,2) 
    miss = missing[:, m]
    observed = setdiff(1:size(rbm.W, 2), miss)

    v_obs = vis[observed, m]
    v_miss[:, m] = zero(vis[miss, m])

    obs_bias = rbm.W[:, observed] * v_obs

    h_pos[:, m], _ = sample_hiddens(rbm, v_miss[:, m], obs_bias, miss)
    v_miss[:, m], _ = sample_visibles(rbm, h_pos[:, m], miss)

    mh = similar(h_pos)
    mv = similar(v_miss)

    for i = 1:n_gibbs
      h_pos[:, m], mh[:, m] = sample_hiddens(rbm, v_miss[:, m], obs_bias, miss)
      v_miss[:, m], mv[:, m] = sample_visibles(rbm, h_pos[:, m], miss)
    end
  end

  v_pos = deepcopy(vis)
  for i = 1:size(missing, 2)
    v_pos[missing[:, i], i] = mv[:, i] #v_miss[:, i]
  end

  #v_pos, h_pos
  mv, mh
end

function persistent_contdiv(rbm::AbstractRBM{T,V,H}, vis::Array{T, 2}, ctx::Dict) where {T,V,H}
  d = size(vis, 1)
  missing = Boltzmann.@get(ctx, :mask, [])
  if(missing == [])
    ratio = Boltzmann.@get(ctx, :ratio, 0.2)
    missing = sample(1:d, floor(Int, d * ratio), replace = false)
  end

  missing = repeat(missing, outer = (1, size(vis, 2)))
  ctx[:missing] = missing

  # corrupting data - NOT NEEDED!!!!
  #vis[missing, :] = 0 #2 * rand(length(missing), size(vis,2)) - 1

  n_gibbs = Boltzmann.@get(ctx, :n_gibbs, 1)
  persistent_chain = Boltzmann.@get_array(ctx, :persistent_chain, size(vis), Boltzmann.sample(V, randn(size(vis))))
  if size(persistent_chain) != size(vis)
    println("persistent_chain not initialized")
    # persistent_chain not initialized or batch size changed
    # re-initialize
    persistent_chain = Boltzmann.sample(V, randn(size(vis)))
  end
  # not persistent, always random!
  #persistent_chain = (2*rand(size(vis))-1)/2
  # NOTE: this is the only line that really changes from Boltzmann.persistent_contdiv!!!
  v_pos, h_pos = get_positive_term(rbm, vis, missing; n_gibbs = n_gibbs)
  # take negative samples from "fantasy particles"
  _, _, v_neg, h_neg = Boltzmann.gibbs(rbm, persistent_chain, n_times=n_gibbs)
  copyto!(ctx[:persistent_chain], v_neg)
  return v_pos, h_pos, v_neg, h_neg
end

function update_simple!(rbm::RBM, X::Boltzmann.Mat, dtheta::Tuple, ctx::Dict)
    # apply gradient updaters. note, that updaters all have
    # the same signature and are thus composable
    Boltzmann.grad_apply_learning_rate!(rbm, X, dtheta, ctx)
    # add gradient to the weight matrix
    Boltzmann.update_weights!(rbm, dtheta, ctx)
end
