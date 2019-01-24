using Boltzmann
using StatsBase

function sample_hiddens(rbm::AbstractRBM{T,V,H}, v_miss::Array{Union{T, Missing}, 1},
			obs_bias, missing::Array{Int, 1}) where {T,V,H}
  signal = rbm.W[:, missing] * v_miss .+ rbm.hbias .+ obs_bias
  mean = tanh.(signal) #rbm.activation(signal)
  p = (mean .+ 1) ./ 2
  r = dropdims(Boltzmann.sample(H, reshape(p, :, 1)), dims = 2), mean
  return Array{Union{T, Missing}, 1}(r[1]), r[2]
end

function sample_visibles(rbm::AbstractRBM{T,V,H}, h_pos::Array{Union{T, Missing}, 1}, missing::Array{Int, 1}) where {T,V,H}
  signal = rbm.W[:, missing]' * h_pos .+ rbm.vbias[missing]
  mean = signal #rbm.activation(signal)
  r = dropdims(Boltzmann.sample(V, reshape(mean, :, 1)), dims = 2), mean
  return Array{Union{T, Missing}, 1}(r[1]), r[2]
end

function sample_hiddens(rbm::AbstractRBM{T,V,H}, vis::Array{T, 2}) where {T,V,H}
  signal = rbm.W * vis .+ rbm.hbias
  mean = tanh.(signal) #rbm.activation(signal)
  p = (mean .+ 1) ./ 2
  return Boltzmann.sample(H, p), mean
end

function sample_visibles(rbm::AbstractRBM{T,V,H}, hid::Array{T, 2}) where {T,V,H}
  signal = rbm.W' * hid  .+ rbm.vbias
  mean = signal #rbm.activation(signal)
  return Boltzmann.sample(V, mean), mean
end

# NOTE: this needs to be vectorized to improve performance!
function get_positive_term(rbm::AbstractRBM{T,V,H}, vis::Array{Union{T, Missing}, 2};
			   n_gibbs = 1) where {T,V,H}

  arche_miss = findall(x -> ismissing(x), vis[:, 1])
  mh = Array{T, 2}(undef, size(rbm.W, 1), size(vis, 2))
  m_v = [] 

  v_pos = deepcopy(vis)

  for m = 1:size(vis, 2) 
    miss = findall(x -> ismissing(x), vis[:, m])
    observed = setdiff(1:size(rbm.W, 2), miss)

    v_obs = vis[observed, m]
    v_miss = zero(vis[miss, m])

    obs_bias = rbm.W[:, observed] * v_obs

    h_pos, _ = sample_hiddens(rbm, v_miss, obs_bias, miss)
    v_miss, _ = sample_visibles(rbm, h_pos, miss)

    for i = 1:n_gibbs
      h_pos, mh[:, m] = sample_hiddens(rbm, v_miss, obs_bias, miss)
      v_miss, m_v = sample_visibles(rbm, h_pos, miss)
    end

    v_pos[miss, m] = m_v 
  end

  v_pos, mh
end

function get_negative_term(rbm::AbstractRBM{T,V,H}, vis::Array{T, 2};
			   n_gibbs = 1) where {T,V,H}
  h_sample, h_neg = sample_hiddens(rbm, vis)
  #v_sample, v_neg = sample_visibles(rbm, h_sample)
  v_neg = []

  for i = 1:n_gibbs
    #h_sample, h_neg = sample_hiddens(rbm, vis)
    v_sample, v_neg = sample_visibles(rbm, h_sample)
    h_sample, h_neg = sample_hiddens(rbm, v_sample)
  end

  v_neg, h_neg
end


function persistent_contdiv(rbm::AbstractRBM{T,V,H}, vis, ctx::Dict) where {T,V,H}

  n_gibbs = Boltzmann.@get(ctx, :n_gibbs, 1)
  persistent_chain = Boltzmann.@get_array(ctx, :persistent_chain, size(vis), Boltzmann.sample(V, randn(size(vis))))

  if size(persistent_chain) != size(vis)
    println("persistent_chain not initialized")
    # persistent_chain not initialized or batch size changed
    # re-initialize
    persistent_chain = Boltzmann.sample(V, randn(size(vis)))
  end

  v_pos, h_pos = get_positive_term(rbm, vis; n_gibbs = n_gibbs) 
  v_neg, h_neg = get_negative_term(rbm, persistent_chain; n_gibbs = n_gibbs)
  copyto!(ctx[:persistent_chain], v_neg)

  return Array{T, 2}(v_pos), Array{T, 2}(h_pos), v_neg, h_neg
end

function update_simple!(rbm::RBM, X::Boltzmann.Mat, dtheta::Tuple, ctx::Dict)
    # apply gradient updaters. note, that updaters all have
    # the same signature and are thus composable
    Boltzmann.grad_apply_learning_rate!(rbm, X, dtheta, ctx)
    # add gradient to the weight matrix
    Boltzmann.update_weights!(rbm, dtheta, ctx)
end
