using Boltzmann
using StatsBase

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

function get_positive_term(rbm::AbstractRBM{T,V,H}, vis::Array{Union{T, Missing}, 2};
			   n_gibbs = 1) where {T,V,H}

  obs = findall(x -> !ismissing(x), vis)
  miss = findall(x -> ismissing(x), vis)

  v_sample = map(x -> ismissing(x) ? 0.0 : x, vis)
  #v_sample[miss] = zeros(size(miss))

  v_mean = []
  h_mean = []

  for i = 1:n_gibbs
    h_sample, h_mean = sample_hiddens(rbm, v_sample)
    v_sample, v_mean = sample_visibles(rbm, h_sample)
    v_mean[obs] = vis[obs]
  end

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

function update_simple!(rbm::RBM, X::Boltzmann.Mat, dtheta::Tuple, ctx::Dict)
    # apply gradient updaters. note, that updaters all have
    # the same signature and are thus composable
    Boltzmann.grad_apply_learning_rate!(rbm, X, dtheta, ctx)
    # add gradient to the weight matrix
    Boltzmann.update_weights!(rbm, dtheta, ctx)
end
