using Boltzmann
using Distributions

using Boltzmann: logistic, sample, @get, @get_array, Degenerate, vbias_init
struct Softmax end

# NOTE:
# W and U (and lbias) need not be separate matrices. By using a single
# matrix and masking appropriately, all the training "utility" functions
# in Boltzmann (weight decay etc...) could be reused.
# NOTE 2: maybe `d` is not a good idea? Should be improved, in any case.
mutable struct DiscriminativeRBM{T,V,H,L} <: AbstractRBM{T,V,H} 
  W::Matrix{T}
  vbias::Vector{T}
  hbias::Vector{T}
  #lbias::Vector{T}
  activation::Function
  d::Int
end

"""
Construct DiscriminativeRBM. Parameters:

 * T - type of RBM parameters (e.g. weights and biases; by default, Float64)
 * V - type of visible units
 * H - type of hidden units
 * L - type of labels' units
 * activation - activation function to use
 * n_vis - number of visible units
 * n_hid - number of hidden units
 * n_lab - number of units to encode labels

Optional parameters:

 * sigma - variance to use during parameter initialization

"""
function DiscriminativeRBM(T::Type, V::Type, H::Type, L::Type, activation::Function,
			   n_vis::Int, n_hid::Int, n_lab::Int; sigma=0.01, X=[])

	vbias = length(X) == 0 ? zeros(n_vis + n_lab) : vbias_init(V, X) #[1:n_vis, :])

    DiscriminativeRBM{T,V,H,L}(map(T, rand(Normal(0, sigma), n_hid, n_vis + n_lab)),
			       vbias, zeros(n_hid),
			       #zeros(n_lab),
			       activation,
                               n_vis)
end

DRBM(n_vis::Int, n_hid::Int, n_lab::Int; sigma = 0.01, X = []) =
  DiscriminativeRBM(Float64, Degenerate, Bernoulli, Softmax, logistic,
		    n_vis, n_hid, n_lab; sigma = sigma, X = X)

# utility functions to get generative (G) and discriminative (D) weights
# TODO: find a better way, maybe?
G(rbm) = view(rbm.W, :, 1:rbm.d)
D(rbm) = @view rbm.W[:, rbm.d:end]
#view(rbm.W, :, length(rbm.vbias):end)

#=

#function up_down(rbm::DiscriminativeRBM{T,V,H,L}, vis::Matrix{T}; n_times = 1)
#	  where {T,V,H,L}
#  v_pos = vis
#  h_pos = prop_up(rbm::DiscriminativeRBM, vis::Matrix)
#
#  for i = 1:n_times
#    h_sample = sample(H, h_pos)
#    y = py(rbm::DiscriminativeRBM, hid::Matrix)
#    x_neg = sample_visibles(rbm, h_sample)
#    v_neg = __cat data and labels__
#    h_neg = 
#  end
#
#end

# no need!!!!
function discr_contdiv(rbm::DiscriminativeRBM, vis::Matrix, ctx::Dict)
  n_gibbs = @get(ctx, :n_gibbs, 1)
  v_pos, h_pos, v_neg, h_neg = up_down(rbm, vis, n_times=n_gibbs)
end

x(rbm, vis) = vis[1:size(rbm.W, 2), :]
lab(rbm, vis) = vis[size(rbm.W, 2) + 1 : end, :]

function prop_up(rbm::DiscriminativeRBM, vis::Matrix)
  p = rbm.W * x(vis) .+ rbm.U * lab(vis) .+ rbm.hbias
  rbm.activation(p)
end

# propagating x (data with no labels) up
function prop_x_up(rbm::DiscriminativeRBM, vis::Matrix)
  rbm.W * x(rbm, vis) .+ rbm.hbias
end

# helper function for p_y_given_x
function term(y, hxys)
  prod(x -> 1 + exp(x), hxys[y], dims = 1)[1]
end

function p_y_given_x(rbm::DiscriminativeRBM, y, hxys)
  num = term(y, hxys) #exp(rbm.lbias[y]) * 
  den = sum((term(i, hxys) for i in 1:size(rbm.U, 2)))

  num / den
end

function gradient_disc(rbm::DiscriminativeRBM{T,V,H,L}, vis, ctx::Dict) where {T,V,H,L}
  ns = size(vis, 2)   # number of samples (batch size)
  nl = size(rbm.U, 2) # number of labels

  # no sampling, the gradient can be computed analytically
  hx = prop_x_up(rbm, vis)
  h_pos = rbm.activation(hx .+ rbm.U * lab(rbm, vis))

  # sigmoid argument of p(h|y,x), taking `W * x + vbias` as argument (hx)
  hxys = [hx .+ rbm.U[:, i] for i in 1:nl]
  ys = [p_y_given_x(rbm, i, hxys) for i = 1:nl]

  data = x(rbm, vis)
  h_negs = map(hxy -> rbm.activation(hxy), hxys)

  pos = h_pos * data' 
  #println(p_y_given_x(rbm, 1, hxys))
  neg = sum(h_negs[i] * data' * ys[i] for i = 1:nl)

  dW = (pos - neg) / ns

  # dU_kl = h_k^l delta_yl - h_k^l y_l
  hxys_mean = [rbm.activation(mean(h, dims = 2)) for h in hxys]
  # mean weigthed by y
  labels = lab(rbm, vis)
  # probably this can be improved!
  hxys_w_mean = hcat((rbm.activation(mean((hxys[i]' .* labels[i, : ])', dims = 2)) for i = 1:nl)...)

  dU_neg = [hxys_mean[j][i] * ys[j] for i = 1:size(rbm.U, 1), j = 1:nl]
  dU = hxys_w_mean - dU_neg

  dc_negs = sum((h_negs[i] * ys[i] for i = 1:nl))
  dc = mean(h_pos, dims = 2) .- mean(dc_negs, dims = 2)
  dc = dropdims(dc, dims = 2)

  #dd = 1 .- mean(rbm.lbias .* labels, dims = 2)
  #dd = dropdims(dd, dims = 2)
  #dd *= 0 #zeros(length(rbm.lbias))
  #println(dd)

  dW, dU, dc#, dd
end

function update_weights!(rbm::DiscriminativeRBM, dtheta::Tuple, ctx::Dict)
    dW, dU, dc = dtheta
    println("IN")
    rbm.W += dW
    rbm.U += dU
    rbm.hbias += dc
    #rbm.lbias += dd

    println("IN2")

    # save previous dW
    dW_prev = @get_array(ctx, :dW_prev, size(dW), similar(dW))
    copyto!(dW_prev, dW)
end

function update_classic!(rbm::DiscriminativeRBM, X::Matrix, dtheta::Tuple, ctx::Dict)
    # add gradient to the weight matrix
    update_weights!(rbm, dtheta, ctx)
end

function score_samples(rbm::DiscriminativeRBM{T,V,H,L}, vis::Matrix;
                          sample_size=10000) where {T,V,H,L}
  println("IN")
  Boltzmann.score_samples(rbm, x(rbm, vis))
end

function pseudo_likelihood(rbm::DiscriminativeRBM, X)
    println("IN")
    m = mean(score_samples(rbm, X))
    return m == -Inf ? 0 : m/size(X,2)
end

=#
