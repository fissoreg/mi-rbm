using DataStructures
using Dates
using JLD

include("sampler_mean.jl")

#TODO: save reporters together with rbm weights

DEFAULT_NH = 50

function make_comb(params)
  if(params |> tail |> length == 0)
    k, vs = head(params)
    return (Cons{Any}(k => v, nil()) for v in vs)
  else
    k, vs = head(params)
    return (Cons{Any}(k => v, comb) for v in vs for comb in make_comb(tail(params)))
  end
end

# doesn't work
#function make_conf(rem, conf = list(), acc = list())
#  r = @match rem begin
#    Cons(k => Cons(hv, tv), t),
#      if length(tv) == 0 && length(t) == 0 end => (Cons(k => list(hv), conf), Cons(k => list(hv), acc))
#    Cons(k => Cons(hv, tv), t),
#      if length(t) == 0 && length(tv) > 0 end => (Cons(k => list(hv), conf), Cons(k => tv, acc))
#    Cons(k => Cons(hv, tv), t),
#      if length(tv) == 0 end => make_conf(t, Cons(k => list(hv), conf), Cons(k => list(hv), acc))
#    Cons(k => Cons(hv, tv), t) => make_conf(t, Cons(k => list(hv), conf), Cons(k => tv, acc))
#  end
#end

# formerly used in conjunction with make_conf
#function inner(params, acc = list())
#  if(all(x -> length(x[2]) == 1, params))
#    return Cons(params, acc)
#  end
#  
#  conf, rem = make_conf(params)
#  println(rem)
#  inner(rem, Cons(conf, acc))
#end

# TODO: come up with a better name!
function build_comb(params)
  l = list([k => list(v...) for (k, v) in params]...)  
  list_conf = make_comb(l)
  (Dict{Any, Any}(conf) for conf in list_conf)
end

#function train(rbm::AbstractRBM{T, V, H}, X, params) where {T, V, H}
#  fit(rbm, X; params)
#end

function save_rbm(fn, rbm, params = Dict())
  fn = "$(fn)_$(Dates.format(now(), "YYYYmmdd_HHMMSS")).jld"
  println(params)
  p = Dict{Any, Any}(zip(keys(params), string.(values(params))))
  println(p)
  save(fn, "W", rbm.W, "hb", rbm.hbias, "vb", rbm.vbias, "params", p)
end

# NOTE: you have to pass the constructor as JLD cannot save functions apparently
# TODO: fix this
function load_rbm(fn, rbm)
  f = load(fn)
  W = f["W"]
  hb = f["hb"]
  vb = f["vb"]

  model = rbm(size(W')...)
  model.W[:] = W
  model.hbias[:] = hb
  model.vbias[:] = vb

  (model, f["params"])
end

function load_all(dir, rbm)
  (load_rbm("$dir/$fn", rbm) for fn in readdir(dir))
end

function compare_rbms(r1, r2, loss)
  l1 = loass(r1)
  l2 = loss(r2)

  (l1 > l2) ? (r1, l1) : (r2, l2)
end

function score_models(models, loss)
  [(rbm, p, loss(rbm, X)) for (rbm, p) in models]
end

function train_all(rbm, X, params; fn = "")
  # number of visible nodes, number of samples
  nv, ns = size(X)

  rbms = []
  combs = build_comb(params)

  for comb in combs
    # extract the number of hidden nodes
    nh = get(params, :nh, DEFAULT_NH)
    #delete!(params, :nh)

    model = rbm(nv, nh; X = X)
    fit(model, X, comb)

    push!(rbms, model)

    if(!(fn == ""))
      save_rbm(fn, model, comb)
    end
  end

  rbms
end
