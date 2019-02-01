pre = Dict(
  :in => [:rbm],
  :preprocessor => rbm -> (r = svd(rbm.W); (r.U, r.S, r.V)),
  :out => [:U, :s, :V]
)

# declaring wanted plots
weights = Dict(
  :ys => [:W],
  :transforms => [x->x[:]],
  :title => "Weights",
  :seriestype => :histogram,
  :leg => false,
  :nbins => 100
)

PL = Dict(
  :ys => [(:rbm, :X)],
  :transforms => [Boltzmann.pseudo_likelihood],
  :title => "Pseudolikelihood",
  :incremental => true,
  :leg => false
)

dW = Dict(
  :ys => [:dW_prev],
  :transforms => [x -> norm(x)],
  :title => "Gradient norm",
  :incremental => true,
  :leg => false
)

svs = zeros(min(d, nh))'

SVs = Dict(
  :ys => [:s],
  :transforms => [s -> (global svs = vcat(svs, s'); svs)],
  :title => "Singular Values",
  :leg => false
)

SV = Dict(
  :ys => [:s for i=1:1],
  :transforms => [x -> x[i] for i=1:1],
  :incremental => true,
  :title => "Singular values",
  :leg => false
)

features = Dict(
  :ys => [:W],
  :transforms => [W -> reshape_mnist(W')],
  :title => "Features",
  :ticks => nothing
)

#mask = missing_mask(d, ratio)

re = Dict(
  :ys => [(:rbm, :Xt, :missing)],
  :transforms => [(rbm, X, mask) -> vcat(rRE(rbm, X, mask), rRE(rbm, X, mask; n_gibbs = 100))],
  :title => "Reconstruction error",
  :incremental => true,
  :lab => ["CD1" "CD100"]
)

re2 = Dict(
  :ys => [(:rbm, :Xt, :missing)],
  :transforms => [(rbm, X, mask) -> rRE(rbm, X, mask; n_gibbs = 100)],
  :title => "Reconstruction error",
  :incremental => true,
  :lab => "CD100"
)

gen_activations(rbm, X; n_gibbs = 1) = Boltzmann.vis_means(rbm, Boltzmann.gibbs(rbm, X; n_times = 1)[4])

sam = rand(1:size(X, 2), 1)

function reconstruction(rbm, X, mask; n_gibbs = 1)
  s = X[:, sam]
  lossy = get_lossy(s, mask)
  #gen_pin = gen_activations(rbm, s) 
  gen_pin = generate(rbm, lossy, mask; n_gibbs = n_gibbs)
  #obs = setdiff(1:size(X, 1), mask)
  #gen_pin[obs] = s[obs]
  (s, gen_pin)
end

gre = Dict(
  :ys => [(:rbm, :Xt, :missing)],
  :transforms => [(rbm, X, mask) -> reconstruction(rbm, X, mask)],
  :title => "Reconstruction",
  :linetype => :scatter
)

gre2 = Dict(
  :ys => [(:rbm, :Xt, :missing)],
  :transforms => [(rbm, X, mask) -> reconstruction(rbm, X, mask; n_gibbs = 100)],
  :title => "Recon - CD100",
  :linetype => :scatter
)

xrange = [0.1, 0.5, 0.9]

rmse = Dict(
  :ys => [(:rbm, :Xt, :Xtest)],
  :transforms => [(rbm, Xt, Xtest) -> (repeat(xrange, inner = (1,2)), [rRE(rbm, x, missing_mask(d, r), n_gibbs = 100) for r in xrange, x in [Xt, Xtest]])],
  :title => "RMSE",
  :lab => ["Train" "Test"]
)

r10 = [0.1, 0.1]'

rmse10 = Dict(
  :ys => [(:rbm, :Xt, :Xtest)],
  :transforms => [(rbm, Xt, Xtest) -> (global r10 = vcat(r10, [rRE(rbm, x, missing_mask(d, 0.1); n_gibbs = 100) for x in [Xt, Xtest]]'); r10)],
  :title => "RMSE - 0.1",
  :lab => ["Train" "Test"],
  :ylims => [0.06, 0.11]

)

r50 = [0.1, 0.1]'

rmse50 = Dict(
  :ys => [(:rbm, :Xt, :Xtest)],
  :transforms => [(rbm, Xt, Xtest) -> (global r50 = vcat(r50, [rRE(rbm, x, missing_mask(d, 0.5); n_gibbs = 100) for x in [Xt, Xtest]]'); r50)],
  :title => "RMSE - 0.5",
  :lab => ["Train" "Test"],
  :ylims => [0.06, 0.11]

)

r90 = [0.1, 0.1]'

rmse90 = Dict(
  :ys => [(:rbm, :Xt, :Xtest)],
  :transforms => [(rbm, Xt, Xtest) -> (global r90 = vcat(r90, [rRE(rbm, x, missing_mask(d, 0.9); n_gibbs = 100) for x in [Xt, Xtest]]'); r90)],
  :title => "RMSE - 0.9",
  :lab => ["Train" "Test"],
  :ylims => [0.06, 0.11]
)

function project(V, s, X, idxs = [1, 2]) #i = 1, j = 2)
  p = [[dot(V[:, i], X[:, k]) for k in 1:size(X, 2)] for i in idxs]

  # dirty way to build a tuple on  the fly...
  (p[1:end-1]..., p[end])
end

function get_color(label)
  label *= 0.2
  RGB(sum(label[1:5]), sum(label[6:10]), sum(label[11:end]))
end

function color_labels(labels)
  get_color.([labels[:, j] for j in 1:size(labels, 2)])
end

scat3d = Dict(
  :ys => [(:V, :s, :Xt)],
  :transforms => [(V, s, X) -> project(V, s, X, [1, 2, 3])],
  :title => "V1 - V2 - V3",
  :linetype => :scatter,
  :ms => 2,
  :msw => 0,
  :c => color_labels(labels)
)

scatter = Dict(
  :ys => [(:V, :s, :Xt)],
  :transforms => [project],
  :title => "V1 - V2",
  :linetype => :scatter,
  :ms => 2,
  :msw => 0,
  :c => color_labels(labels)
)

scatter2 = Dict(
  :ys => [(:V, :s, :Xt)],
  :transforms => [(V, s, X) -> project(V, s, X, [1, 3])],
  :title => "V1 - V3",
  :linetype => :scatter,
  :ms => 2,
  :msw => 0,
  :c => color_labels(labels)

)

scatter3 = Dict(
  :ys => [(:V, :s, :Xt)],
  :transforms => [(V, s, X) -> project(V, s, X, [2, 3])],
  :transforms => [project],
  :title => "V2 - V3",
  :linetype => :scatter,
  :ms => 2,
  :msw => 0,
  :c => color_labels(labels)

)
