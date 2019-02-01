using DelimitedFiles

function load_yeast(; set = "train")
  data = readdlm("yeast_$set.svm", String)

  y = data[:, 1]
  x = data[:, 2:end]
  
  ns = length(y)
  
  comp_labels = [parse.(Int, split(s, ",")) for s in y]
  
  labels = zeros(14, ns)
  
  for i in 1:ns, l in comp_labels[i]
    labels[l + 1, i] = 1
  end
  
  x = map(s -> parse(Float64, split(s, ":")[2]), x)

  x', labels
end
