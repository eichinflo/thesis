# this is a copy of the jupyter notebook in this folder for running the code
# on the server
using MAT, Plots, ColorSchemes

# by broichha@imbi.uni-freiburg.de
file = matopen("../dummydata/dummyData2000.mat") 
data = read(file, "pix3D")
close(file)

# store data in WHCN order (width, height, channel, batches)
data = reshape(data, (60, 60, 1, size(data, 3)))
print("Data imported.\n")

print("Data dimensions: ", size(data), "\n")

using Flux
using Flux: @epochs, mse
using Base.Iterators: partition 
using Images
using Flux: Conv, MaxPool, Dense, ConvTranspose
using BSON: @save, @load

# hyperparameters TODO: all params over here
changed = true; # set to true, if model parameters were changed 
latent_dimension = 15;
epochs = 3;
out_ch1 = 8;
learning_rate = 0.001;
batch_size = 20;

layer1 = Conv((3, 3), 1=>out_ch1, relu, pad=1);
sample1 = layer1(data[:, :, :, 1:10])
print("Size of output at layer1: ", size(sample1), "\n");
pool1 = MaxPool((3, 3));
sample2 = pool1(sample1)
out_size = size(sample2) 
print("Size of output after max pooling: ", out_size, "\n")

conv1(x) = pool1(layer1(x));

flattened_size = out_size[1] * out_size[2] * out_size[3]; 
layer2 = Dense(flattened_size, latent_dimension, relu);
sample3 = layer2(reshape(sample2, (flattened_size, 10)));
print("Size of output at layer2: ", size(sample3), "\n")

encoder(x) = layer2(reshape(conv1(x), (flattened_size, :)));

# reparametrization trick in vae
function sampling(mean, variance) 
    epsilon = rand(latent_dimension) 
    return mean .+ exp.(0.5 .* variance) .* epsilon
end;

z_mean = Dense(latent_dimension, latent_dimension, relu) 
z_log_var = Dense(latent_dimension, latent_dimension, relu) 
z(x) = sampling(z_mean(x), z_log_var(x)) 
sample4 = z(sample3)
print("Size of z: ", size(sample4), "\n")

# decoder
layer3 = Dense(latent_dimension, 400, relu) 
layer4 = ConvTranspose((3, 3), 1=>out_ch1, relu, stride=3) 
sample5 = layer4(reshape(layer3(sample4), (20, 20, 1, 10))) 
print("Size of output at layer 4: ", size(sample5), "\n") 
layer5 = ConvTranspose((3, 3), out_ch1=>1, relu, pad=1); 
sample6 = layer5(sample5);
print("Size of output at layer 5: ", size(sample6), "\n")

decoder(x) = layer5(layer4(reshape(layer3(x), (20, 20, 1, :)))); 

cvae(x) = decoder(z(encoder(x))); 
loss(x) = mse(cvae(x), x);
optimizer = Flux.ADAM(learning_rate);

# for some weird reason ∇maxpool has a problem with Float64 data as dy 
# seems to be computed in Float32 and the function takes only arrays 
# with values of same type
data = convert.(Float32, data);

# this also looks a bit hacked, but it does the job of bringing the data 
# into the shape preferred by Flux.train! and creating batches (we'll 
# find a better solution)
batches = [reshape(data[:, :, :, ((i-1)*batch_size+1):(i*batch_size)], (60, 60, 1, batch_size)) for i in 1:size(data, 4)÷batch_size];

# load previous parameters, if existing
if isfile("cvae_model.bson") && !changed
   @load "cvae_model.bson" cvae 
end

print("\nTraining...\n")
@epochs epochs Flux.train!(loss, params(cvae), zip(batches), optimizer)

# store model for later
@save "cvae_model.bson" cvae

print("done.\n\n")

sequence = [reshape(data[:, :, :, i], (60, 60, 1, 1)) for i in 1:size(data, 4)];
anim1 = @animate for i=1:2000
    output = cvae(sequence[i]) 
    output = reshape(Flux.Tracker.data(output), 60, 60) 
    Plots.heatmap(output, seriescolor=cgrad(ColorSchemes.gray.colors))
end;
gif(anim1, "out.gif", fps=15)
