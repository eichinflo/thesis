# this is intended to be a first project within the Julia universe in order to get
# acquainted with the language and its Flux ML-environment
# we'll write a variational autoencoder that reconstructs the MNIST dataset containing
# handwritten digits
# partially inspired by https://github.com/FluxML/model-zoo/blob/master/vision/mnist

using Flux, Flux.Data.MNIST
using Flux: @epochs, onehotbatch, mse, throttle
using Base.Iterators: partition
using Juno: @progress
using Images

images = MNIST.images()[1:1000]
# split data into batches
data = [float(hcat(vec.(d)...)) for d in partition(images, 50)]

latent_dimension = 10

# reparametrization trick
function sampling(mean, variance)
    epsilon = rand(latent_dimension)
    return mean .+ exp.(0.5 .* variance) .* epsilon
end

# encoder
layer1 = Dense(784, 100, relu)
layer2 = Dense(100, latent_dimension, relu)
encoder(x) = layer2(layer1(x))

z_mean = Dense(latent_dimension, latent_dimension, relu)
z_log_variance = Dense(latent_dimension, latent_dimension, relu)
z(x) = sampling(z_mean(x), z_log_variance(x))

# decoder 
layer3 = Dense(latent_dimension, 100, relu)
layer4 = Dense(100, 784, sigmoid)
decoder(x) = layer4(layer3(x))

autoencoder(x) = decoder(z(encoder(x)))
loss(data) = mse(autoencoder(data), data)
optimizer = ADAM()

@epochs 1000 Flux.train!(loss, params(autoencoder), zip(data), optimizer)

img(x::Vector) = Gray.(reshape(clamp.(x, 0, 1), 28, 28))

function sample()
    # sample 20 random digits
    before = [images[i] for i in rand(1:length(images), 20)]
    after = img.(map(x -> cpu(autoencoder)(float(vec(x))).data, before))
    hcat(vcat.(before, after)...)
end

save("vae_out.gif", sample())
