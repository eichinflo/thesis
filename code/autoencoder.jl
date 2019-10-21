# this is intended to be a first project within the Julia universe in order to get
# acquainted with the language and its Flux ML-environment
# we'll write a simple autoencoder that reconstructs the MNIST dataset containing
# handwritten digits
# inspired by https://github.com/FluxML/model-zoo/blob/master/vision/mnist

using Flux, Flux.Data.MNIST
using Flux: @epochs, onehotbatch, mse, throttle
using Base.Iterators: partition
using Juno: @progress
using Images

images = MNIST.images()[1:1200]
# split data into batches
data = [float(hcat(vec.(d)...)) for d in partition(images, 200)]

latent_dimension = 10

# encoder
layer1 = Dense(784, 100, leakyrelu)
layer2 = Dense(100, latent_dimension, leakyrelu)
# decoder 
layer3 = Dense(latent_dimension, 100, leakyrelu)
layer4 = Dense(100, 784, leakyrelu)

autoencoder = Chain(layer1, layer2, layer3, layer4)

loss(data) = mse(autoencoder(data), data)
optimizer = ADAM()

@epochs 10 Flux.train!(loss, params(autoencoder), zip(data), optimizer)

img(x::Vector) = Gray.(reshape(clamp.(x, 0, 1), 28, 28))

function sample()
    # sample 20 random digits
    before = [images[i] for i in rand(1:length(images), 20)]
    after = img.(map(x -> cpu(autoencoder)(float(vec(x))).data, before))
    hcat(vcat.(before, after)...)
end

save("sample.gif", sample())
