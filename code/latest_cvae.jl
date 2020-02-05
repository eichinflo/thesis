# Convolutional Variational Auto Encoder model for dummydata


############################# hyperparameters ##################################
random_seed = 42

latent_dim1 = 8  # number of z-variaables for vae1
latent_dim2 = 1  # number of z-variables for vae2

inter1 = 8  # size of image after interaction
inter2 = 6

batch_size = 10

plot_recon = true

store_params = false
load_params = false
not_on_server = true  # we don't want to plot on server, leads to problems


plot_input_path = "input.gif"
plot_output_path = "output.gif"
# change accordingly on your machine
path_to_project = "/home/flo/projects/thesis/code/"
params_path = "params/"
plot_path = "plots/"
################################################################################


# load libraries
using MAT  # for importing dummydata
using Flux  # for model and learning
using Flux: @epochs, mse
using Base.Iterators: partition  # for creating batches
using Flux: Conv, MaxPool, Dense, ConvTranspose
using JLD2  # store model parameters for later

using Distributions  # to compute loss function


# Estend Distributions with a numericallz stable logpdf for 'p' close to 1 or 0
my_logpdf(b::Bernoulli, y::Any) =
    y * log(b.p + eps(Float32)) + (1f0 - y) * log(1 - b.p + eps(Float32))

# helper function to compute tensor size after applying f on sample
# in other words: compute the product of the first three size-dims of f(sample)
function no_of_entries(f)
    sample = data[:, :, :, 1:2]
    sample = f(sample)
    no_of_entries = size(sample)[1] * size(sample)[2] * size(sample)[3]
end

# helper function to compute result of applying f on X and reshaping to given size
apply_and_reshape(X, f, reshaped_size) = reshape(f(X), reshaped_size)


########################### load data ##########################################
file = matopen("/home/flo/projects/thesis/dummydata/dummyData2000.mat")
data = read(file, "pix3D")
close(file)

# store data in WHCN order (width, height, channel, batches)
data = reshape(data, (60, 60, 1, size(data, 3)))

data = convert.(Float64, data)

# binarize data
data = (data .- min(data...)) ./ (max(data...) - min(data...))
data = data .> 0.5

# create an array of array-batches of size batch_size
batches = [reshape(
    data[:, :, :, ((i-1)*batch_size+1):(i*batch_size)],
    (60, 60, 1, batch_size),
) for i = 1:size(data, 4)÷batch_size]


########################### model ##############################################
# there is two VAEs, vae1 with a small receptive field for modelling foreground
# signals and vae2 with bigger receptive field to model background signal
# the naming scheme of the respective comonents works accordingly

## vae1 with small receptive field of size (5, 5)
conv_enc1 = Conv((6, 6), 1 => 32, relu, pad = 2, stride=(3,3))  # e.g. convolution of encoder of vae1
pool_enc1 = MaxPool((3, 3), stride = (2, 2), pad=0)
conv2_enc1 = Conv((2, 2), 32 => latent_dim1, stride=(1, 1), pad=0)

# compute data size after pooling in order to determine size after flattening
flattened_size1 = no_of_entries(x -> conv2_enc1(pool_enc1(conv_enc1(x))))
# mean and log-variance of vae1's z-variable
μ1 = Dense(flattened_size1, latent_dim1)
logσ1 = Dense(flattened_size1, latent_dim1)
# later on, we want to let z-variables interact via this layer TODO
interaction1 = Dense(latent_dim1, inter1^2 * latent_dim1)
# 'deconvolutions' of vae1's decoder
transp1_dec1 = ConvTranspose((2, 2), latent_dim1 => 32, relu, stride = (1, 1), pad = 0)
transp2_dec1 = ConvTranspose((3, 3), 32 => 32, relu, stride = (2, 2), pad = 0)
transp3_dec1 = ConvTranspose((6, 6), 32 => 1, sigmoid, stride = (3, 3), pad=0)

# final decoder layer, we might want to drop this actually TODO
dense_dec1 = Dense(60 * 60, 60 * 60, sigmoid)

#### for debugging
function debug()
    s = data[:, :, :, 1:2]
    s_conv1 = conv_enc1(s)
    s_pool1 = pool_enc1(s_conv1)
    s_conv2 = conv2_enc1(s_pool1)
    s_z = z.(μ1(enc1(s)), logσ1(enc1(s)))
    s_int = int1(s_z)
    s_transp1 = transp1_dec1(s_int)
    s_transp2 = transp2_dec1(s_transp1)
    s_transp3 = transp3_dec1(s_transp2)
    print("s_conv1: ", size(s_conv1), "\n")
    print("s_pool1: ", size(s_pool1), "\n")
    print("s_conv2: ", size(s_conv2), "\n")
    print("s_z: ", size(s_z), "\n")
    print("s_int: ", size(s_int), "\n")
    print("s_transp1: ", size(s_transp1), "\n")
    print("s_transp2: ", size(s_transp2), "\n")
    print("s_transp3: ", size(s_transp3), "\n")
end
####

## vae2 with big receptive field of size ??, analogous to vae1
conv_enc2 = Conv((30, 30), 1 => 4, relu, pad = 2, stride = (2, 2))
pool_enc2 = MaxPool((6, 6), stride = (6, 6))
flattened_size2 = no_of_entries(
    data[:, :, :, 1:2],
    x -> pool_enc2(conv_enc2(x)),
)
μ2 = Dense(flattened_size2, latent_dim2)
logσ2 = Dense(flattened_size2, latent_dim2)
interaction2 = Dense(latent_dim2, inter2^2)
transp1_dec2 = ConvTranspose((6, 6), 1 => 4, relu, stride = 4, pad = 4)
transp2_dec2 = ConvTranspose((30, 30), 4 => 1, sigmoid, stride = (2, 2), pad = 2)


####################### how to use model #######################################
if load_params
    # load model from old checkpoint
    (@load string(path_to_project, params_path, "parameters1.jld2") conv_enc1 conv2_enc1 μ1 logσ1 interaction1 transp1_dec1 transp2_dec1 dense_dec1)
    (@load string(path_to_project, params_path, "parameters2.jld2") conv_enc2 μ2 logσ2 interaction2 transp1_dec2 transp2_dec2)
    (@load string(path_to_project, params_path, "parameters0.jld2") interact0)
end

# compute output of encoders for data X
enc1(X) =
    apply_and_reshape(X, x -> conv2_enc1(pool_enc1(conv_enc1(x))), (flattened_size1, :))
enc2(X) =
    apply_and_reshape(X, x -> pool_enc2(conv_enc2(x)), (flattened_size2, :))

# sample z variable for mean μ and logvar logσ
z(μ, logσ) = μ + exp(logσ) * randn(Float32)

# compute output of interactions
int1(X) = apply_and_reshape(X, interaction1, (inter1, inter1, latent_dim1, :))
int2(X) = apply_and_reshape(X, interaction2, (inter2, inter2, 1, :))

# compute output of decoders
h(X) = apply_and_reshape(X, x -> transp3_dec1(transp2_dec1(transp1_dec1(x))), (60 * 60, :))
dec1(X) = apply_and_reshape(X, x -> h(x), (60, 60, 1, :))
dec2(X) = apply_and_reshape(X, x -> transp2_dec2(transp1_dec2(x)), (60, 60, 1, :))

# KL-divergence between approximation posterior and N(0, 1) prior.
kl_q_p(μ, logσ) = 0.5 * sum(exp.(2 .* logσ) + μ .^ 2 .- 1 .- (2 .* logσ))

# logp(x|z) - conditional probability of data given latents. bool correct?
logp_x_z1(x, z) = sum(my_logpdf.(Bernoulli.(dec1(int1(z))), x))
logp_x_z2(x, z) = sum(my_logpdf.(Bernoulli.(dec2(int2(z))), x))

# Monte Carlo estimator of mean ELBO using M samples.
function mc(X, enc, μ, logσ, logp_x_z)
    output_enc = enc(X)
    μ̂, logσ̂ = μ(output_enc), logσ(output_enc)
    return (logp_x_z(X, z.(μ̂, logσ̂)) - kl_q_p(μ̂, logσ̂)) * 1 // batch_size
end
mc1(X) = mc(X, enc1, μ1, logσ1, logp_x_z1)
mc2(X) = mc(X, enc2, μ2, logσ2, logp_x_z2)

# parameters of our vaes
ps1 = Flux.params(
    conv_enc1,
    conv2_enc1,
    μ1,
    logσ1,
    interaction1,
    transp1_dec1,
    transp2_dec1,
    transp3_dec1,
    #dense_dec1,
)
ps2 = Flux.params(
    conv_enc2,
    μ2,
    logσ2,
    interaction2,
    transp1_dec2,
    transp2_dec2,
)

# add regularization term
dec_size1 = size(Flux.params(interaction1, transp1_dec1, transp2_dec1, transp3_dec1).order)[1]
dec_size2 = size(Flux.params(interaction2, transp1_dec2, transp2_dec2).order)[1]
ps_size1 = size(ps1.order)[1]
ps_size2 = size(ps2.order)[1]
# SSQ of decoder parameters
regularization(ps, a, b) = sum(x -> sum(x .^ 2), ps.order[(a-b+1):a])
reg1() = regularization(ps1, ps_size1, dec_size1)
reg2() = regularization(ps2, ps_size2, dec_size2)


########################## train model #########################################
# loss is a combination of estimated ELBO and regularization
loss1(X) = -mc1(X) + 0.01f0 * reg1()  # (seems to be problematic with zygote)
loss2(X) = -mc2(X) + 0.01f0 * reg2()

# evaluation callback for supervision TODO add random set
evalcb1() = @show(-mc1(batches[1]))
evalcb2() = @show(-mc2(batches[1]))
optimizer1 = ADAM()
optimizer2 = ADAM()

# train cvae1 (small receptive field)
@epochs 50 Flux.train!(
    loss1,
    ps1,
    zip(batches),
    optimizer1,
    cb = Flux.throttle(evalcb1, 5),
)

# train final dense layer on decoder output
#dense_dec(X) = apply_and_reshape(X, x -> dense_dec1(h(x)), (60, 60, 1, :))
#ps_dense = Flux.params(dense_dec1)
#logp_x_z_dense(x, z) = sum(my_logpdf.(Bernoulli.(dense_dec(int1(z))), x))
#reg_dense() = regularization(ps_dense, 1, 1)
#mc_dense(X) = mc(X, enc1, μ1, logσ1, logp_x_z_dense)
#loss_dense(X) = -mc_dense(X) - 0.01 * reg_dense()
#optimizer_dense = ADAM()
#evalcb_dense() = @show(-mc_dense(batches[1]))
#@epochs 10 Flux.train!(
#    loss_dense,
#    ps_dense,
#    zip(batches),
#    optimizer_dense,
#    cb = Flux.throttle(evalcb_dense, 5)
#)

# train cvae2 (big receptive field)
@epochs 50 Flux.train!(
    loss2,
    ps2,
    zip(batches),
    optimizer2,
    cb = Flux.throttle(evalcb2, 20),
)

if store_params
    # store model parameters in specified folder
    (@save string(path_to_project, params_path, "parameters1_sep.jld2") conv_enc1 conv2_enc1 μ1 logσ1 interaction1 transp1_dec1 transp2_dec1 transp3_dec1)
    (@save string(path_to_project, params_path, "parameters2_sep.jld2") conv_enc2 μ2 logσ2 interaction2 transp1_dec2 transp2_dec2)
end

########################## plots, supervision, debugging #######################
if not_on_server
    # problematic on server
    using Plots
    using Images
    using ColorSchemes
end

if plot_input_path > "" & not_on_server
    sequence = [reshape(data[:, :, :, i], (60, 60, 1, 1)) for i = 1:size(
        data,
        4,
    )]
    anim1 = @animate for i = 1:2000
        Plots.heatmap(
            reshape(sequence[i], (60, 60)),
            seriescolor = cgrad(ColorSchemes.gray.colors),
        )
    end
    gif(anim1, string(path_to_project, plot_path, plot_input_path), fps = 15)
end

# compute outputs for both cvaes
function cvae1(X)
    encoded = enc1(X)
    z1 = z.(μ1(encoded), logσ1(encoded))
    return dec1(int1(z1))
end

function cvae2(X)
    encoded = enc2(X)
    z2 = z.(μ2(encoded), logσ2(encoded))
    return dec2(int2(z2))
end

if plot_output_path > "" & not_on_server
    sequence = [reshape(data[:, :, :, i], (60, 60, 1, 1)) for i = 1:size(
        data,
        4,
    )]
    anim1 = @animate for i = 1:2000
        output = cvae1(sequence[i])
        output = reshape(Flux.Tracker.data(output), 60, 60)
        Plots.heatmap(output, seriescolor = cgrad(ColorSchemes.gray.colors))
    end
    gif(anim1, string(path_to_project, plot_path, plot_output_path), fps = 15)
end


# TODO: train on server for longer
# TODO: make it attractive for cvae2 to have 0, regularization higher?
# TODO: 'nicer' reconstruction plotsdata


function make_reconstruction_plot(decoder, value, dim, title)
    # make a reconstruction plot by feeding vectors with one "hot" dimension
    # into decoder for +-value
    indices = get_indices(dim, value)

    plots = [Plots.heatmap(
                Flux.Tracker.data(decoder(i))[:, :],
                seriescolor = cgrad(ColorSchemes.gray.colors),
                title=string(i)) for i in indices]
    png(
        plot(plots..., layout = (dim, 2), size = (850, 300*(dim))),
        string(path_to_project, plot_path, title, ".png"),
    )
end

function get_indices(dim, value)
    # creates a list of indices to emulate z-states for
    # dim(z) = dim and one of its values set to +-value
    input = Int32.(zeros(dim))
    input[1] = 1
    indices = []
    for i in 1:dim
        push!(indices, input .* value)
        push!(indices, input .* (-value))
        input = circshift(input, 1)
    end
    return indices
end

make_reconstruction_plot(x -> dec1(int1(x)), 10, latent_dim1, "test4")

function plot_output_for(image, vae)
    output = vae(reshape(image, (60, 60, 1, 1)))
    plot = Plots.heatmap(Flux.Tracker.data(output)[:, :],
                seriescolor = cgrad(ColorSchemes.gray.colors),
                title="Output image")
    plot2 = Plots.heatmap(reshape(image, (60, 60)),
                seriescolor = cgrad(ColorSchemes.gray.colors),
                title="Input image")
    plots = [plot, plot2]
    Plots.plot(plots..., layout=(1,2), size = (850, 350))
end

plot_output_for(data[:,:,:,10], cvae1)


#### train vae1 and vae2  with activated interaction for same target

interact0 = Dense(latent_dim1 + latent_dim2, inter1^2 + inter2^2)
interact(X1, X2) = (X = [X1; X2]; interact0(X))
function interaction(X1, X2)
    # calculate interaction matrix
    i = interact(X1, X2)
    # divide interaction matrix into vae1's and vae2's part to feed into decoders
    i1 = reshape(i[1:inter1^2, :], (inter1, inter1, 1, :))
    i2 = reshape(i[inter1^2+1:inter1^2+inter2^2, :], (inter2, inter2, 1, :))
    return (i1, i2)
end

function cvae1and2(X)
    enc1_out = enc1(X)
    enc2_out = enc2(X)
    z1 = z.(μ1(enc1_out), logσ1(enc1_out))
    z2 = z.(μ2(enc2_out), logσ2(enc2_out))
    interacted = interact(z1, z2)
    interacted1 = reshape(interacted[1:inter1^2, :], (inter1, inter1, 1, :))
    interacted2 = reshape(interacted[inter1^2+1:inter1^2+inter2^2, :], (inter2, inter2, 1, :))
    return (dec1(interacted1), dec2(interacted2))
end

# train both vaes with interaction enabled
ps0 = Flux.params(
    conv_enc1,
    conv2_enc1,
    μ1,
    logσ1,
    interaction1,
    transp1_dec1,
    transp2_dec1,
    #dense_dec1,
    conv_enc2,
    μ2,
    logσ2,
    interaction2,
    transp1_dec2,
    transp2_dec2,
)
function logp_x_z0(x, z1, z2)
    i1, i2 = interaction(z1, z2)
    return sum(my_logpdf.(Bernoulli.(dec1(i1)), x)) + sum(my_logpdf.(Bernoulli.(dec2(i2)), x))
end
function mc0(X)
    out1, out2 = enc1(X), enc2(X)
    mu1, log1, mu2, log2 = μ1(out1), logσ1(out1), μ2(out2), logσ2(out2)
    klqp = kl_q_p(mu1, log1) + kl_q_p(mu2, log2)
    return (logp_x_z0(X, z.(mu1, log1), z.(mu2, log2)) - klqp) * 1 // batch_size
end
reg0() = reg1() + reg2()
loss0(X) = -mc0(X) + 0.01f0 * reg0()
optimizer0 = ADAM()
evalcb0() = @show(-mc0(batches[1]))
@epochs 10 Flux.train!(
    loss0,
    ps0,
    zip(batches),
    optimizer0,
    cb = Flux.throttle(evalcb0, 10),
)


#### train vae1 and vae2 for combined target
interact0 = Dense(latent_dim1 + latent_dim2, inter1^2*latent_dim1 + inter2^2)
interact(X1, X2) = (X = [X1; X2]; interact0(X))
function interaction(X1, X2)
    # calculate interaction matrix
    i = interact(X1, X2)
    # divide interaction matrix into vae1's and vae2's part to feed into decoders
    index1 = inter1^2*latent_dim1 # index of last interaction element of vae1
    i1 = reshape(i[1:index1, :], (inter1, inter1, latent_dim1, :))
    i2 = reshape(i[index1+1:index1+inter2^2, :], (inter2, inter2, 1, :))
    return (i1, i2)
end

# train both vaes with interaction enabled
ps0 = Flux.params(
    conv_enc1,
    conv2_enc1,
    μ1,
    logσ1,
    interaction1,
    transp1_dec1,
    transp2_dec1,
    #dense_dec1,
    conv_enc2,
    μ2,
    logσ2,
    interaction2,
    transp1_dec2,
    transp2_dec2,
)
function logp_x_z0(x, z1, z2)
    i1, i2 = interaction(z1, z2)
    decoded = min.(1, dec1(i1) .+ dec2(i2))
    return sum(my_logpdf.(Bernoulli.(decoded), x))
end
function mc0(X)
    out1, out2 = enc1(X), enc2(X)
    mu1, log1, mu2, log2 = μ1(out1), logσ1(out1), μ2(out2), logσ2(out2)
    klqp = kl_q_p(mu1, log1) + kl_q_p(mu2, log2)
    return (logp_x_z0(X, z.(mu1, log1), z.(mu2, log2)) - klqp) * 1 // batch_size
end
reg0() = reg1() + reg2()
loss0(X) = -mc0(X) + 0.01f0 * reg0()
optimizer0 = ADAM()
evalcb0() = @show(-mc0(batches[1]))
@epochs 3 Flux.train!(
    loss0,
    ps0,
    zip(batches),
    optimizer0,
    cb = Flux.throttle(evalcb0, 10),
)

function cvae1and2(X)
    enc1_out = enc1(X)
    enc2_out = enc2(X)
    z1 = z.(μ1(enc1_out), logσ1(enc1_out))
    z2 = z.(μ2(enc2_out), logσ2(enc2_out))
    i1, i2 = interaction(z1, z2)
    return min.(1, dec1(i1) .+ dec2(i2))
end

plot_output_for(data[:,:,:,7], cvae1and2)
