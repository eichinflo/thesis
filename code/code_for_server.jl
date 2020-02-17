# Convolutional Variational Auto Encoder model for dummydata

# load libraries
using Flux  # for model and learning
using Flux: @epochs, mse
using Base.Iterators: partition  # for creating batches
using Flux: Conv, MaxPool, Dense, ConvTranspose
using JLD2  # store model parameters for later
using Distributions  # to compute loss function
using MAT  # for importing dummydata
using Random

################### hyperparameters  and options ###############################
Random.seed!(42)  # set random seed for reproducability

# data
binary = true  # binarize data?
batch_size = 5
scaled_vae2 = true  # use downscaled target for vae2?
scaling = 30  # how many pixels per scaled pixel?

# training
train_seperately = true  # train vae1 and vae2 seperately before combination
epochs_dual = 1  # epochs of combined training
epochs_vae1 = 1  # epochs of seperate training for vae1
epochs_vae2 = 1  # epochs of seperate training for vae1
callbacks = 50

# model
latent_dim1 = 4  # number of z-variaables for vae1
latent_dim2 = 1  # number of z-variables for vae2
inter1 = 8  # size of image after interaction
inter2 = 2
store_params = true  # store models after training?
load_params = false  # load models from checkpoint before training?

# plotting
plot_recon = false  # plot data reconstruction?
not_on_server = false  # we don't want to plot on server, leads to problems

# paths TODO: update
dual_params1 = ""
dual_params2 = ""
plot_input_path = "input.gif"
plot_output_path = "output.gif"
# change accordingly on your machine
path_to_project = "/home/flo/projects/thesis/code/"
params_path = "params/"
plot_path = "plots/"
################################################################################


######################## helper functions ######################################
"Numerically stable logpdf for 'p' close to 1 or 0."
my_logpdf(b::Bernoulli, y::Any) =
    y * log(b.p + eps(Float32)) + (1f0 - y) * log(1 - b.p + eps(Float32))

"Compute the product of the first three size-dimensions of f(sampleofdata).
This function needs data to be already loaded globally."
function no_of_entries(f)
    sample = data[:, :, :, 1:2]
    sample = f(sample)
    no_of_entries = size(sample)[1] * size(sample)[2] * size(sample)[3]
end

"Load dummy data from path_path_to_dummy_data and reshape to WHCN order.
Binarize if binary=true, else just normalize entries to [0, 1]. Be aware that
this can lead to numerically instable behaviour if you have outliers in data."
function get_dummy_data(path_to_dummy_data; binary=true, image_size=60)
    file = matopen(path_to_dummy_data)
    data = read(file, "pix3D")
    close(file)
    # store data in WHCN order (width, height, channel, batches)
    data = reshape(data, (image_size, image_size, 1, size(data, 3)))
    data = convert.(Float64, data)
    # normalize entries to [0, 1]
    data = (data .- min(data...)) ./ (max(data...) - min(data...))
    if binary
        # binarize data
        data = data .> 0.5
    end
    return data
end

"Create array of array-batches of size batch_size from data."
function get_batches(data; batch_size=5, image_size=60)
    batches = [reshape(
        data[:, :, :, ((i-1)*batch_size+1):(i*batch_size)],
        (image_size, image_size, 1, batch_size),
    ) for i = 1:size(data, 4)÷batch_size]
end

"Get a scaled down version of data with size scaled_size.
scaled_size must divide size of data images."
function get_scaled(data; full_size=60, scaled_size=scaling)
    data_scaled = scaled(data[:,:,:,1], full_size, scaled_size)
    for i in range(2, stop=size(data)[4])
        image = data[:, :, :, i]
        data_scaled = cat(data_scaled, scaled(image, full_size, scaled_size), dims=3)
    end
    return reshape(data_scaled, (Int(full_size/scaled_size),
                                 Int(full_size/scaled_size), 1, :))
end

scal_func(x) = sigmoid(12*(x/scaling^2-0.5)) # this has yielded good results

function scaled(im, fs, ss)
    scaled = []
    n = Int(fs/ss)
    for i in range(1, stop=n)
        for j in range(1, stop=n)
            append!(scaled,
                    scal_func(sum(im[1+(i-1)*ss:i*ss, 1+(j-1)*ss:j*ss])))
        end
    end
    return reshape(scaled, (n, n))
end

"Returns String-filename with hint at state of global variables like binary,
scaled_vae2 and epochs_dual."
function get_filename(fname, suffix)
    sc, bin = string(scaled_vae2*1), string(binary*1)  # convert to Int64 and then to String
    e1, e2, ed = string(epochs_vae1), string(epochs_vae2), string(epochs_dual)
    if !train_seperately
        ep1, ep2 = 0, 0
    end
    return string(fname, "_sc=", sc, "_bin=", bin, "_e1=", e1, "_e2=", e2, "_ed=", ed, suffix)

end

########################### load data###########################################
data = get_dummy_data(
    string(path_to_project, "../dummydata/dummyData2000.mat"),
    binary = binary,
)

# not nice, but does the job
data_scaled = get_scaled(data)
batches_scaled = [(reshape(data[:,:,:,i:i+batch_size-1], (60, 60, 1, batch_size)),
                    reshape(data_scaled[:,:,:,i:i+batch_size-1] , (2, 2, 1, batch_size)))
                   for i in range(1, step=batch_size, length=Int(size(data)[4]/batch_size))]  # TODO replace with zip

# create an array of array-batches of size batch_size
batches = get_batches(data, batch_size = 5)


########################### models #############################################
# there is two VAEs, vae1 with a small receptive field for modelling foreground
# signals and vae2 with bigger receptive field to model background signal
# the naming scheme of the respective comonents works accordingly

## vae1 with small receptive field of size (5, 5)
conv_enc1 = Conv((6, 6), 1 => 32, relu, pad = 2, stride = (3, 3))  # e.g. convolution of encoder of vae1
pool_enc1 = MaxPool((3, 3), stride = (2, 2), pad = 0)
conv2_enc1 = Conv((2, 2), 32 => latent_dim1, stride = (1, 1), pad = 0)
# compute data size after pooling in order to determine size after flattening
flattened_size1 = no_of_entries(x -> conv2_enc1(pool_enc1(conv_enc1(x))))
# mean and log-variance of vae1's z-variable
μ1 = Dense(flattened_size1, latent_dim1)
logσ1 = Dense(flattened_size1, latent_dim1)
# later on, we want to let z-variables interact via this layer
interaction1 = Dense(latent_dim1, inter1^2 * latent_dim1)
# 'deconvolutions' of vae1's decoder
transp1_dec1 = ConvTranspose(
    (2, 2),
    latent_dim1 => 32,
    relu,
    stride = (1, 1),
    pad = 0,
)
transp2_dec1 = ConvTranspose((3, 3), 32 => 32, relu, stride = (2, 2), pad = 0)
transp3_dec1 = ConvTranspose((6, 6), 32 => 1, sigmoid, stride = (3, 3), pad = 0)
# final decoder layer, we might want to drop this actually TODO
dense_dec1 = Dense(60 * 60, 60 * 60, sigmoid)

## vae2 with big receptive field of size ??, analogous to vae1
conv_enc2 = Conv((30, 30), 1 => 2, relu, pad = 2, stride = (2, 2))
pool_enc2 = MaxPool((6, 6), stride = (6, 6))
flattened_size2 = no_of_entries(x -> pool_enc2(conv_enc2(x)))
μ2 = Dense(flattened_size2, latent_dim2)
logσ2 = Dense(flattened_size2, latent_dim2)
interaction2 = Dense(latent_dim2, inter2^2)
transp1_dec2 = ConvTranspose((6, 6), 1 => 4, relu, stride = 4, pad = 4)
transp2_dec2 = ConvTranspose(
    (30, 30),
    4 => 1,
    sigmoid,
    stride = (2, 2),
    pad = 2,
)

if scaled_vae2
    # we can just use a simple dense decoder for a scaled target
    scaled_size = Int(60/scaling)
    scaled_dec = Dense(inter2^2, scaled_size^2, sigmoid)
end


####################### how to use model #######################################
if load_params
    # load model from pretrained checkpoint
    # note: changes to the code above have no effect, disable this piece if necessary
    (@load string(path_to_project, params_path, "parameters1_sep.jld2") conv_enc1 conv2_enc1 μ1 logσ1 interaction1 transp1_dec1 transp2_dec1 transp3_dec1)
    (@load string(path_to_project, params_path, "parameters2_sep.jld2") conv_enc2 μ2 logσ2 interaction2 transp1_dec2 transp2_dec2)
end

"Compute result of applying f on X and reshaping to given reshaped_size."
apply_and_reshape(X, f, reshaped_size) = reshape(f(X), reshaped_size)

"Compute encoder output of vae1 for data X."
enc1(X) = apply_and_reshape(
    X,
    x -> conv2_enc1(pool_enc1(conv_enc1(x))),
    (flattened_size1, :),
)

"Compute encoder output of vae2 for data X."
enc2(X) =
    apply_and_reshape(X, x -> pool_enc2(conv_enc2(x)), (flattened_size2, :))

"Sample z variable for mean μ and log(variance) logσ."
z(μ, logσ) = μ + exp(logσ) * randn(Float32)

"Compute output of seperate interaction of vae1."
int1(X) = apply_and_reshape(X, interaction1, (inter1, inter1, latent_dim1, :))

"Compute output of seperate interaction of vae2."
int2(X) = apply_and_reshape(X, interaction2, (inter2, inter2, 1, :))

"Compute decoder output of vae1."
dec1(X) = apply_and_reshape(X,
                            x -> transp3_dec1(transp2_dec1(transp1_dec1(x))),
                            (60, 60, 1, :))

"Compute decoder output of vae2."
dec2(X) =
    apply_and_reshape(X, x -> transp2_dec2(transp1_dec2(x)), (60, 60, 1, :))
if scaled_vae2
    dec2(X) = apply_and_reshape(X, x -> scaled_dec(reshape(x, (inter2*inter2, :))),
                                (scaled_size, scaled_size, 1, :))
end

"Compute output of vae1 for data X in WHCN order."
function cvae1(X)
    encoded = enc1(X)
    z1 = z.(μ1(encoded), logσ1(encoded))
    return dec1(int1(z1))
end

"Compute output of vae2 for data X in WHCN order."
function cvae2(X)
    encoded = enc2(X)
    z2 = z.(μ2(encoded), logσ2(encoded))
    return dec2(int2(z2))
end

"KL-divergence between approximation posterior and N(0, 1) prior."
kl_q_p(μ, logσ) = 0.5 * sum(exp.(2 .* logσ) + μ .^ 2 .- 1 .- (2 .* logσ))

# logp(x|z) - conditional probability of data given latents.
log_p_x_z(z, X, dec) = sum(my_logpdf.(Bernoulli.(dec(z)), X))
logp_x_z1(X, z) = log_p_x_z(z, X, x -> dec1(int1(x)))
logp_x_z2(X, z) = log_p_x_z(z, X, x -> dec2(int2(x)))

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
dec_size1 = size(Flux.params(
    transp1_dec1,
    transp2_dec1,
    transp3_dec1,
).order)[1]
dec_size2 = size(Flux.params(transp1_dec2, transp2_dec2).order)[1]
if scaled_vae2
    ps2 = Flux.params(conv_enc2, μ2, logσ2, interaction2, scaled_dec)
    dec_size2 = size(Flux.params(scaled_dec).order)[1]
end
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

if scaled_vae2
    function L(X, y)
        output_enc = enc2(X)
        μ̂, logσ̂ = μ2(output_enc), logσ2(output_enc)
        # TODO find appropriate scaling
        return (logp_x_z2(y, z.(μ̂, logσ̂))*100 - kl_q_p(μ̂, logσ̂)) * 1 // batch_size
    end
    mc2(x) = (L(x...))
end

# evaluation callback for supervision TODO add random set
evalcb1() = @show(-mc1(batches[2]))
evalcb2() = @show(-mc2(batches[2]))
optimizer1 = ADAM()
optimizer2 = ADAM()


if train_seperately
# train cvae1 (small receptive field)
    @epochs epochs_vae1 Flux.train!(
        loss1,
        ps1,
        zip(batches),
        optimizer1,
        cb = Flux.throttle(evalcb1, callbacks),
    )

    if scaled_vae2
        evalcb_scal() = @show(-mc2(batches_scaled[2]))
        @epochs epochs_vae2 Flux.train!(
            loss2,
            ps2,
            zip(batches_scaled),
            optimizer2,
            cb = Flux.throttle(evalcb_scal, callbacks),
            )
    else
        # train cvae2 (big receptive field)
        @epochs epochs_vae2 Flux.train!(
            loss2,
            ps2,
            zip(batches),
            optimizer2,
            cb = Flux.throttle(evalcb2, callbacks),
            )
    end
end

# store model in specified folder
if store_params
    (@save string(path_to_project, params_path, get_filename("params1_sep", ".jld2")) conv_enc1 conv2_enc1 μ1 logσ1 interaction1 transp1_dec1 transp2_dec1 transp3_dec1)
    (@save string(path_to_project, params_path, get_filename("params2_sep", ".jld2")) conv_enc2 μ2 logσ2 interaction2 transp1_dec2 transp2_dec2)
end

if not_on_server
    # create the plots found in the thesis
    include("/home/flo/projects/thesis/code/plotting.jl")
    #TODO solve path situation!!!
    make_reconstruction_plot(x -> dec1(int1(x)), 4, latent_dim1, "recon_vae1_sep")
    make_reconstruction_plot(x -> dec2(int2(x)), 4, latent_dim1, "recon_vae2_sep")
    plot_output_for(data[:, :, :, 1], cvae1, "im1_vae1")
    plot_output_for(data[:, :, :, 7], cvae1, "im2_vae1")
    plot_output_for(data[:, :, :, 1], cvae2, "im1_vae2")
    plot_output_for(data[:, :, :, 7], cvae2, "im2_vae2")
end

#### train vae1 and vae2  with activated interaction for same target
if train_seperately
    # get initial interaction weights W and bias b from pretrained layers
    W_i1, W_i2 = Flux.data(interaction1.W), Flux.data(interaction2.W)
    W_i0 = [
        W_i1 zeros(size(W_i1)[1], size(W_i2)[2])
        zeros(size(W_i2)[1], size(W_i1)[2]) W_i2
    ]
    b_i1, b_i2 = Flux.data(interaction1.b), Flux.data(interaction2.b)
    b_i0 = [b_i1; b_i2]
    interact0 = Dense(
        latent_dim1 + latent_dim2,
        inter1^2 + inter2^2,
        initW = (x, y) -> W_i0,
        initb = (x) -> b_i0,
    )
else
    # we initialize random W and b
    interact0 = Dense(
        latent_dim1 + latent_dim2,
        inter1^2*latent_dim1 + inter2^2
    )
end
# how to apply interaction
interact(X1, X2) = (X = [X1; X2]; interact0(X))

"Apply interaction on z-outputs X1 and X2 seperate outputs into a tuple."
function interaction(X1, X2)
    # calculate interaction matrix
    i = interact(X1, X2)
    # divide interaction matrix into vae1's and vae2's part to feed into decoders
    index1 = inter1^2 * latent_dim1 # index of last interaction element of vae1
    i1 = reshape(i[1:index1, :], (inter1, inter1, latent_dim1, :))
    i2 = reshape(
        i[index1+1:index1+inter2^2, :],
        (inter2, inter2, latent_dim2, :),
    )
    return (i1, i2)
end

"Compute output of dual vae. switch_off sets respective z variable to 0.
Returns a tuple with outputs for vae1 and vae2 respectively."
function dual_vae_out(X; switch_off_z1 = false, switch_off_z2 = false)
    #calculate respective outputs for vae1 and vae2 with interaction enabled
    enc1_out = enc1(X)
    enc2_out = enc2(X)
    z1 = z.(μ1(enc1_out), logσ1(enc1_out))
    z2 = z.(μ2(enc2_out), logσ2(enc2_out))
    if switch_off_z1
        z1 = convert.(Float32, zeros(size(z1)))
    end
    if switch_off_z2
        z2 = convert.(Float32, zeros(size(z2)))
    end
    interacted1, interacted2 = interaction(z1, z2)
    return (dec1(interacted1), dec2(interacted2))
end

"Compute output of vae1 in dual vae."
vae1_out(X) = dual_vae_out(X)[1]

"Compute output of vae2 in dual vae."
vae2_out(X) = dual_vae_out(X)[2]

# train both vaes with interaction enabled
ps = Flux.params(
    conv_enc1,
    conv2_enc1,
    μ1,
    logσ1,
    interaction1,
    transp1_dec1,
    transp2_dec1,
    transp3_dec1,
    conv_enc2,
    μ2,
    logσ2,
    interact0,
    transp1_dec2,
    transp2_dec2,
)

# MC estimator of ELBO
function mc_dual(X)
    out1, out2 = enc1(X), enc2(X)
    mu1, log1, mu2, log2 = μ1(out1), logσ1(out1), μ2(out2), logσ2(out2)
    z1, z2 = z.(mu1, log1), z.(mu2, log2)
    i1, i2 = interaction(z1, z2)
    logp_x_z1 = log_p_x_z(i1, X, dec1)
    logp_x_z2 = log_p_x_z(i2, X, dec2)
    klqp = kl_q_p(mu1, log1) + kl_q_p(mu2, log2)
    return (logp_x_z1 + logp_x_z2 - klqp) * 1 // batch_size
end
if scaled_vae2
    function mc_scal(X, y)
        out1, out2 = enc1(X), enc2(X)
        mu1, log1, mu2, log2 = μ1(out1), logσ1(out1), μ2(out2), logσ2(out2)
        z1, z2 = z.(mu1, log1), z.(mu2, log2)
        i1, i2 = interaction(z1, z2)
        logp_x_z1 = log_p_x_z(i1, X, dec1)
        #TODO find appropriate scaling! Same above in mc2...
        logp_x_z2 = log_p_x_z(i2, y, dec2) * 100
        klqp = kl_q_p(mu1, log1) + kl_q_p(mu2, log2)
        return (logp_x_z1 + logp_x_z2 - klqp) * 1 // batch_size
    end
    mc_dual(x) = mc_scal(x...)
end

reg() = reg1() + reg2()
loss(X) = -mc_dual(X) + 0.01f0 * reg()
opt = ADAM()
evalcb() = @show(-mc_dual(batches[2]))
if scaled_vae2
    evalcb() = @show(-mc_dual(batches_scaled[2]))
end

if load_params
    #TODO: change with filename function
    (@load string(path_to_project, params_path, dual_params) conv_enc1 conv2_enc1 μ1 logσ1 interaction1 transp1_dec1 transp2_dec1 transp3_dec1 interact0)
    (@load string(path_to_project, params_path, dual_params) conv_enc2 μ2 logσ2 interaction2 transp1_dec2 transp2_dec2)
end
if !scaled_vae2
    @epochs epochs_dual Flux.train!(
        loss,
        ps,
        zip(batches),
        opt,
        cb = Flux.throttle(evalcb, callbacks),
    )
else
    @epochs epochs_dual Flux.train!(
        loss,
        ps,
        zip(batches_scaled),
        opt,
        cb = Flux.throttle(evalcb, callbacks),
    )
end

if store_params
    (@save string(path_to_project, params_path, get_filename("params1_int", ".jld2")) conv_enc1 conv2_enc1 μ1 logσ1 interaction1 transp1_dec1 transp2_dec1 transp3_dec1 interact0)
    (@save string(path_to_project, params_path, get_filename("params2_int", ".jld2")) conv_enc2 μ2 logσ2 interaction2 transp1_dec2 transp2_dec2)
end

if not_on_server
    plot_output_for(data[:, :, :, 1], cvae1_out, "im1_vae1_int")
    plot_output_for(data[:, :, :, 7], cvae1_out, "im2_vae1_int")
    plot_output_for(data[:, :, :, 1], cvae2_out, "im1_vae2_int")
    plot_output_for(data[:, :, :, 7], cvae2_out, "im2_vae2_int")
end

function cvae1_out_z(z)
    #calculate respective outputs for vae1 and vae2 with interaction enabled
    z1 = z[1:latent_dim1]
    z2 = z[latent_dim1+1:latent_dim1+latent_dim2]
    interacted1, interacted2 = interaction(z1, z2)
    return dec1(interacted1)
end

if not_on_server
    make_reconstruction_plot(
        x -> cvae1_out_z(x),
        10,
        latent_dim1 + latent_dim2,
        "recon_vae1_int",
    )
end

# make reconstruction plots with z2 turned off (i.e. background noise gone)

######################### some experimental stuff ##############################
#### train vae1 and vae2 for combined target
# functions for combining decoded images
"""
comb(x, y) = max(x, y)
comb(x, y) = (x^10 + y^10)^(1/10)

function logp_x_z0(x, z1, z2)
    i1, i2 = interaction(z1, z2)
    decoded = comb.(dec1(i1), dec2(i2))
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
@epochs 10 Flux.train!(
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
"""

"""
if not_on_server
    cvae1_only(X) = cvae1and2_out(X, switch_off_z2 = true)[1]
    # make reconstruction gif
    sequence = [reshape(data[:, :, :, i], (60, 60, 1, 1)) for i = 1:size(
        data,
        4,
    )][1:100]
    gif1 = @gif for i = 1:100
                    out3 = cvae1_only(sequence[i])
                    out3 = reshape(Flux.Tracker.data(out3), 60, 60)
                    plt3 = Plots.heatmap(
                        out3,
                        seriescolor = cgrad(ColorSchemes.gray.colors),
                        title = "output",
                        )
                    out2 = cvae1_out(sequence[i])
                    out2 = reshape(Flux.Tracker.data(out2), 60, 60)
                    plt2 = Plots.heatmap(
                        out2,
                        seriescolor = cgrad(ColorSchemes.gray.colors),
                        title = "output",
                        )
                    plt1 = Plots.heatmap(
                        reshape(sequence[i], (60, 60)),
                        seriescolor = cgrad(ColorSchemes.gray.colors),
                        title = "input",
                        )
                    plots = [plt1, plt2, plt3]
                    Plots.plot(plots..., layout = (1, 3), size = (1400, 350))
                end
    print("hello2")
    gif(gif1, "reconstructed.gif")
end
"""
