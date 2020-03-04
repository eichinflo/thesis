# model defintion and train methods
# there is two VAEs, vae1 with a small receptive field for modelling foreground
# signals and vae2 with bigger receptive field to model background signal
# the naming scheme of the respective comonents works accordingly

## vae1 with small receptive field of size (5, 5)
conv_enc1 = Conv((6, 6), 1 => 16, relu, pad = 2, stride = (3, 3))  # e.g. convolution of encoder of vae1
pool_enc1 = MaxPool((3, 3), stride = (2, 2), pad = 0)
conv2_enc1 = Conv((2, 2), 16 => latent_dim1, stride = (1, 1), pad = 0)
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
    latent_dim1 => 16,
    relu,
    stride = (1, 1),
    pad = 0,
)
transp2_dec1 = ConvTranspose((3, 3), 16 => 16, relu, stride = (2, 2), pad = 0)
transp3_dec1 = ConvTranspose((6, 6), 16 => 1, sigmoid, stride = (3, 3), pad = 0)
# final decoder layer, we might want to drop this actually TODO
dense_dec1 = Dense(60 * 60, 60 * 60, sigmoid)
# we can just use a simple dense decoder for a scaled target
scaled_size = Int(60/scaling)
scaled_dec = Dense(inter2^2, scaled_size^2, sigmoid)

## vae2 with big receptive field of size ??, analogous to vae1
conv_enc2 = Conv((30, 30), 1 => 2, relu, pad = 2, stride = (2, 2))
pool_enc2 = MaxPool((6, 6), stride = (6, 6))
flattened_size2 = no_of_entries(x -> pool_enc2(conv_enc2(x)))
μ2 = Dense(flattened_size2, latent_dim2)
logσ2 = Dense(flattened_size2, latent_dim2)
interaction2 = Dense(latent_dim2, inter2^2)
transp1_dec2 = ConvTranspose((10, 10), 1 => 4, relu, stride = 8, pad = 0)
transp2_dec2 = ConvTranspose(
    (30, 30),
    4 => 1,
    sigmoid,
    stride = (2, 2),
    pad = 2,
)

# layer for training with enabled interaction between z variables of vae1
# and vae2
interact0 = Dense(
    latent_dim1 + latent_dim2,
    inter1^2*latent_dim1 + inter2^2
)

####################### how to use model #######################################
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

z(μ, logσ, eps) = μ + exp(logσ) * eps


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
dec2_scal(X) = apply_and_reshape(X, x -> scaled_dec(reshape(x, (inter2*inter2, :))),
                                (scaled_size, scaled_size, 1, :))

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

"Compute output of vae2 for data X in WHCN order."
function cvae2_scal(X)
    encoded = enc2(X)
    z2 = z.(μ2(encoded), logσ2(encoded))
    return dec2_scal(int2(z2))
end

"KL-divergence between approximation posterior and N(0, 1) prior."
kl_q_p(μ, logσ) = 0.5 * sum(exp.(2 .* logσ) + μ .^ 2 .- 1 .- (2 .* logσ))


if binary
    "logp(x|z) - conditional probability of data given latents."
    log_p_x_z(z, X, dec) = sum(my_logpdf.(Bernoulli.(dec(z)), X))
else
    # so far we assume \sigma^2 = 1 as we're only interested in images
    "logp(x|z) - conditional probability of data given latents."
    log_p_x_z(z, X, dec) = -0.5 * sum(log(2 * pi) .+ (dec(z) .- X).^2)
end

log_p_z1(z) = sum(logpdf_norm.(z))
log_p_z2(z) = sum(logpdf_norm.(z))

eps1_distr = MvNormal(zeros(latent_dim1), zeros(latent_dim1) .+ 1)
eps2_distr = MvNormal(zeros(latent_dim2), zeros(latent_dim2) .+ 1)

log_q_z_x(ϵ, log_sigma) = sum(logpdf_norm.(ϵ) - log_sigma)

logpdf_norm(x) = -log(sqrt(2*pi)) - 0.5 * x^2

"Abstract loss function."
function L(X, y, enc, μ, logσ, logp_x_z, logq_z_x, eps_distr, logp_z)
    output_enc = enc(X)
    μ̂, logσ̂ = μ(output_enc), logσ(output_enc)
    ϵ = rand(eps_distr, size(logσ̂)[2])
    z_ = z.(μ̂, logσ̂, ϵ)
    return (logp_x_z(y, z_) + logp_z(z_)  - log_q_z_x(ϵ, logσ̂ )) * 1 // batch_size
end

"Loss for vae1 i.e. monte carlo estimator of ELBO1."
L1(X) = L(X, X, enc1, μ1, logσ1, logp_x_z1, log_q_z_x, eps1_distr, log_p_z1)

"Loss for vae2 without scaling i.e. monte carlo estimator of ELBO2."
L2(X) = L(X, X, enc2, μ2, logσ2, logp_x_z2, log_q_z_x, eps2_distr, log_p_z2)

f(X, y) = L(X, y, enc2, μ2, logσ2, logp_x_z2_scal, log_q_z_x, eps2_distr, log_p_z2)

"Loss for vae2 with scaling i.e. monte carlo estimator of ELBO2."
L2_scal(x) = f(x...)

"logp(x|z) - conditional probability of data given latents."
logp_x_z1(X, z) = log_p_x_z(z, X, x -> dec1(int1(x)))

"logp(x|z) - conditional probability of data given latents."
logp_x_z2(X, z) = log_p_x_z(z, X, x -> dec2(int2(x)))

"logp(x|z) - conditional probability of data given latents."
logp_x_z2_scal(X, z) = log_p_x_z(z, X, x -> dec2_scal(int2(x)))

"Monte Carlo estimator of mean ELBO using samples X."
#function L(X, y, enc, μ, logσ, logp_x_z)
#    output_enc = enc(X)
#    μ̂, logσ̂ = μ(output_enc), logσ(output_enc)
#    return (logp_x_z(y, z.(μ̂, logσ̂)) - kl_q_p(μ̂, logσ̂)) * 1 // batch_size
#end
"Monte Carlo estimator of mean ELBO using samples X."
# L1(X) = L(X, X, enc1, μ1, logσ1, logp_x_z1)

"Monte Carlo estimator of mean ELBO using samples X."
# L2(X) = L(X, X, enc2, μ2, logσ2, logp_x_z2)

# f(X, y) = L(X, y, enc2, μ2, logσ2, logp_x_z2_scal)
"Monte Carlo estimator of mean ELBO using samples X and target y for vae2."
# L2_scal(x) = f(x...)

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
ps2_scal = Flux.params(conv_enc2, μ2, logσ2, interaction2, scaled_dec)

# add regularization term
dec_size1 = size(Flux.params(
    transp1_dec1,
    transp2_dec1,
    transp3_dec1,
).order)[1]
dec_size2 = size(Flux.params(transp1_dec2, transp2_dec2).order)[1]
dec_size2_scal = size(Flux.params(scaled_dec).order)[1]
ps_size1 = size(ps1.order)[1]
ps_size2 = size(ps2.order)[1]
ps_size2_scal = size(ps2_scal.order)[1]

# SSQ of decoder parameters
regularization(ps, a, b) = sum(x -> sum(x .^ 2), ps.order[(a-b+1):a])
reg1() = regularization(ps1, ps_size1, dec_size1)
reg2() = regularization(ps2, ps_size2, dec_size2)
reg2_scal() = regularization(ps2_scal, ps_size2_scal, dec_size2_scal)

# loss is a combination of estimated ELBO and regularization
loss1(X) = -L1(X) + 0.01f0 * reg1()  # (seems to be problematic with zygote)
loss2(X) = -L2(X) + 0.01f0 * reg2()
loss2_scal(X) = -L2_scal(X) + 0.01f0 * reg2_scal()

# evaluation callback for supervision TODO: add checkpointing
evalcb1() = @show(-L1(batches[2]))
evalcb2() = @show(-L2(batches[2]))
evalcb2_scal() = @show(-L2_scal(batches_scaled[2]))
optimizer1 = ADAM()
optimizer2 = ADAM()
optimizer2_scal = ADAM()

"Just an abstraction of @epochs and Flux.train! for the vae1."
function train_vae1(batches, epochs)
    @epochs epochs Flux.train!(
        loss1,
        ps1,
        zip(batches),
        optimizer1,
        cb = Flux.throttle(evalcb1, callbacks),
    )
end

"Just an abstraction of @epochs and Flux.train! for the vae2 with unscaled target."
function train_vae2(batches, epochs)
    # train cvae2 (big receptive field)
    @epochs epochs_vae2 Flux.train!(
        loss2,
        ps2,
        zip(batches),
        optimizer2,
        cb = Flux.throttle(evalcb2, callbacks),
    )
end

"Just an abstraction of @epochs and Flux.train! for the vae2 with scaled target."
function train_vae2_scal(batches, epochs)
    # train cvae2 (big receptive field)
    print("Training scaled vae2...")
    @epochs epochs Flux.train!(
        loss2_scal,
        ps2_scal,
        zip(batches),
        optimizer2_scal,
        cb = Flux.throttle(evalcb2_scal, callbacks),
    )
    print("done. \n")
end

# some functions to realize interaction
"Use weights of interaction1 and interaction2 layers to
combine both and builds new, bigger layer with according weights."
function get_interaction(interaction1, interaction2)
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
    return interact0
end

"How to apply interaction between z1 and z2 z variables."
interact(z1, z2) = (z = [z1; z2]; interact0(z))

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

interaction_v(X) = interaction(X[1:latent_dim1], X[latent_dim1+1:latent_dim1+latent_dim2])

"Compute output of dual vae. switch_off sets respective z variable to 0.
Returns a tuple with outputs for vae1 and vae2 respectively."
function dual_vae_out(X; switch_off_z1 = false, switch_off_z2 = false, scal=true)
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
    if scal
        return (dec1(interacted1), dec2_scal(interacted2))
    else
        return (dec1(interacted1), dec2(interacted2))
    end
end


"Compute output of vae1 in dual vae."
vae1_out_scal(X) = dual_vae_out(X)[1]

"Compute output of vae2 in dual vae."
vae2_out_scal(X) = dual_vae_out(X)[2]

"Compute output of vae1 in dual vae."
vae1_out_scal_o(X) = dual_vae_out(X, switch_off_z2=true)[1]

"Compute output of vae2 in dual vae."
vae2_out_scal_o(X) = dual_vae_out(X, switch_off_z2=true)[2]

# train both vaes with interaction enabled
ps_dual = Flux.params(
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
ps_dual_scal = Flux.params(
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
    scaled_dec
)

"MC estimator of ELBO1 and ELBO2 for dual vae."
function _L_dual(X, y; dec2=dec2_scal, scal_logp=1, scal_q=1)
    out1, out2 = enc1(X), enc2(X)
    mu1, log1, mu2, log2 = μ1(out1), logσ1(out1), μ2(out2), logσ2(out2)
    eps1, eps2 = rand(eps1_distr, size(log1)[2]), rand(eps2_distr, size(log2)[2])
    z1, z2 = z.(mu1, log1, eps1), z.(mu2, log2, eps2)
    i1, i2 = interaction(z1, z2)
    logp_x_z1 = log_p_x_z(i1, X, dec1) + log_p_z1(i1)
    logp_x_z2 = (log_p_x_z(i2, y, dec2) + log_p_z2(i2)) * scal_logp
    logqzx = log_q_z_x(eps1, log1) + log_q_z_x(eps2, log2) * scal_q
    return (logp_x_z1 + logp_x_z2 - logqzx) * 1 // batch_size
end

"MC estimator of ELBO1 and ELBO2 for dual vae."
function L(X, y, enc, μ, logσ, logp_x_z, logq_z_x, eps_distr, logp_z)
    output_enc = enc(X)
    μ̂, logσ̂ = μ(output_enc), logσ(output_enc)
    ϵ = rand(eps_distr, size(logσ̂)[2])
    z_ = z.(μ̂, logσ̂, ϵ)
    return (logp_x_z(y, z_) + logp_z(z_)  - log_q_z_x(ϵ, logσ̂ )) * 1 // batch_size
end

"MC estimator of ELBO1 and ELBO2 for dual vae."
L_dual(X) = _L_dual(X, X, dec2=dec2)

"MC estimator of ELBO1 and ELBO2 for dual vae."
L_dual_scal(x) = _L_dual(x...,scal_logp=450)

reg_dual() = reg1() + reg2()
reg_dual_scal() = reg1() + reg2_scal()
loss_dual(X) = -L_dual(X) + 0.01f0 * reg_dual()
loss_dual_scal(X) = -L_dual_scal(X) + 0.01f0 * reg_dual_scal()
opt = ADAM()
opt_scal = ADAM()
evalcb() = @show(-L_dual(batches[2]))
evalcb_scal() = @show(-L_dual_scal(batches_scaled[2]))

function train_dual_scal(batches, epochs)
    @epochs epochs_dual Flux.train!(
        loss_dual_scal,
        ps_dual_scal,
        zip(batches),
        opt_scal,
        cb = Flux.throttle(evalcb_scal, callbacks),
    )
end

function train_dual(batches, epochs)
    @epochs epochs_dual Flux.train!(
        loss_dual,
        ps_dual,
        zip(batches),
        opt,
        cb = Flux.throttle(evalcb, callbacks),
    )
end
