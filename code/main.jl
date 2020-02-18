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
scal_func(x) = sigmoid(12*(x/scaling^2-0.5)) # this has yielded good results

# training
train_seperately = true  # train vae1 and vae2 seperately before combination
epochs_dual = 1  # epochs of combined training
epochs_vae1 = 1  # epochs of seperate training for vae1
epochs_vae2 = 1  # epochs of seperate training for vae1
callbacks = 5

# model
latent_dim1 = 4  # number of z-variaables for vae1
latent_dim2 = 1  # number of z-variables for vae2
inter1 = 8  # size of image after interaction
inter2 = 2
store_params = true  # store models after training?
load_params = true  # load models from checkpoint before training?

# plotting
plot_recon = false  # plot data reconstruction?
not_on_server = false  # we don't want to plot on server, leads to problems

# paths TODO: update
dual_params1 = "params1_int_sc=1_bin=1_e1=0_e2=0_ed=200.jld2"
dual_params2 = "params2_int_sc=1_bin=1_e1=0_e2=0_ed=200.jld2"
plot_input_path = "input.gif"
plot_output_path = "output.gif"
# change accordingly on your machine
path_to_project = "/home/flo/projects/thesis/code/"
params_path = "params/"
plot_path = "plots/"
sep_params1 = ""
sep_params2 = ""

################################################################################
# all the abstractions used in the following are defined here
# if you don't understand something try looking up the docstring in the console
# with '?[function-name]'
# the interested reader can go to code.jl and see what's under the hood
include(string(path_to_project, "code.jl"))
################################################################################

########################### load data ##########################################
data = get_dummy_data(
    string(path_to_project, "../dummydata/dummyData2000.mat"),
    binary = binary,
)
data_scaled = get_scaled(data)
batches_scaled = get_scaled_batches(data, data_scaled)
# create an array of array-batches of size batch_size
batches = get_batches(data, batch_size = 5)


########################### models #############################################
# load model layers and methods of how to use them
include(string(path_to_project, "model.jl"))

if load_params & (sep_params1 != "") & (sep_params2 != "")
    # load model from pretrained checkpoint defined at sep_params
    # note: changes to code above have no effect, disable this piece if needed
    (@load string(path_to_project, params_path, load_sep) conv_enc1 conv2_enc1 μ1 logσ1 interaction1 transp1_dec1 transp2_dec1 transp3_dec1)
    (@load string(path_to_project, params_path, load_sep2) conv_enc2 μ2 logσ2 interaction2 transp1_dec2 transp2_dec2 scaled_dec)
end


########################## train model #########################################
# the following functions are defined in model.jl and are just a call of
# Flux.train! with according parameters
if train_seperately
# train cvae1 (small receptive field)
    train_vae1(batches, epochs_vae1)
    if scaled_vae2
        train_vae2_scal(batches_scaled, epochs_vae2)
    else
        # train cvae2 (big receptive field)
        train_vae2(batches, epochs_vae2)
    end
end

# store model in specified folder
if store_params
    (@save string(path_to_project, params_path, get_filename("params1_sep", ".jld2")) conv_enc1 conv2_enc1 μ1 logσ1 interaction1 transp1_dec1 transp2_dec1 transp3_dec1)
    (@save string(path_to_project, params_path, get_filename("params2_sep", ".jld2")) conv_enc2 μ2 logσ2 interaction2 transp1_dec2 transp2_dec2 scaled_dec)
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
    # overwrite interaction layer with interactions of seperately
    # trained models vae1 and vae2
    interact0 = get_interaction(interaction1, interaction2)
end

if load_params
    #TODO: change with filename function
    (@load string(path_to_project, params_path, dual_params1) conv_enc1 conv2_enc1 μ1 logσ1 interaction1 transp1_dec1 transp2_dec1 transp3_dec1 interact0)
    (@load string(path_to_project, params_path, dual_params2) conv_enc2 μ2 logσ2 interaction2 transp1_dec2 transp2_dec2 scaled_dec)
end

if scaled_vae2
    train_dual_scal(batches_scaled, epochs_dual)
else
    train_dual(batches, epochs_dual)
end

if store_params
    (@save string(path_to_project, params_path, get_filename("params1_int", ".jld2")) conv_enc1 conv2_enc1 μ1 logσ1 interaction1 transp1_dec1 transp2_dec1 transp3_dec1 interact0)
    (@save string(path_to_project, params_path, get_filename("params2_int", ".jld2")) conv_enc2 μ2 logσ2 interaction2 transp1_dec2 transp2_dec2 scaled_dec)
end

if not_on_server
    plot_output_for(data[:, :, :, 1], cvae1_out, "im1_vae1_int")
    plot_output_for(data[:, :, :, 7], cvae1_out, "im2_vae1_int")
    plot_output_for(data[:, :, :, 1], cvae2_out, "im1_vae2_int")
    plot_output_for(data[:, :, :, 7], cvae2_out, "im2_vae2_int")
end
