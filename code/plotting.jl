# code for plotting
using Plots
using Images
using ColorSchemes

"Create a list of indices to emulate differnet states of latent variables z
with dimension dim=dim(z) and one of the entries set to +-value, rest to 0."
function get_indices(dim, value)
    input = Int32.(zeros(dim))
    input[1] = 1
    indices = []
    for i = 1:dim
        push!(indices, input .* value)
        push!(indices, input .* (-value))
        input = circshift(input, 1)
    end
    return indices
end

"Plot reconstruction for by feeding z-vectors with one 'hot' dimension, i.e.
one dimension set to +/-value and others to zero."
function make_reconstruction_plot(decoder, value, dim, title)
        # make a reconstruction plot by feeding vectors with one "hot" dimension
        # into decoder for +-value
    indices = get_indices(dim, value)
    #TODO try clim param
    plots = [Plots.heatmap(
        Flux.Tracker.data(decoder(i))[:, :],
        seriescolor = cgrad(ColorSchemes.gray.colors),
        title = string(i),
    ) for i in indices]
    png(
        plot(plots..., layout = (dim, 2), size = (1000, 250 * (dim))),
        string(path_to_project, plot_path, title, ".png"),
    )
end
function plot_output_for(image, vae, title)
    output = vae(reshape(image, (60, 60, 1, 1)))
    plot = Plots.heatmap(
        Flux.Tracker.data(output)[:, :],
        seriescolor = cgrad(ColorSchemes.gray.colors),
        title = "Output image",
    )
    plot2 = Plots.heatmap(
        reshape(image, (60, 60)),
        seriescolor = cgrad(ColorSchemes.gray.colors),
        title = "Input image",
    )
    plots = [plot, plot2]
    png(
        Plots.plot(plots..., layout = (1, 2), size = (850, 350)),
        string(path_to_project, plot_path, title, ".png"),
    )
end

function make_superduperplot(vae1_out, vae2_out;from=1, to=100, z2_on=true)
    sequence = [reshape(data[:, :, :, i], (60, 60, 1, 1)) for i = 1:size(
        data,
        4,
    )][from:to]
    gif1 = @animate for i = from:to
        plot_reconstruction(i, sequence, data_scaled, vae1_out, vae2_out)
    end
    gif(gif1, "reconstructed_super.gif")
end

function plot_reconstruction(i, sequence, data_scaled, vae1_out, vae2_out)
    out3 = vae2_out(sequence[i])
    out3 = reshape(Flux.Tracker.data(out3), (2, 2))
    plt3 = Plots.heatmap(
        out3,
        clim = (0, 1),
        seriescolor = cgrad(ColorSchemes.gray.colors),
        title = "output_vae2",
    )
    out2 = Float32.(reshape(data_scaled[:, :, :, i], (2, 2)))
    plt2 = Plots.heatmap(
        out2,
        seriescolor = cgrad(ColorSchemes.gray.colors),
        title = "target_vae2",
        clim = (0, 1),
    )
    plt0 = Plots.heatmap(
        reshape(sequence[i], (60, 60)),
        seriescolor = cgrad(ColorSchemes.gray.colors),
        title = "input",
        clim=(0, 1)
    )
    out1 = vae1_out(sequence[i])
    out1 = reshape(Flux.Tracker.data(out1), (60, 60))
    plt1 = Plots.heatmap(
        out1,
        clim = (0, 1),
        seriescolor = cgrad(ColorSchemes.gray.colors),
        title = "output_vae2",
    )
    plots = [plt0, plt1, plt2, plt3]
    return Plots.plot(plots..., layout = (1, 4), size = (1800, 300))
end
