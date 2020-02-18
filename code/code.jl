# functions for model and training

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

"Return batches with images and their according scaled versions."
function get_scaled_batches(data, data_scaled; batch_size=5)
    return [(reshape(data[:,:,:,i:i+batch_size-1], (60, 60, 1, batch_size)),
            reshape(data_scaled[:,:,:,i:i+batch_size-1], (2, 2, 1, batch_size)))
            for i in range(1, step=batch_size,
                length=Int(size(data)[4]/batch_size))]  # TODO replace with zip

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

"Apply scaling to a single image im."
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
        e1, e2 = 0, 0
    end
    return string(fname, "_sc=", sc, "_bin=", bin, "_e1=", e1, "_e2=", e2, "_ed=", ed, suffix)

end
