import torch


@torch.no_grad()
def freq_mask(x, y, rate=0.1, dim=1):
    # Get lengths of the input tensors along the specified dimension.
    x_len = x.shape[dim]
    y_len = y.shape[dim]

    # Concatenate x and y along the specified dimension.
    # x and y represent past and future targets respectively.
    xy = torch.cat([x, y], dim=dim)

    # Perform a real-valued fast Fourier transform (RFFT) on the concatenated tensor.
    # This transforms the time series data into the frequency domain.
    xy_f = torch.fft.rfft(xy, dim=dim)

    # Create a random mask with a probability defined by 'rate'.
    # This mask will be used to randomly select frequencies to be zeroed out.
    m = torch.rand_like(xy_f, dtype=xy.dtype) < rate

    # Apply the mask to the real and imaginary parts of the frequency data,
    # setting the selected frequencies to zero. This 'masks' those frequencies.
    freal = xy_f.real.masked_fill(m, 0)
    fimag = xy_f.imag.masked_fill(m, 0)

    # Combine the masked real and imaginary parts back into complex frequency data.
    xy_f = torch.complex(freal, fimag)

    # Perform an inverse RFFT to transform the data back to the time domain.
    # The masked frequencies will affect the reconstructed time series.
    xy = torch.fft.irfft(xy_f, dim=dim)

    # If the reconstructed data length differs from the original concatenated length,
    # adjust it to maintain consistency. This step ensures the output shape matches the input.
    if x_len + y_len != xy.shape[dim]:
        xy = torch.cat([x[:, 0:1, ...], xy], 1)

    # Split the reconstructed data back into two parts corresponding to the original x and y.
    return torch.split(xy, [x_len, y_len], dim=dim)
