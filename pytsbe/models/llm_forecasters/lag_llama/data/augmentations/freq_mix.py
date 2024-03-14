import numpy as np
import torch


@torch.no_grad()
def freq_mix(x, y, rate=0.1, dim=1):
    # Get lengths of the input tensors along the specified dimension.
    x_len = x.shape[dim]
    y_len = y.shape[dim]

    # Concatenate x and y along the specified dimension.
    # x and y represent past and future targets respectively.
    xy = torch.cat([x, y], dim=dim)

    # Perform a real-valued fast Fourier transform (RFFT) on the concatenated tensor.
    xy_f = torch.fft.rfft(xy, dim=dim)

    # Create a random mask with a probability defined by 'rate'.
    # This mask will be used to select which frequencies to manipulate.
    m = torch.rand_like(xy_f, dtype=xy.dtype) < rate

    # Calculate the amplitude of the frequency components.
    amp = abs(xy_f)

    # Sort the amplitudes and create a mask to ignore the most dominant frequencies.
    _, index = amp.sort(dim=dim, descending=True)
    dominant_mask = index > 2
    m = torch.bitwise_and(m, dominant_mask)

    # Apply the mask to the real and imaginary parts of the frequency data,
    # setting masked frequencies to zero.
    freal = xy_f.real.masked_fill(m, 0)
    fimag = xy_f.imag.masked_fill(m, 0)

    # Shuffle the batches in x and y to mix data from different sequences.
    b_idx = np.arange(x.shape[0])
    np.random.shuffle(b_idx)
    x2, y2 = x[b_idx], y[b_idx]

    # Concatenate the shuffled tensors and perform RFFT.
    xy2 = torch.cat([x2, y2], dim=dim)
    xy2_f = torch.fft.rfft(xy2, dim=dim)

    # Invert the mask and apply it to the shuffled frequency data.
    m = torch.bitwise_not(m)
    freal2 = xy2_f.real.masked_fill(m, 0)
    fimag2 = xy2_f.imag.masked_fill(m, 0)

    # Combine the original and shuffled frequency data.
    freal += freal2
    fimag += fimag2

    # Reconstruct the complex frequency data and perform an inverse RFFT.
    xy_f = torch.complex(freal, fimag)
    xy = torch.fft.irfft(xy_f, dim=dim)

    # If the reconstructed data length differs from the original concatenated length,
    # adjust it to maintain consistency.
    if x_len + y_len != xy.shape[dim]:
        xy = torch.cat([x[:, 0:1, ...], xy], 1)

    # Split the reconstructed data back into two parts corresponding to the original x and y.
    return torch.split(xy, [x_len, y_len], dim=dim)
