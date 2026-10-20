"""OpenDPD reporting metrics (Apache-2.0); retained equations and indexing."""

from typing import Tuple, Union
import numpy as np


def magnitude_spectrum(
    input_signal: Union[np.ndarray, np.complex128],
    sample_rate: int,
    nfft: int,
    shift: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Fast Fourier Transform (FFT) of the input signal.

    Parameters:
    - input_signal (np.ndarray[np.complex128]): A 2D numpy array where the first dimension
                                               represents batch size and the second dimension
                                               represents the time sequence of complex numbers.
    - sample_rate (int): The rate at which the input signal was sampled.
    - shift (bool, optional): Whether or not to shift the zero-frequency component to
                              the center of the spectrum. Defaults to False.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple where the first element is the frequency components
                                     and the second element is the FFT of the input signal for each batch.
    """
    spectrum = np.fft.fft(input_signal, n=nfft, axis=-1)
    if shift:
        spectrum = np.fft.fftshift(spectrum, axes=-1)
        freq = np.fft.fftshift(np.fft.fftfreq(input_signal.shape[1], d=1 / sample_rate))
    else:
        freq = np.linspace(0, sample_rate, input_signal.shape[1])
    return (freq, spectrum)


def NMSE(prediction, ground_truth):
    i_hat = prediction[..., 0]
    i_true = ground_truth[..., 0]
    q_hat = prediction[..., 1]
    q_true = ground_truth[..., 1]
    MSE = np.mean(np.square(i_true - i_hat) + np.square(q_true - q_hat), axis=-1)
    energy = np.mean(np.square(i_true) + np.square(q_true), axis=-1)
    NMSE = np.mean(10 * np.log10(MSE / energy))
    return NMSE


def EVM(
    prediction,
    ground_truth,
    sample_rate=int(800000000.0),
    bw_main_ch=200000000.0,
    n_sub_ch=10,
    nperseg=2560,
):
    """
    Calculate EVM based on the given formula.

    Parameters:
    - prediction (array): Prediction/measurements of the PA or DPD-PA outputs.
    - ground_truth (array): Desired output of PA.
    - sample_rate (int, optional): Sampling rate. Default is 800e6.
    - bw_main_ch (float, optional): Bandwidth of main channel. Default is 200e6.
    - n_sub_ch (int, optional): Number of sub-channels. Default is 10.
    - nperseg (int, optional): Length of each segment for the FFT. If not provided, use the entire signal. Default is 2560.

    Returns:
    - float: EVM value in dB.
    """
    prediction_complex = prediction[..., 0] + 1j * prediction[..., 1]
    freq, spectrum_prediction = magnitude_spectrum(
        prediction_complex, sample_rate=sample_rate, nfft=nperseg, shift=True
    )
    ground_truth_complex = ground_truth[..., 0] + 1j * ground_truth[..., 1]
    freq, spectrum_ground_truth = magnitude_spectrum(
        ground_truth_complex, sample_rate=sample_rate, nfft=nperseg, shift=True
    )
    index_left = np.min(np.where(freq >= -bw_main_ch / 2))
    index_right = np.max(np.where(freq <= bw_main_ch / 2))
    channel_index_len = int((index_right - index_left) / n_sub_ch)
    error = np.zeros((prediction.shape[0], n_sub_ch))
    for c in range(n_sub_ch):
        error[:, c] = np.mean(
            np.abs(
                spectrum_prediction[
                    :,
                    index_left
                    + c * channel_index_len : index_left
                    + (c + 1) * channel_index_len,
                ]
                - spectrum_ground_truth[
                    :,
                    index_left
                    + c * channel_index_len : index_left
                    + (c + 1) * channel_index_len,
                ]
            ),
            axis=-1,
        )
        error[:, c] = error[:, c] / np.mean(
            np.abs(
                spectrum_ground_truth[
                    :,
                    index_left
                    + c * channel_index_len : index_left
                    + (c + 1) * channel_index_len,
                ]
            ),
            axis=-1,
        )
    EVM_avg_seq = error.mean(axis=-1)
    EVM_db = 20 * np.log10(np.mean(EVM_avg_seq))
    return EVM_db


def ACLR(prediction, fs=800000000.0, nperseg=2560, bw_main_ch=200000000.0, n_sub_ch=10):
    """
    Calculates the left and right Adjacent Channel Leakage Ratio (ACLR).

    Parameters:
    - frequencies: Array of frequency values from the Welch method.
    - psd: Power Spectral Density values corresponding to the frequencies.
    - bw_main_ch: Bandwidth of the main channel baseband signals (default is 200e6).
    - bw_side_ch: Bandwidth of the side channel baseband signals (default is 20e6).

    Returns:
    - aclr_left: ACLR value for the left adjacent channel.
    - aclr_right: ACLR value for the right adjacent channel.
    """
    complex_signal = IQ_to_complex(prediction)
    freq, psd = power_spectrum(complex_signal, fs=fs, nperseg=nperseg, axis=-1)
    index_left = np.min(np.where(freq >= -bw_main_ch / 2))
    index_right = np.max(np.where(freq <= bw_main_ch / 2))
    sub_ch_index_len = int((index_right - index_left) / n_sub_ch)
    sub_ch_power = np.zeros(n_sub_ch)
    for c in range(n_sub_ch):
        sub_ch_power[c] = np.sum(
            psd[
                index_left
                + c * sub_ch_index_len : index_left
                + (c + 1) * sub_ch_index_len
            ]
        )
    max_sub_ch_power = sub_ch_power.max()
    left_side_ch_power = np.sum(psd[index_left - sub_ch_index_len : index_left])
    aclr_left = np.mean(10 * np.log10(left_side_ch_power / max_sub_ch_power))
    right_side_channel_power = np.sum(psd[index_right : index_right + sub_ch_index_len])
    aclr_right = np.mean(10 * np.log10(right_side_channel_power / max_sub_ch_power))
    return (aclr_left, aclr_right)


def power_spectrum(complex_signal, fs=800000000.0, nperseg=2560, axis=-1):
    """
    Compute the Power Spectral Density (PSD) of a given complex signal using the Welch method.

    Parameters:
    - complex_signal: Input complex signal for which the PSD is to be computed.
    - fs (float, optional): Sampling frequency of the signal. Default is 800e6 (800 MHz).
    - nperseg (int, optional): Number of datasets points to be used in each block for the Welch method. Default is 2560.

    Returns:
    - frequencies_signal_subset: Frequencies at which the PSD is computed.
    - psd_signal_subset: PSD values.
    """
    import numpy as np
    from scipy.signal import welch

    freq, ps = welch(
        complex_signal,
        fs=fs,
        nperseg=nperseg,
        return_onesided=False,
        scaling="spectrum",
        axis=-1,
    )
    half_nfft = int(nperseg / 2)
    freq = np.concatenate((freq[half_nfft:], freq[:half_nfft]))
    ps = np.concatenate((ps[..., half_nfft:], ps[..., :half_nfft]), axis=-1)
    ps = np.mean(ps, axis=0)
    return (freq, ps)


def IQ_to_complex(IQ_signal):
    """
    Convert a multi-dimensional array of I-Q pairs into a 2D array of complex signals.

    Args:
    - IQ_in_segment (3D array): The prediction I-Q datasets with shape (#segments, frame_length, 2).

    Returns:
    - 2D array of shape (#segments, frame_length) containing complex signals.
    """
    I_values = IQ_signal[..., 0]
    Q_values = IQ_signal[..., 1]
    complex_signals = I_values + 1j * Q_values
    return complex_signals
