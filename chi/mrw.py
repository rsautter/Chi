import numpy as np

def mrw(n: int, lambda2: float = 0.08, L: int = None) -> np.ndarray:
    """
    Bacry(2000) - https://arxiv.org/abs/cond-mat/0005405
    Gera o campo de volatilidade/dissipacao via aproximacao espectral de correlacao logaritmica.

    Parameters:
    - n: Tamanho da serie
    - lambda2: Coeficiente de intermitencia multifratal (ex: 0.03 a 0.25)
    - L: Escala integral (tempo maximo de correlacao)
    """
    if L is None:
        L = n // 2

    # Construcao da autocovariancia teorica logaritmica
    tau = np.arange(n)
    gamma = np.zeros(n)

    # Regularizacao no zero: gamma(0) = lambda^2 * ln(L)
    gamma[0] = lambda2 * np.log(L)
    mask = (tau > 0) & (tau < L)
    gamma[mask] = lambda2 * np.log(L / tau[mask])

    # Simetria circular para embedding de Toeplitz / FFT
    gamma_sym = np.concatenate([gamma, gamma[-2:0:-1]])

    # Densidade espectral de potencia (PSD)
    psd = np.fft.rfft(gamma_sym).real
    psd = np.maximum(psd, 0)  # Limpeza numerica

    # Sintese de processo Gaussiano correlacionado
    white_noise = np.random.normal(size=len(gamma_sym))
    noise_fft = np.fft.rfft(white_noise)
    omega_full = np.fft.irfft(noise_fft * np.sqrt(psd))[:n]

    # Subtrai a variancia para conservacao na media E[exp(omega)] ~ 1
    var_omega = lambda2 * np.log(L)
    omega = omega_full - var_omega

    # Taxa de dissipacao / densidade de eventos extremos
    epsilon = np.exp(omega)
    return epsilon