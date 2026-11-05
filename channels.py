"""
channels.py
-----------
Wireless channel implementations for DeepJSCC experiments.

Convention (follows DeepJSCC / JSCC literature)
-----------------------------------------------
Bandwidth ratio  ρ = k / n
  k = number of complex channel symbols transmitted
  n = number of source real values (3 × 64 × 64 = 12,288)

For ρ = 1/6:  k = 2,048  →  latent has 4,096 real values (2k reals).

Power normalisation: each complex symbol has E[|x|²] = 1.
  The latent vector holds [Re(x_1), Im(x_1), Re(x_2), Im(x_2), …]
  as consecutive pairs, so we normalise to unit average power.

Noise model:
  AWGN   – additive white Gaussian noise, no fading.
  Rayleigh – block fading (one complex gain h per image in a batch),
             receiver knows h (perfect CSI) and equalises.

SNR definition:  SNR = E[|h x|²] / E[|n|²]  in linear scale.
  For AWGN:    noise variance per real component = 1 / (2 · SNR).
  For Rayleigh: same, applied after the channel × h term.
"""

import torch
import unittest


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Power normalisation
# ─────────────────────────────────────────────────────────────────────────────
def power_normalize(z: torch.Tensor) -> torch.Tensor:
    """
    Normalise latent vector so that average power per complex symbol = 1.

    Args:
        z : (B, 2k)  real-valued latent, consecutive pairs = complex symbols.

    Returns:
        z_norm : (B, 2k)  normalised tensor; avg |x_i|^2 = 1.

    Derivation:
        Each symbol is (a, b) → |x|^2 = a^2 + b^2.
        Mean power over the whole vector = z.pow(2).mean(dim=1, keepdim=True)
        This already gives mean of (a^2+b^2)/2 per element, i.e. per real component.
        Mean power per *complex* symbol = z.pow(2).mean() * 2 … but to set it
        to 1 we divide by sqrt(z.pow(2).mean()), not by sqrt(2*mean).

        The formula from the spec:
            z / sqrt(2 * P + eps)   where P = z.pow(2).mean(dim=1, keepdim=True)
        sets E[|x_i|^2] = E[(a^2+b^2)] = 2*E[a^2] = 2*P / (2*P) = 1  ✓
    """
    P = z.pow(2).mean(dim=1, keepdim=True)       # mean of squares of all reals
    return z / torch.sqrt(2.0 * P + 1e-8)        # avg power per complex sym = 1


# ─────────────────────────────────────────────────────────────────────────────
# 2.  AWGN channel
# ─────────────────────────────────────────────────────────────────────────────
def awgn(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Additive White Gaussian Noise channel.

    Args:
        x      : (B, 2k)  power-normalised latent (avg power per complex sym = 1).
        snr_db : scalar dB SNR.

    Returns:
        y : (B, 2k)  received signal = x + noise.

    Noise std per real component = sqrt(1 / (2 * SNR)).
    Then E[|noise|^2] = 2 * (1/(2*SNR)) = 1/SNR  and  E[|x|^2]=1  → SNR ✓.
    """
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_std   = (1.0 / (2.0 * snr_linear)) ** 0.5
    noise       = torch.randn_like(x) * noise_std
    return x + noise


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Rayleigh fading channel (block fading, perfect CSI at receiver)
# ─────────────────────────────────────────────────────────────────────────────
def rayleigh(x: torch.Tensor, snr_db: float):
    """
    Rayleigh block-fading channel.
    One complex fading gain h per image (same h for all symbols in an image).
    Receiver knows h and performs MMSE equalization.

    Args:
        x      : (B, 2k)  power-normalised latent.
        snr_db : scalar dB SNR.

    Returns:
        y_eq   : (B, 2k)  equalised received signal (≈ x + effective noise).
        h_tuple: ((B,1,1), (B,1,1))  real and imaginary parts of h, saved
                  in case the caller needs them for debugging.

    Fading:
        h = (h_r + j*h_i) / sqrt(2),  h_r,h_i ~ N(0,1)
        → E[|h|^2] = 1  (Rayleigh with unit mean power).

    Received (before noise):
        y_r = h_r * x_r - h_i * x_i
        y_i = h_r * x_i + h_i * x_r

    After AWGN:  y = h*x + n,  n ~ CN(0, 1/SNR).

    MMSE equalisation (single-tap, one h per sample):
        x_hat = h* y / (|h|^2 + 1/SNR)
    This is optimal when h is known at the receiver.
    We return x_hat, which for high SNR ≈ x.
    """
    B, k2 = x.shape
    k      = k2 // 2   # number of complex symbols

    # Reshape to complex pairs: (B, k, 2)
    xc = x.view(B, k, 2)            # xc[..., 0]=Re, xc[..., 1]=Im

    # Draw one complex fading gain per image
    h = torch.randn(B, 1, 2, device=x.device) / (2.0 ** 0.5)
    h_r = h[..., 0:1]   # (B, 1, 1)  real part
    h_i = h[..., 1:2]   # (B, 1, 1)  imag part

    # Apply fading:  y = h * xc  (complex multiplication)
    y_r = h_r * xc[..., 0:1] - h_i * xc[..., 1:2]   # (B, k, 1)
    y_i = h_r * xc[..., 1:2] + h_i * xc[..., 0:1]   # (B, k, 1)
    y   = torch.cat([y_r, y_i], dim=-1).view(B, k2)   # (B, 2k)

    # Add AWGN
    y_noisy = awgn(y, snr_db)

    # MMSE equalization:  x_hat = h* y / (|h|^2 + sigma^2)
    #   sigma^2 = 1/(2*SNR) per real component → |noise|^2 per symbol = 1/SNR
    snr_linear = 10.0 ** (snr_db / 10.0)
    sigma2     = 1.0 / snr_linear              # noise power per complex symbol
    h_pow      = h_r ** 2 + h_i ** 2           # |h|^2, shape (B, 1, 1)

    yn = y_noisy.view(B, k, 2)
    # Conjugate multiply: h* y
    xhat_r = h_r * yn[..., 0:1] + h_i * yn[..., 1:2]
    xhat_i = h_r * yn[..., 1:2] - h_i * yn[..., 0:1]
    denom  = h_pow + sigma2                    # (B, 1, 1)
    xhat_r = xhat_r / denom
    xhat_i = xhat_i / denom
    x_eq   = torch.cat([xhat_r, xhat_i], dim=-1).view(B, k2)

    return x_eq, (h_r, h_i)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Convenience wrapper – sample a random SNR per batch during training
# ─────────────────────────────────────────────────────────────────────────────
def sample_snr(snr_min: float = 0.0, snr_max: float = 20.0) -> float:
    """Sample a random SNR (in dB) uniformly from [snr_min, snr_max]."""
    return float(torch.empty(1).uniform_(snr_min, snr_max).item())


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Unit tests
# ─────────────────────────────────────────────────────────────────────────────
class TestChannels(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.B  = 256        # batch
        self.k2 = 4096       # 2 × 2048 complex symbols (ρ = 1/6)

    # ── power normalisation ──────────────────────────────────────────────────
    def test_power_normalize_unit_power(self):
        """Average power per complex symbol must equal 1 after normalisation."""
        z     = torch.randn(self.B, self.k2) * 5.0   # arbitrary scale
        z_n   = power_normalize(z)
        # Power per complex symbol = mean of (a^2 + b^2) over k symbols
        # = 2 × mean of z^2 over all reals
        power = z_n.pow(2).mean(dim=1) * 2.0          # (B,)
        self.assertTrue(
            torch.allclose(power, torch.ones(self.B), atol=1e-4),
            f"Expected power ≈ 1, got {power.mean():.6f}"
        )

    # ── AWGN SNR accuracy ────────────────────────────────────────────────────
    def test_awgn_measured_snr(self):
        """
        Measured SNR from the noisy channel must be close to the requested SNR.
        We use a large batch and many symbols to get a tight estimate.
        """
        for snr_db_target in [0, 5, 10, 15, 20]:
            torch.manual_seed(1)
            B, k2 = 512, 4096
            z = torch.randn(B, k2)
            x = power_normalize(z)                    # avg signal power = 1

            sig_power   = x.pow(2).mean()             # ≈ 0.5 per real element
            y           = awgn(x, snr_db_target)
            noise       = y - x
            noise_power = noise.pow(2).mean()

            measured_snr_db = 10.0 * torch.log10(sig_power / noise_power).item()
            # Signal power per real = 0.5 (because complex power = 1, spread over 2 reals)
            # Noise power per real  = 1/(2*SNR)
            # SNR (per real)        = 0.5 * 2*SNR = SNR  → should match snr_db_target
            self.assertAlmostEqual(
                measured_snr_db, snr_db_target, delta=1.0,
                msg=f"AWGN: wanted {snr_db_target} dB, measured {measured_snr_db:.2f} dB"
            )

    # ── Rayleigh equalized output power ──────────────────────────────────────
    def test_rayleigh_output_shape(self):
        """Equalized output must have the same shape as input."""
        z   = torch.randn(self.B, self.k2)
        x   = power_normalize(z)
        y, _ = rayleigh(x, snr_db=10.0)
        self.assertEqual(y.shape, x.shape)

    def test_rayleigh_high_snr_approx_identity(self):
        """
        At very high SNR, the equalized output should be close to the input
        (fading is perfectly compensated and noise is negligible).
        """
        torch.manual_seed(2)
        B, k2 = 64, 4096
        z     = torch.randn(B, k2)
        x     = power_normalize(z)
        y, _  = rayleigh(x, snr_db=40.0)          # 40 dB = very high SNR
        mse   = (y - x).pow(2).mean().item()
        self.assertLess(mse, 0.05,
            f"Rayleigh @40dB: MSE={mse:.6f} should be < 0.05")

    def test_rayleigh_measured_snr(self):
        """
        Before equalization, measured SNR on the noisy faded signal
        should approximately match requested SNR (averaged over h).
        """
        for snr_db_target in [5, 10, 15]:
            torch.manual_seed(3)
            B, k2 = 1024, 4096
            z     = torch.randn(B, k2)
            x     = power_normalize(z)

            # Re-implement fading without equalization to measure raw SNR
            k = k2 // 2
            xc  = x.view(B, k, 2)
            h   = torch.randn(B, 1, 2) / (2.0 ** 0.5)
            h_r, h_i = h[..., 0:1], h[..., 1:2]
            y_r = h_r * xc[..., 0:1] - h_i * xc[..., 1:2]
            y_i = h_r * xc[..., 1:2] + h_i * xc[..., 0:1]
            y   = torch.cat([y_r, y_i], dim=-1).view(B, k2)

            sig_power = y.pow(2).mean()
            y_noisy   = awgn(y, snr_db_target)
            noise_pow = (y_noisy - y).pow(2).mean()
            measured  = 10.0 * torch.log10(sig_power / noise_pow).item()
            # E[|h|^2] = 1, so signal power ≈ 0.5 (per real), same as AWGN
            self.assertAlmostEqual(
                measured, snr_db_target, delta=1.5,
                msg=f"Rayleigh: wanted {snr_db_target} dB, measured {measured:.2f} dB"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Run tests from command line
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running channel unit tests ...")
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestChannels)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nAll channel tests PASSED [OK]")
    else:
        print("\nSome tests FAILED [X]")

