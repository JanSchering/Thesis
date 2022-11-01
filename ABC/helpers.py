#%%
from typing import Callable
import numpy as np
import torch as t
import matplotlib.pyplot as plt

def gaussian_pdf(mu: t.Tensor, sigma_sq: t.Tensor) -> Callable:
    # assume scalar mean
    assert mu.numel() == 1
    # assume scalar variance
    assert sigma_sq.numel() == 1

    def pdf(x: t.Tensor) -> float:
        """
        Calculate the probability density at points <x> for a Gaussian distribution
        N(<mu>,<sigma>)

        x (torch.Tensor): The points to evaluate the density at.
        """
        return (1 / (t.sqrt(2 * t.tensor(np.pi)) * t.sqrt(sigma_sq))) * t.exp(
            -((x - mu) ** 2) / (2 * sigma_sq)
        )

    return pdf

if __name__ == "__main__":
    print("test the gaussian pdf function")
    pdf_1 = gaussian_pdf(t.tensor(0), t.tensor(1))
    pdf_2 = gaussian_pdf(t.tensor(0), t.tensor(0.2))
    pdf_3 = gaussian_pdf(t.tensor(1.0), t.tensor(5.0))
    x = t.linspace(-5, 5, 1000)
    y1 = pdf_1(x)
    y2 = pdf_2(x)
    y3 = pdf_3(x)
    plt.plot(x, y1, label="mu:0, sigma^2:1")
    plt.plot(x, y2, label="mu:0, sigma^2:0.2")
    plt.plot(x, y3, label="mu:1, sigma^2:5")
    plt.legend()
    plt.show()

# %%
