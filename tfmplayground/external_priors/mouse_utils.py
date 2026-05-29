import torch


class ParameterSampler:
    def __init__(
        self,
        a_mu: float = 0.0,
        a_sigma: float = 1.0,
        b_mu: float = 0.0,
        b_sigma: float = 1.0,
        noise_sigma: float = 0.2,
    ):
        self.a_mu = a_mu
        self.a_sigma = a_sigma
        self.b_mu = b_mu
        self.b_sigma = b_sigma
        self.noise_sigma = noise_sigma
        
        self.slopes = None
        self.intercepts = None
    
    def sample_intercept(
        self,
        b_mu: float | None = None,
        b_sigma: float | None = None,
        seq_len: int | None = None,
        ):
        if b_mu is None or b_sigma is None:
            b_mu = self.b_mu
            b_sigma = self.b_sigma
        
        if seq_len is None:
            seq_len = 1
        
        return torch.normal(b_mu, b_sigma, size=(seq_len,))
    
    def sample_slope(
        self,
        a_mu: float | None = None,
        a_sigma: float | None = None,
        seq_len: int | None = None,
    ):
        if a_mu is None or a_sigma is None:
            a_mu = self.a_mu
            a_sigma = self.a_sigma

        if seq_len is None:
            seq_len = 1

        return torch.normal(a_mu, a_sigma, size=(seq_len,))

    def set_intercepts(
        self,
        b_mu: float | None = None,
        b_sigma: float | None = None,
        seq_len: int | None = None,
    ):
        self.intercepts = self.sample_intercept(b_mu=b_mu, b_sigma=b_sigma, seq_len=seq_len)
    
    def set_slopes(
        self,
        a_mu: float | None = None,
        a_sigma: float | None = None,
        seq_len: int | None = None,
    ):
        self.slopes = self.sample_slope(a_mu=a_mu, a_sigma=a_sigma, seq_len=seq_len)
    
    def get_intercept(
        self,
        idx: int = 0
    ):
        if self.intercepts is None:
            raise ValueError("Interceptors have not been set yet.")
        return self.intercepts[idx]
    
    def get_slope(
        self,
        idx: int = 0
    ):
        if self.slopes is None:
            raise ValueError("Slopes have not been set yet.")
        return self.slopes[idx]
