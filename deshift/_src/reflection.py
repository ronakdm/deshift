import torch
import numpy as np

# reference implementations

def project_odd_ref(v):
    u = v.copy()
    for i in range(len(v) // 2):
        if v[2 * i] > v[2 * i + 1]:
            mid = (v[2 * i] + v[2 * i + 1]) / 2.
            u[2 * i] = mid
            u[2 * i + 1] = mid
    return u

def project_even_ref(v):
    u = v.copy()
    for i in range(len(v) // 2):
        if len(v) > 2 * i + 2 and v[2 * i + 1] > v[2 * i + 2]:
            mid = (v[2 * i + 1] + v[2 * i + 2]) / 2.
            u[2 * i + 1] = mid
            u[2 * i + 2] = mid
    return u

def project_monotone_cone_ref(v, max_iter=10):
    """
    dykstra's projection algorithm
    """
    n = len(v)
    z = v.copy()
    p = np.zeros(shape=(n,))
    q = np.zeros(shape=(n,))

    for _ in range(max_iter):
        y = project_even_ref(z + p)
        p = z + p - y
        z = project_odd_ref(y + q)
        q = y + q - z

    return z

def reflection_oracle_ref(
        spectrum: np.ndarray, 
        shift_cost: float,
        penalty: str,
        losses: np.ndarray
    ):
    """
    implementation-in-development
    """
    if shift_cost < 1e-12:
        return torch.from_numpy(spectrum[np.argsort(np.argsort(losses))])
    sample_size = len(losses)
    scaled_losses = losses / shift_cost
    perm = np.argsort(losses)
    sorted_losses = scaled_losses[perm]

    if penalty == "chi2":
        v = sorted_losses + 1 / sample_size - spectrum
        primal_sol = project_monotone_cone_ref(v)
    else:
        raise NotImplementedError
    inv_perm = np.argsort(perm)
    primal_sol = primal_sol[inv_perm]
    if penalty == "chi2":
        q = scaled_losses - primal_sol + 1 / sample_size
    return torch.from_numpy(q).float(), primal_sol


# pytorch implementations

def project_odd(v):
    u = v.clone() 
    mids = []
    if len(u) >= 2:
        evens = v[1::2]
        odds = v[0:2*len(evens):2]
        if torch.any(odds > evens):
            mids = (odds[odds > evens] + evens[odds > evens]) / 2.
            odds[odds > evens] = mids
            evens[odds > evens] = mids
            u[1::2] = evens
            u[0:2*len(evens):2] = odds
    return u, len(mids)

def project_even(v):
    u = v.clone()
    mids = []
    if len(u) >= 3:
        odds = v[2::2]
        evens = v[1:2*len(odds):2]
        if torch.any(evens > odds):
            mids = (odds[evens > odds] + evens[evens > odds]) / 2.
            odds[evens > odds] = mids
            evens[evens > odds] = mids
            u[2::2] = odds
            u[1:2*len(odds):2] = evens
    return u, len(mids)

def project_monotone_cone(v, max_iter=16):
    """
    dykstra's projection algorithm
    """
    n = len(v)
    z = v.clone()
    p = torch.zeros(n, device=v.get_device())
    q = torch.zeros(n, device=v.get_device())

    for _ in range(max_iter):
        z_prev = z.clone()

        y, even_changes = project_even(z + p)
        p = z + p - y
        z, odd_changes = project_odd(y + q)
        q = y + q - z

        error = torch.norm(z - z_prev)
        if even_changes + odd_changes == 0 or error < 1e-8:
            break
    return z


def reflection_oracle(
        spectrum: torch.Tensor, 
        shift_cost: float,
        penalty: str,
        losses: torch.Tensor
    ):
    """
    implementation-in-development
    """
    if shift_cost < 1e-12:
        return spectrum[torch.argsort(torch.argsort(losses))]
    sample_size = len(losses)
    scaled_losses = losses / shift_cost
    perm = torch.argsort(losses)
    sorted_losses = scaled_losses[perm]

    if penalty == "chi2":
        v = sorted_losses + 1 / sample_size - spectrum
        primal_sol = project_monotone_cone(v)
    else:
        raise NotImplementedError
    inv_perm = torch.argsort(perm)
    primal_sol = primal_sol[inv_perm]
    if penalty == "chi2":
        q = scaled_losses - primal_sol + 1 / sample_size
    return q, primal_sol