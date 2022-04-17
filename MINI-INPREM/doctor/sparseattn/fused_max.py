import torch
from torch import nn
from .base import _BaseBatchProjection
from .sparsemax import SparsemaxFunction

class Fusedmax(nn.Module):
    def __init__(self, alpha=1):
        self.alpha = alpha
        super(Fusedmax, self).__init__()

    def forward(self, x, lengths=None):
        fused_prox = FusedProxFunction(self.alpha)
        sparsemax = SparsemaxFunction()
        return sparsemax(fused_prox(x, lengths), lengths)


class FusedProxFunction(_BaseBatchProjection):

    def __init__(self, alpha=1):
        self.alpha = alpha

    def project(self, x):
        x_np = x.detach().numpy().copy()
        prox_tv1d(x_np, self.alpha)
        y_hat = torch.from_numpy(x_np)
        return y_hat

    def project_jv(self, dout, y_hat):
        dout = dout.clone()
        _inplace_fused_prox_jv(y_hat.detach().numpy(), dout.numpy())
        return dout


# ===== From C Code =====
def prox_tv1d(w, stepsize):
    """
    Computes the proximal operator of the 1-dimensional total variation operator.

    This solves a problem of the form

         argmin_x TV(x) + (1/(2 stepsize)) ||x - w||^2

    where TV(x) is the one-dimensional total variation

    Parameters
    ----------
    w: array
        vector of coefficieents
    stepsize: float
        step size (sometimes denoted gamma) in proximal objective function

    References
    ----------
    Condat, Laurent. "A direct algorithm for 1D total variation denoising."
    IEEE Signal Processing Letters (2013)
    """

    #cdef long width, k, k0, kplus, kminus
    #cdef floating umin, umax, vmin, vmax, twolambda, minlambda
    width = w.size

    # /to avoid invalid memory access to input[0] and invalid lambda values
    if width > 0 and stepsize >= 0:
        k, k0 = 0, 0			# k: current sample location, k0: beginning of current segment
        umin = stepsize  # u is the dual variable
        umax = - stepsize
        vmin = w[0] - stepsize
        vmax = w[0] + stepsize	  # bounds for the segment's value
        kplus = 0
        kminus = 0 	# last positions where umax=-lambda, umin=lambda, respectively
        twolambda = 2.0 * stepsize  # auxiliary variable
        minlambda = -stepsize		# auxiliary variable
        while True:				# simple loop, the exit test is inside
            while k >= width-1: 	# we use the right boundary condition
                if umin < 0.0:			# vmin is too high -> negative jump necessary
                    while True:
                        w[k0] = vmin
                        k0 += 1
                        if k0 > kminus:
                            break
                    k = k0
                    kminus = k
                    vmin = w[kminus]
                    umin = stepsize
                    umax = vmin + umin - vmax
                elif umax > 0.0:    # vmax is too low -> positive jump necessary
                    while True:
                        w[k0] = vmax
                        k0 += 1
                        if k0 > kplus:
                            break
                    k = k0
                    kplus = k
                    vmax = w[kplus]
                    umax = minlambda
                    umin = vmax + umax -vmin
                else:
                    vmin += umin / (k-k0+1)
                    while True:
                        w[k0] = vmin
                        k0 += 1
                        if k0 > k:
                            break
                    return
            umin += w[k + 1] - vmin
            if umin < minlambda:       # negative jump necessary
                while True:
                    w[k0] = vmin
                    k0 += 1
                    if k0 > kminus:
                        break
                k = k0
                kminus = k
                kplus = kminus
                vmin = w[kplus]
                vmax = vmin + twolambda
                umin = stepsize
                umax = minlambda
            else:
                umax += w[k + 1] - vmax
                if umax > stepsize:
                    while True:
                        w[k0] = vmax
                        k0 += 1
                        if k0 > kplus:
                            break
                    k = k0
                    kminus = k
                    kplus = kminus
                    vmax = w[kplus]
                    vmin = vmax - twolambda
                    umin = stepsize
                    umax = minlambda
                else:                   # no jump necessary, we continue
                    k += 1
                    if umin >= stepsize:		# update of vmin
                        kminus = k
                        vmin += (umin - stepsize) / (kminus - k0 + 1)
                        umin = stepsize
                    if umax <= minlambda:	    # update of vmax
                        kplus = k
                        vmax += (umax + stepsize) / (kplus - k0 + 1)
                        umax = minlambda


def _inplace_fused_prox_jv(y_hat, dout):
    n_features = dout.shape[0]
    #cdef Py_ssize_t i, last_ix
    #cdef unsigned int n
    #cdef floating acc

    last_ix = 0
    n = 0
    acc = 0

    for i in range(n_features + 1):
        if i in (0, n_features) or y_hat[i] != y_hat[i - 1]:
            if i > 0:
                dout[last_ix:i] = acc / n

            if i < n_features:
                last_ix = i
                acc = dout[i]
                n = 1

        else:
            acc += dout[i]
            n += 1
    return dout

