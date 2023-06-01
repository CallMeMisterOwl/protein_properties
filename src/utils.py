import random
import numpy as np
from pytorch_lightning import seed_everything
import torch  


def seed_all(seed=13):
    """
    Seed function to guarantee the reproducibility of the code.

    See https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
    """
    seed_everything(seed, workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)

# https://gist.github.com/slowkow/06c6dba9180d013dfd82bec217d22eb5
def align_sequences_nw(x: str, y: str, match=100, mismatch=100, gap=1):
    nx = len(x)
    ny = len(y)
    
    # Optimal score at each possible pair of characters.
    F = np.zeros((nx + 1, ny + 1))
    F[:, 0] = np.linspace(0, -nx * gap, nx + 1)
    F[0, :] = np.linspace(0, -ny * gap, ny + 1)
    
    # Pointers to trace through an optimal alignment.
    P = np.zeros((nx + 1, ny + 1))
    P[:, 0] = 3
    P[0, :] = 4
    
    # Temporary scores.
    t = np.zeros(3)
    
    for i in range(nx):
        for j in range(ny):
            if x[i] == y[j]:
                t[0] = F[i, j] + match
            else:
                t[0] = F[i, j] - mismatch
            t[1] = F[i, j + 1] - gap
            t[2] = F[i + 1, j] - gap
            
            tmax = np.max(t)
            F[i + 1, j + 1] = tmax
            
            # Update pointers using bitwise OR
            P[i + 1, j + 1] = ((t[0] == tmax) << 1) | ((t[1] == tmax) << 2) | ((t[2] == tmax) << 3)
    
    # Trace through an optimal alignment.
    i = nx
    j = ny
    rx = []
    ry = []
    
    while i > 0 or j > 0:
        pointer = int(P[i, j])  # Convert to integer
        
        if pointer & 0b0010:  # Check if bit 1 is set
            rx.append(x[i - 1])
            ry.append(y[j - 1])
            i -= 1
            j -= 1
        elif pointer & 0b0100:  # Check if bit 2 is set
            rx.append(x[i - 1])
            ry.append('-')
            i -= 1
        elif pointer & 0b1000:  # Check if bit 3 is set
            rx.append('-')
            ry.append(y[j - 1])
            j -= 1
    
    # Reverse the strings.
    rx = ''.join(rx)[::-1]
    ry = ''.join(ry)[::-1]
    
    return [rx, ry]
