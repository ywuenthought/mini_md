import numpy as np
import numpy.typing as npt

def lj_energy_forces(
    coords: npt.NDArray[np.float64],
    forces: npt.NDArray[np.float64],
    cutoff: float,
    epsilon: float,
    sigma: float,
) -> float: ...
