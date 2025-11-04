# mini_md (pybind11 interop demo)

### Notes
- Pure compute loops release the GIL using `py::gil_scoped_release`.
- NumPy arrays are passed as contiguous `float64` buffers; shapes are validated.
- The LJ kernel here is intentionally naive O(N^2) for clarity. Swap with neighbor lists for scale.
