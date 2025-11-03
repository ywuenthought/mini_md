// src/mini_md/_core.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifdef MINI_MD_USE_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

using DoubleArray = py::array_t<
    double,
    // 1. C-contiguous in memory
    // 2. type-cast to double if needed
    py::array::c_style | py::array::forcecast>;

// Simple Lennard-Jones potential/forces for demonstration.
// coords: (N,3) double array
// forces: (N,3) double array (in/out, will be written)
// cutoff: if >0, use simple cut-off without neighbor list
// Returns total potential energy.
double lj_energy_forces(
    DoubleArray coords,
    DoubleArray forces,
    double cutoff,
    double epsilon,
    double sigma)
{
    auto pcor = coords.request();
    auto pfor = forces.request();

    if (pcor.ndim != 2 || pcor.shape[1] != 3)
        throw std::runtime_error("coords must be (N,3)");
    if (pfor.ndim != 2 || pfor.shape[1] != 3)
        throw std::runtime_error("forces must be (N,3)");
    if (pcor.shape[0] != pfor.shape[0])
        throw std::runtime_error("N mismatch");

    const ssize_t N = pcor.shape[0];
    const size_t N3 = static_cast<size_t>(N) * 3u;
    const double *c = static_cast<double *>(pcor.ptr);

    double *f = static_cast<double *>(pfor.ptr);
    std::fill(f, f + N3, 0.0);

    const double rc2 =
        cutoff > 0.0
            ? cutoff * cutoff
            : std::numeric_limits<double>::infinity();
    const double sg2 = sigma * sigma;

    double energy = 0.0;
    {
        py::gil_scoped_release nogil;

        size_t nt = 1;
#ifdef MINI_MD_USE_OPENMP
        nt = omp_get_max_threads();
#endif

        // Temporary arrays to avoid race conditions
        std::vector<double> f_tls(nt * N3, 0.0);

// Naive O(N^2) for clarity; replace with neighbor lists in real use.
// Parallelize outer loop optionally.
#ifdef MINI_MD_USE_OPENMP
#pragma omp parallel for reduction(+ : energy) schedule(static)
#endif
        for (ssize_t i = 0; i < N; ++i)
        {
            const size_t i3 = 3u * static_cast<size_t>(i);
            const double xi = c[i3 + 0];
            const double yi = c[i3 + 1];
            const double zi = c[i3 + 2];

            size_t tid = 0;
#ifdef MINI_MD_USE_OPENMP
            tid = omp_get_thread_num();
#endif
            double *ft =
                f_tls.data() + tid * N3;

            for (ssize_t j = i + 1; j < N; ++j)
            {
                const size_t j3 = 3u * static_cast<size_t>(j);
                const double dx = xi - c[j3 + 0];
                const double dy = yi - c[j3 + 1];
                const double dz = zi - c[j3 + 2];
                const double r2 = dx * dx + dy * dy + dz * dz;

                if (r2 > rc2 || r2 == 0.0)
                    continue;

                const double inv_r2 = 1.0 / r2;
                // (sigma/r)^2
                const double sr2 = sg2 * inv_r2;
                // (sigma/r)^6
                const double sr6 = sr2 * sr2 * sr2;
                // (sigma/r)^12
                const double sr12 = sr6 * sr6;
                const double e = 4.0 * epsilon * (sr12 - sr6);
                energy += e;

                // F = 24*epsilon*(2*sr12 - sr6) * (1/r^2) * r_vec
                const double fscalar = 24.0 * epsilon * (2.0 * sr12 - sr6) * inv_r2;
                const double fx = fscalar * dx;
                const double fy = fscalar * dy;
                const double fz = fscalar * dz;

                ft[i3 + 0] += fx;
                ft[i3 + 1] += fy;
                ft[i3 + 2] += fz;
                ft[j3 + 0] -= fx;
                ft[j3 + 1] -= fy;
                ft[j3 + 2] -= fz;
            }
        }

#ifdef MINI_MD_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t k = 0; k < N3; ++k)
        {
            for (size_t t = 0; t < nt; ++t)
            {
                f[k] += f_tls[t * N3 + k];
            }
        }
    }

    return energy;
}

PYBIND11_MODULE(mini_md_core, m)
{
    m.def("lj_energy_forces", &lj_energy_forces);
}
