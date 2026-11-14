# Speed benchmark: native Python `rtools` port vs the C++ `FeatureExtraction` binary

`qradiomics.feature.rtools` (released in v0.9.7) is a native Python port
(numpy + scipy; SimpleITK for image I/O only) of the `FeatureExtraction`
binary from [taznux/radiomics-tools](https://github.com/taznux/radiomics-tools).
This report measures its speed against the binary it replaces.

## Summary

- On synthetic ROIs of 1015 to 400014 voxels (single thread, full
  feature set `2isgcr`, 64 bins), the in-process Python port is
  4.5x to 29x faster than the binary. A fresh Python
  process (interpreter start + import + compute) runs at 0.275x to 25.3x the
  binary's speed.
- The gap grows with ROI size; most C++ time on large ROIs is in the shape family.
- Below roughly 4800 voxels one C++ process beats one fresh Python
  process, because importing the port costs about 0.437 s once. Inside a
  running Python process the port is faster on every case measured.
- Neither implementation gains from 8 threads.
- All 30120 printed feature values were identical between the two.

## Method

- **Data.** Synthetic only. Scaling set: PET-like ellipsoid ROIs of
  1015 to 400014 voxels, plus CT-like and integer-valued variants of
  the 49976-voxel case. Parity set: the 40 cases produced by
  `qradiomics.feature.rtools.parity.synthetic_case(seed)` for seeds 0 to 39
  (161 to 15839 ROI voxels, anisotropic spacing, PET/CT/integer intensity).
- **Settings.** `FeatureSelect=2isgcr` (all families), 64 bins, same inputs for both.
- **Repeats.** 1 warm-up + 7 timed repeats per case and path (5 for
  the largest ROI, whose C++ run takes over two minutes). Median and
  interquartile range (IQR) are reported; paths are interleaved per repeat.
- **Threads.** Both paths pinned with `taskset` to the same physical cores
  (no SMT siblings, one NUMA node), with
  `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`, `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` all set
  to the same T (T=1 and T=8).
- **Timing paths** (wall clock, `time.perf_counter`):
  - `cxx`: the binary as a subprocess (process start + read + compute + write);
  - `py_proc` ("end-to-end"): a fresh `python -I -c` subprocess
    (interpreter start + imports + read + compute);
  - `py_inproc` ("in-process"): `feature_text()` in an already warmed
    Python process (read + compute, no import).
- **Speedup** = median(C++) / median(Python path); above 1 means Python is faster.

Hardware/software: AMD EPYC 7763 64-Core Processor (2 sockets x 64 cores, SMT 2), Ubuntu 26.04.1 LTS, Linux 7.0.0-34-generic; Python 3.12.7, numpy 2.4.6, scipy 1.17.1, SimpleITK 2.5.2; C++ binary built against ITK 5.1.2 with `-O3 -DNDEBUG` (ITK statically linked).

## Results (T=1)

Scaling set (PET-like ellipsoids), seconds:

| ROI voxels | C++ median s [IQR] | Python end-to-end median s [IQR] | Python in-process median s [IQR] | speedup end-to-end | speedup in-process |
|---:|---:|---:|---:|---:|---:|
| 1015 | 0.163 [0.162-0.164] | 0.593 [0.585-0.599] | 0.0363 [0.0359-0.0365] | 0.275x | 4.5x |
| 4992 | 0.709 [0.708-0.711] | 0.686 [0.685-0.689] | 0.125 [0.125-0.127] | 1.03x | 5.66x |
| 20035 | 3.42 [3.42-3.44] | 0.971 [0.964-0.978] | 0.251 [0.249-0.251] | 3.53x | 13.7x |
| 49976 | 10.2 [10.1-10.3] | 1.35 [1.35-1.38] | 0.57 [0.569-0.574] | 7.56x | 18x |
| 99967 | 23.9 [23.8-23.9] | 1.97 [1.96-2.08] | 1.11 [1.11-1.13] | 12.1x | 21.6x |
| 206999 | 59.7 [59.3-60.1] | 3.25 [3.24-3.25] | 2.5 [2.5-2.51] | 18.4x | 23.9x |
| 400014 | 137 [137-137] | 5.41 [5.38-5.45] | 4.71 [4.65-4.72] | 25.3x | 29x |

Parity set (40 cases), speedup of each Python path over the binary:

| path | median | min | max | geometric mean |
|---|---:|---:|---:|---:|
| Python end-to-end | 0.57x | 0.135x | 4.85x | 0.53x |
| Python in-process | 4.16x | 2.94x | 17.4x | 4.52x |

## Startup cost

A bare `python -c pass` takes 0.0331 s; `import qradiomics.feature.rtools`
takes 0.47 s, so the import adds 0.437 s per new process. The C++
binary starts and exits in 0.0199 s. This fixed cost is why the
end-to-end Python path loses on small ROIs:

- the end-to-end speedup crosses 1 at about 4800 voxels (log-log
  interpolation between the two nearest scaling cases);
- in the parity set the binary is faster end-to-end on 33 of 40 cases, all
  at or below 3136 voxels; the smallest case where end-to-end Python
  wins has 3751 voxels;
- the in-process path is faster than the binary on every case, family and
  thread setting measured (lowest speedup 2.94x).

For batch extraction, run many ROIs in one Python process (as `qr extract`
does) so the import is paid once.

## Threads

Both implementations are effectively single-threaded in wall-clock terms.
Going from T=1 to T=8 changes the C++ median by a factor of 0.996 to 1.02
and the in-process Python median by 0.994 to 1.03 across the scaling
cases (above 1 means T=8 is faster). At T=8 the binary's CPU-time / wall-time
ratio stays between 1.010 and 1.084.

## Feature families

One family per run (`c` and `r` also compute `i` in both implementations).
The shape family (`s`) dominates C++ time (72% of the full-set time on
the larger ROI below) and is where the port gains most; GLRLM (`r`) is the
least accelerated family.

ROI with 49976 voxels (T=1; full set C++ 10.2 s):

| family | C++ s | Python in-process s | speedup in-process |
|---|---:|---:|---:|
| 2 (2D slice) | 0.0562 | 0.00792 | 7.1x |
| i (intensity) | 0.0899 | 0.00857 | 10.5x |
| g (geometry) | 0.233 | 0.00864 | 27x |
| s (shape) | 6.32 | 0.0861 | 73.5x |
| c (GLCM) | 2.01 | 0.0775 | 25.9x |
| r (GLRLM) | 1.85 | 0.408 | 4.54x |

ROI with 206999 voxels (T=1; full set C++ 59.7 s):

| family | C++ s | Python in-process s | speedup in-process |
|---|---:|---:|---:|
| 2 (2D slice) | 0.13 | 0.0167 | 7.8x |
| i (intensity) | 0.277 | 0.0237 | 11.7x |
| g (geometry) | 0.948 | 0.0236 | 40.2x |
| s (shape) | 43.1 | 0.235 | 183x |
| c (GLCM) | 8.17 | 0.278 | 29.4x |
| r (GLRLM) | 7.59 | 2.04 | 3.72x |

## Parity

The first timed repeat of every run was compared value by value with
`compare_texts`: 30120 / 30120 values printed-identical over 1272
(case, threads, Python path, family) rows; maximum relative difference 0.
The binary prints 6 significant digits, so parity is on printed text.

## Limits

- Synthetic data only; clinical ROIs may have different shape/texture cost
  profiles. No claim is made about clinical ROIs.
- One machine and one build of the binary. The host was shared with other
  jobs; idle cores were chosen at the start, but later interference is possible.
- Inputs sit in the OS page cache, so cold-read I/O is not measured.
- CPU frequency was not fixed (`schedutil` governor, boost on).

## Reproduce

Install qradiomics and build `FeatureExtraction` from
[taznux/radiomics-tools](https://github.com/taznux/radiomics-tools), then
save the script below as `bench_rtools.py` and run it on one pinned core:

```bash
export QRADIOMICS_RTOOLS_BIN=/path/to/radiomics-tools/bin/FeatureExtraction
taskset -c 2 env ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 python bench_rtools.py 0 1 2
```

```python
"""Time and compare the native port vs the C++ binary on synthetic cases."""
import statistics
import sys
import tempfile
import time
from pathlib import Path

from qradiomics.feature.rtools.extractor import feature_text, feature_text_cxx
from qradiomics.feature.rtools.parity import compare_texts, synthetic_case


def timed(fn, *args, reps=5, **kw):
    fn(*args, **kw)  # warm-up
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*args, **kw)
        times.append(time.perf_counter() - t0)
    return out, statistics.median(times)


seeds = [int(s) for s in sys.argv[1:]] or list(range(40))
with tempfile.TemporaryDirectory() as td:
    for seed in seeds:
        img, mask, label = synthetic_case(seed, Path(td))
        cx, t_cx = timed(feature_text_cxx, img, mask, label=label)  # QRADIOMICS_RTOOLS_BIN
        py, t_py = timed(feature_text, img, mask, label=label)
        rows = compare_texts(cx, py)
        exact = sum(1 for r in rows if r[5])
        print(f"seed={seed} cxx={t_cx:.3f}s py={t_py:.3f}s "
              f"speedup={t_cx / t_py:.2f}x identical={exact}/{len(rows)}")
```

This times the in-process path only. The end-to-end path adds interpreter
start and import (about 0.437 s on the machine above).
