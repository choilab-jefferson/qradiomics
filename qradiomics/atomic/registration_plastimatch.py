"""Deformable/rigid image registration via the plastimatch CLI.

plastimatch (https://plastimatch.org) is the de-facto open-source DIR
(deformable image registration) engine in radiotherapy research and QA --
it ships a purpose-built multi-stage rigid -> affine -> B-spline pipeline
that most TPS/QA workflows already trust for CT/CBCT alignment. Rather than
reimplementing DIR, this module shells out to the ``plastimatch register``
and ``plastimatch warp`` subcommands, matching the "wrap the binary as a
subprocess" pattern already used elsewhere in this project for DICOM->NRRD
conversion (see wiki/architecture/SEMANTIC_ATOMIC_PROCESSES.md).

Unlike :mod:`qradiomics.atomic.registration` (the SimpleITK backend, which
is in-memory ``sitk.Image``-based and rigid/affine only), this backend is
file-path based -- plastimatch reads/writes MHA/MHD/NRRD/NIfTI directly --
and it is the only backend here that supports true deformable (B-spline)
registration out of the box.

Requires the ``plastimatch`` binary on PATH (e.g. ``apt install
plastimatch``, or build from source -- see https://plastimatch.org). There
is no PyPI wheel for the binary itself, so it is not a pip dependency of
this package.

The per-stage command-file parameters below follow the plastimatch "Image
registration command file" DSL, verified against ``man plastimatch``
(PLASTIMATCH REGISTER section) on a locally installed plastimatch 1.10.

World-origin rotation-center trap (and why this module works in a shifted frame)
----------------------------------------------------------------------------
plastimatch's ``xform=rigid``/``xform=affine`` (``impl=itk``) stages rotate
about the world-coordinate origin (0, 0, 0) unless told otherwise. DICOM
patient-coordinate CT/CBCT is almost never near the world origin (real
scanners routinely place it hundreds of mm to over a meter away). A rigid
rotation about a distant center turns even a small angular optimizer step
into a huge linear displacement at the image (lever-arm effect:
displacement ~= angle * distance from the rotation center), which throws
every sampled voxel outside the moving image's buffer. ITK aborts the
metric evaluation, and plastimatch **returns the identity transform with
exit code 0** -- a silent failure: no error, a transform file is written,
downstream stages happily consume an unregistered image.

The obvious fix -- seed the optimizer's rotation center via ``xform_in`` --
turns out to be insufficient on its own: plastimatch's own ``xform_out``
re-serializes the *converged* result with ``FixedParameters`` collapsed back
to ``0 0 0`` **without recomputing the translation to compensate**. Reading
that file back (via ``sitk.ReadTransform`` or ``plastimatch warp --xf``)
therefore reproduces the same lever-arm error the seeding was meant to
avoid -- verified directly: applying the as-written transform to a
known-correspondence point landed >10 mm off, while re-applying the exact
same rotation+translation with the *true* center (the one actually used
during optimization) landed within 0.001 mm.

The fix used here instead avoids ever asking plastimatch to rotate about a
distant point: both images are translated (a pure metadata/header change --
no resampling, no interpolation, no information loss) so the fixed image's
own geometric center sits at the world origin before ``plastimatch
register`` ever runs. plastimatch's origin-default behavior is then exactly
correct (proven separately: images placed at or near the origin register
correctly with zero special-casing). Afterward:

* ``output_image`` has its origin shifted back to the original frame (a
  metadata edit -- the resampled pixel data plastimatch produced is already
  correct, only its coordinate label needs restoring).
* ``output_transform``, if the run ends in ``rigid``/``affine``, is
  re-expressed in the original frame by moving its (now safely near-zero)
  rotation center back by the same shift -- this is a pure translation of
  the *center*, not a re-optimization, and is exact.
* If the run ends in ``bspline``, the same shift is subtracted from the
  native "MGH_GPUIT_BSP" transform file's ``img_origin`` header field,
  which anchors the deformation field to world space the same way an
  image's origin does.

Both corrections were verified end-to-end against known-correspondence
points (sub-millimeter residual). Callers therefore get back an
``output_image`` and ``output_transform`` that are valid in the *original*
(un-shifted) coordinate frame -- exactly as if plastimatch had handled a
distant rotation center correctly in the first place. See the companion
demonstration and controlled experiments in
https://github.com/choilab-jefferson/medimage (Ch. 11, "Registration, and
How It Fails Quietly"), which independently reproduced the world-origin
symptom (though not this specific serialization bug -- discovered while
porting that chapter's fix into this module).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

__all__ = [
    "PlastimatchResult",
    "plastimatch_available",
    "register_pair_plastimatch",
    "warp_with_plastimatch_transform",
]

# Log substrings that mean plastimatch/ITK gave up mid-optimization and fell
# back to (or never left) the identity transform, despite exiting 0. Kept as
# a backstop even though the origin-shift below is designed to prevent the
# root cause from ever triggering.
_SILENT_FAILURE_MARKERS = (
    "Too many samples map outside moving image buffer",
    "ITK failed with too few samples",
)

_DEFAULT_STAGES: Tuple[str, ...] = ("rigid", "affine", "bspline")
_LINEAR_STAGES = {"rigid", "affine"}

# Defaults chosen to mirror the atomic.registration sitk backend (Mattes /
# mutual-information metric -- these images are usually CT vs CBCT, i.e.
# intra-patient but not identical intensity distributions) while following
# the exact stage examples from `man plastimatch` (PLASTIMATCH REGISTER).
_STAGE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "rigid": {
        "xform": "rigid",
        "optim": "rsg",
        "impl": "itk",
        "metric": "mi",
        "max_its": "100",
    },
    "affine": {
        "xform": "affine",
        "optim": "rsg",
        "impl": "itk",
        "metric": "mi",
        "max_its": "100",
    },
    "bspline": {
        "xform": "bspline",
        "impl": "plastimatch",
        "threading": "openmp",
        "metric": "mi",
        "max_its": "60",
        "regularization_lambda": "0.005",
        "grid_spac": "20 20 20",
        "res": "4 4 2",
    },
}


@dataclass
class PlastimatchResult:
    """Outcome of :func:`register_pair_plastimatch`."""

    warped_image_path: Path
    transform_path: Path
    command_file_path: Path
    stages: Tuple[str, ...]
    log: str


def plastimatch_available() -> bool:
    """True if the ``plastimatch`` binary is importable via PATH."""
    return shutil.which("plastimatch") is not None


def _require_plastimatch() -> None:
    if shutil.which("plastimatch") is None:
        raise RuntimeError(
            "plastimatch binary not found on PATH. Install it (e.g. "
            "`apt install plastimatch`, `conda install -c conda-forge "
            "plastimatch`, or see https://plastimatch.org) to use "
            "backend='plastimatch'."
        )


def _geometric_center(image_path: Path) -> Tuple[float, float, float]:
    """World-coordinate center of ``image_path``, read from its header only.

    Same definition SimpleITK's ``CenteredTransformInitializer`` uses for the
    sitk backend: the physical location of the (size - 1) / 2 continuous
    index, which is correct even under non-identity direction cosines.
    """
    import numpy as np
    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(image_path))
    reader.ReadImageInformation()
    # ImageFileReader only exposes the raw header fields (no pixel buffer
    # read), so the origin/direction/spacing -> physical-point mapping is
    # applied by hand: physical = origin + direction_matrix @ (index * spacing).
    size = np.asarray(reader.GetSize(), dtype=float)
    spacing = np.asarray(reader.GetSpacing(), dtype=float)
    origin = np.asarray(reader.GetOrigin(), dtype=float)
    direction = np.asarray(reader.GetDirection(), dtype=float).reshape(3, 3)
    center_index = (size - 1) / 2.0
    center = origin + direction @ (center_index * spacing)
    return (float(center[0]), float(center[1]), float(center[2]))


def _write_shifted_copy(image_path: Path, shift: Tuple[float, ...], dest_path: Path) -> None:
    """Write a copy of ``image_path`` with its origin translated by ``shift``.

    A pure coordinate-frame relabeling -- pixel data and spacing/direction
    are untouched, so this loses no information and needs no interpolation.
    """
    import SimpleITK as sitk

    image = sitk.ReadImage(str(image_path))
    new_origin = tuple(o + s for o, s in zip(image.GetOrigin(), shift))
    image.SetOrigin(new_origin)
    sitk.WriteImage(image, str(dest_path))


def _shift_image_origin_inplace(image_path: Path, shift: Tuple[float, ...]) -> None:
    """Translate ``image_path``'s own origin by ``shift`` (metadata-only)."""
    import SimpleITK as sitk

    image = sitk.ReadImage(str(image_path))
    new_origin = tuple(o + s for o, s in zip(image.GetOrigin(), shift))
    image.SetOrigin(new_origin)
    sitk.WriteImage(image, str(image_path))


def _shift_linear_transform_inplace(transform_path: Path, shift: Tuple[float, ...]) -> None:
    """Move a linear (rigid/affine) ITK transform file's rotation center by ``shift``.

    The transform was computed with images translated by ``shift`` from
    their true position, so its rotation center (near the world origin, by
    construction) is off from the true center by exactly ``shift``. Moving
    the center back is an exact correction -- the rotation and translation
    parameters themselves are unchanged, per the standard
    ``T(x) = R(x - c) + c + t`` parameterization.
    """
    import SimpleITK as sitk

    tx = sitk.ReadTransform(str(transform_path))
    center = tx.GetFixedParameters()
    corrected_center = tuple(c - s for c, s in zip(center, shift))
    tx.SetFixedParameters(corrected_center)
    sitk.WriteTransform(tx, str(transform_path))


_IMG_ORIGIN_RE = re.compile(r"img_origin = ([\-\d\.eE]+) ([\-\d\.eE]+) ([\-\d\.eE]+)")


def _shift_bspline_transform_inplace(transform_path: Path, shift: Tuple[float, ...]) -> None:
    """Move a native plastimatch B-spline ("MGH_GPUIT_BSP") transform's
    ``img_origin`` header field by ``shift`` -- the deformable-transform
    analogue of :func:`_shift_linear_transform_inplace`: it anchors the
    coefficient grid to world space the same way an image origin does, so
    the correction is the same metadata-only shift.
    """
    text = transform_path.read_text()
    match = _IMG_ORIGIN_RE.search(text)
    if match is None:
        raise RuntimeError(
            f"Expected an 'img_origin = x y z' header line in {transform_path} "
            "(plastimatch native B-spline transform format) but did not find one."
        )
    origin = tuple(float(match.group(i)) for i in (1, 2, 3))
    corrected = tuple(o - s for o, s in zip(origin, shift))
    replacement = "img_origin = %.6f %.6f %.6f" % corrected
    transform_path.write_text(text[: match.start()] + replacement + text[match.end() :])


def _check_for_silent_failure(log: str) -> None:
    hit = next((marker for marker in _SILENT_FAILURE_MARKERS if marker in log), None)
    if hit is not None:
        raise RuntimeError(
            "plastimatch exited 0 but its own log reports a registration "
            f"failure ({hit!r}) -- it silently fell back to (or never left) "
            "the identity transform. Log:\n" + log
        )


def register_pair_plastimatch(
    fixed_path: Union[str, Path],
    moving_path: Union[str, Path],
    *,
    output_image: Union[str, Path],
    output_transform: Union[str, Path],
    stages: Sequence[str] = _DEFAULT_STAGES,
    stage_overrides: Optional[Dict[str, Dict[str, str]]] = None,
    auto_center: bool = True,
) -> PlastimatchResult:
    """Register ``moving`` onto ``fixed`` with plastimatch's multi-stage DIR engine.

    Args:
        fixed_path: Reference image (e.g. planning CT). With the default
            ``auto_center=True``, this must be a single-file format
            (mha/mhd/nii/nrrd) since the origin-shift step round-trips it
            through SimpleITK, which does not resolve a DICOM series
            directory; pass ``auto_center=False`` if you need to hand
            plastimatch a DICOM directory directly (accepting the
            world-origin rotation-center trap described in the module
            docstring as a tradeoff).
        moving_path: Image to align (e.g. on-treatment CBCT).
        output_image: Where the warped moving image is written, in
            ``fixed_path``'s original coordinate frame.
        output_transform: Where the final-stage transform is written, also
            corrected back into ``fixed_path``'s original coordinate frame
            (see module docstring). For a run ending in a ``bspline`` stage
            this is a native B-spline coefficient file; for a rigid/affine
            run it is a standard ITK linear transform. Either can be
            replayed with :func:`warp_with_plastimatch_transform`.
        stages: Ordered stage names, each one of ``"rigid"``, ``"affine"``,
            ``"bspline"``. Default ``("rigid", "affine", "bspline")`` --
            the standard linear-then-deformable progression. Pass e.g.
            ``("rigid", "affine")`` to stop before the deformable stage.
        stage_overrides: Optional ``{stage_name: {param: value}}`` to
            override/extend individual stage parameters (e.g. tighter
            ``grid_spac`` for a smaller ROI) without editing the defaults.
        auto_center: Translate both images so the fixed image's geometric
            center sits at the world origin before handing them to
            plastimatch, then translate the result back (see module
            docstring). Default ``True`` -- leave it on unless you are
            deliberately reproducing raw plastimatch behavior for
            comparison/teaching purposes; turning it off reintroduces the
            world-origin rotation-center trap for any image whose DICOM
            origin is far from (0, 0, 0), which is the common case for real
            CT/CBCT.

    Returns:
        :class:`PlastimatchResult` with the warped image path, transform
        path, the generated command file (kept alongside the transform for
        inspection/reuse), and plastimatch's combined stdout/stderr log.

    Raises:
        ValueError: An unknown stage name was passed.
        RuntimeError: The ``plastimatch`` binary is missing; the subprocess
            exited non-zero; it exited zero but did not produce the
            expected output; or it exited zero and produced output but its
            log shows the known silent-failure signature.
    """
    for stage in stages:
        if stage not in _STAGE_TEMPLATES:
            raise ValueError(
                f"Unknown plastimatch stage {stage!r}; expected one of "
                f"{sorted(_STAGE_TEMPLATES)}"
            )
    _require_plastimatch()

    fixed_path = Path(fixed_path)
    moving_path = Path(moving_path)
    output_image = Path(output_image)
    output_transform = Path(output_transform)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_transform.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="qr_register_plastimatch_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        if auto_center:
            shift = tuple(-c for c in _geometric_center(fixed_path))
            run_fixed_path = tmp_path / ("fixed" + fixed_path.suffix)
            run_moving_path = tmp_path / ("moving" + moving_path.suffix)
            _write_shifted_copy(fixed_path, shift, run_fixed_path)
            _write_shifted_copy(moving_path, shift, run_moving_path)
        else:
            shift = (0.0, 0.0, 0.0)
            run_fixed_path = fixed_path
            run_moving_path = moving_path

        overrides = stage_overrides or {}
        command_file = output_transform.with_name(output_transform.name + ".cmd.txt")
        lines = [
            "[GLOBAL]",
            f"fixed={run_fixed_path}",
            f"moving={run_moving_path}",
            f"img_out={output_image}",
            f"xform_out={output_transform}",
            "",
        ]
        for stage in stages:
            params = dict(_STAGE_TEMPLATES[stage])
            params.update(overrides.get(stage, {}))
            lines.append("[STAGE]")
            lines.extend(f"{key}={value}" for key, value in params.items())
            lines.append("")
        command_file.write_text("\n".join(lines))

        proc = subprocess.run(
            ["plastimatch", "register", str(command_file)],
            capture_output=True,
            text=True,
        )
        log = proc.stdout + proc.stderr
        if proc.returncode != 0:
            raise RuntimeError(f"plastimatch register failed (exit {proc.returncode}):\n{log}")
        if not output_image.exists():
            raise RuntimeError(
                f"plastimatch register exited 0 but did not write {output_image}:\n{log}"
            )
        if not output_transform.exists():
            raise RuntimeError(
                f"plastimatch register exited 0 but did not write {output_transform}:\n{log}"
            )
        _check_for_silent_failure(log)

        if auto_center:
            _shift_image_origin_inplace(output_image, tuple(-s for s in shift))
            if stages[-1] in _LINEAR_STAGES:
                _shift_linear_transform_inplace(output_transform, shift)
            else:
                _shift_bspline_transform_inplace(output_transform, shift)

    return PlastimatchResult(
        warped_image_path=output_image,
        transform_path=output_transform,
        command_file_path=command_file,
        stages=tuple(stages),
        log=log,
    )


def warp_with_plastimatch_transform(
    input_path: Union[str, Path],
    fixed_path: Union[str, Path],
    transform_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    interpolation: str = "linear",
) -> Path:
    """Resample ``input_path`` onto ``fixed_path``'s grid via a saved transform.

    Used to carry a label mask (``interpolation="nn"``) through the same
    transform produced by :func:`register_pair_plastimatch`, or to replay a
    transform saved by an earlier registration run. ``transform_path`` is
    expected to already be in ``fixed_path``'s coordinate frame -- exactly
    what :func:`register_pair_plastimatch` writes to ``output_transform``,
    with no further shift bookkeeping required here.
    """
    if interpolation not in ("linear", "nn"):
        raise ValueError(f"interpolation must be 'linear' or 'nn', got {interpolation!r}")
    _require_plastimatch()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [
            "plastimatch",
            "warp",
            "--input",
            str(input_path),
            "--fixed",
            str(fixed_path),
            "--xf",
            str(transform_path),
            "--interpolation",
            interpolation,
            "--output-img",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"plastimatch warp failed (exit {proc.returncode}):\n{log}")
    if not output_path.exists():
        raise RuntimeError(f"plastimatch warp exited 0 but did not write {output_path}:\n{log}")
    return output_path
