"""Per-layout ground-truth manifests for the TCIA Phantom-FDA dataset.

Coordinate frame: **mm in the DICOM world frame** (LPS), the same frame
as ``ImagePositionPatient`` in the released DICOM headers. To project a
nodule centre into voxel indices for a specific reconstruction, take the
series' ``ImagePositionPatient`` (slice 0) as the origin and divide by
``PixelSpacing`` / ``SliceThickness``.

Reference volumes were derived from precision weighing (Ohaus Adventurer
Pro AV 2646, 0.1 mg tolerance, three replicates averaged) ÷ manufacturer
density — see ``wiki/data/PHANTOM_FDA.md`` §2.

Layout 4 (the qradiomics quantification target) is fully captured here.
Layouts 1-3 can be added later as needed; the structures are reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ShapeClass = Literal["SPH", "ELL", "LOB", "SPI", "IRR"]
Lung = Literal["L", "R"]


@dataclass(frozen=True)
class Nodule:
    """A single synthetic phantom nodule.

    Attributes
    ----------
    nodule_id:
        Stable per-layout integer (1-based) matching the TCIA per-layout
        document's row order.
    lung:
        ``"L"`` or ``"R"``.
    diameter_mm:
        Nominal (manufactured) diameter in mm.
    shape:
        Coarse shape class. Three-letter codes are the TCIA convention.
    hu:
        Manufactured CT density in Hounsfield Units.
    center_mm:
        ``(x, y, z)`` centre in DICOM world (LPS) coordinates.
    volume_mm3:
        Reference volume from the weighing protocol (§2).
    exclude:
        Set to ``True`` when the nodule has a known manufacturing defect
        and should be omitted from primary bias / variance metrics.
    notes:
        Free-text caveat, kept short.
    """

    nodule_id: int
    lung: Lung
    diameter_mm: float
    shape: ShapeClass
    hu: int
    center_mm: tuple[float, float, float]
    volume_mm3: float
    exclude: bool = False
    notes: str = ""


@dataclass(frozen=True)
class Layout:
    """A named nodule arrangement within the Phantom-FDA phantom."""

    layout_id: int
    name: str
    attachment: Literal["attached", "suspended"]
    nodules: tuple[Nodule, ...]

    def __len__(self) -> int:
        return len(self.nodules)

    def by_shape(self, shape: ShapeClass) -> tuple[Nodule, ...]:
        return tuple(n for n in self.nodules if n.shape == shape)

    def by_density(self, hu: int) -> tuple[Nodule, ...]:
        return tuple(n for n in self.nodules if n.hu == hu)

    def included(self) -> tuple[Nodule, ...]:
        """Subset usable for primary metrics (defect-free)."""
        return tuple(n for n in self.nodules if not n.exclude)


# ---------------------------------------------------------------------------
# Layout 4 — vessel-attached, 12 nodules
#   Source: TCIA Phantom-FDA per-layout document (see wiki/data/PHANTOM_FDA.md §4)
# ---------------------------------------------------------------------------
#
# Layout 4 spans the three non-spherical shape classes (ELL/LOB/SPI) at
# two diameters (10/20 mm) and two density bands (-630 HU ground-glass,
# +100 HU solid). The 10mm / +100HU ELL nodule (#2) has a known internal
# defect (Nodule 6 replacement planned) — marked exclude=True.

LAYOUT_4: Layout = Layout(
    layout_id=4,
    name="Layout 4",
    attachment="attached",
    nodules=(
        Nodule(1,  "L", 10.0, "ELL", -630, (176.0, 354.0, 178.0),  547.0),
        Nodule(2,  "R", 10.0, "ELL",  100, (341.0, 333.0, 162.0),  545.0,
               exclude=True,
               notes="manufacturing defect; replacement scheduled (Nodule 6)"),
        Nodule(3,  "L", 20.0, "ELL", -630, (169.0, 322.0,  95.0), 4210.0),
        Nodule(4,  "R", 20.0, "ELL",  100, (401.0, 296.0, 122.0), 4155.0),
        Nodule(5,  "L", 10.0, "LOB", -630, (159.0, 329.0, 347.0),  530.0),
        Nodule(6,  "R", 10.0, "LOB",  100, (395.0, 272.0, 329.0),  535.0),
        Nodule(7,  "L", 20.0, "LOB", -630, (136.0, 292.0, 337.0), 4305.0),
        Nodule(8,  "R", 20.0, "LOB",  100, (349.0, 350.0, 268.0), 4441.0),
        Nodule(9,  "L", 10.0, "SPI", -630, (167.0, 315.0, 520.0),  539.0),
        Nodule(10, "R", 10.0, "SPI",  100, (357.0, 296.0, 530.0),  535.0),
        Nodule(11, "L", 20.0, "SPI", -630, (133.0, 269.0, 475.0), 4335.0),
        Nodule(12, "R", 20.0, "SPI",  100, (386.0, 248.0, 503.0), 4305.0),
    ),
)


# Convenient by-id lookup. (Frozen dict idiom — read-only.)
LAYOUT_4_BY_ID: dict[int, Nodule] = {n.nodule_id: n for n in LAYOUT_4.nodules}
