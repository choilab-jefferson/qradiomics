"""Tests for qradiomics.data_model — hierarchical Cohort + manifest contract.

Schema (mirrored from qradiomics-dev, JLR-aligned):
  Cohort.patients : list[Patient]
  Patient.id, Patient.studies (list), Patient.treatment_courses (list)
  TreatmentCourse.id, TreatmentCourse.studies (list)
  Study.id, Study.image_series (list), Study.rt_structures (list)
  ImageSeries.id, ImageSeries.file_paths (list)
  RTStructureSet.id, RTStructureSet.file_path, RTStructureSet.rois (list)
  ROI.name, ROI.label, ROI.roi_type
"""
from pathlib import Path

import pytest

from qradiomics.data_model import (
    Cohort,
    ImageSeries,
    Modality,
    Patient,
    ROI,
    RTStructureSet,
    Study,
    StudyType,
    TreatmentCourse,
    load_cohort,
    save_cohort,
)
from qradiomics.manifest import MANIFEST_COLUMNS, flatten_cohort, read_manifest, write_manifest


def _make_diagnostic_cohort() -> Cohort:
    img = ImageSeries(
        id="img_preCT_001",
        modality=Modality.CT,
        file_paths=["/x/preCT.nrrd"],
        image_tag="preCT",
    )
    rt = RTStructureSet(
        id="rt_001",
        file_path="/x/GTV-label.nrrd",
        references_image_series_id="img_preCT_001",
        rois=[ROI(name="GTV", label=1, roi_type="GTV")],
    )
    s = Study(
        id="s_diag",
        study_type=StudyType.DIAGNOSTIC,
        image_series=[img],
        rt_structures=[rt],
    )
    p = Patient(id="P1", studies=[s])
    return Cohort(id="dx", name="diagnostic", patients=[p])


def _make_longitudinal_cohort() -> Cohort:
    studies = []
    for tp, day, st in [("baseline", 0, StudyType.PLANNING_CT),
                        ("week4", 28, StudyType.ON_TREATMENT_CBCT)]:
        img = ImageSeries(
            id=f"img_{tp}",
            modality=Modality.CT if "baseline" == tp else Modality.CBCT,
            file_paths=[f"/x/{tp}.nrrd"],
            image_tag=tp,
        )
        rt = RTStructureSet(
            id=f"rt_{tp}", file_path=f"/x/{tp}-mask.nrrd",
            references_image_series_id=f"img_{tp}",
            rois=[ROI(name="GTV", label=1, roi_type="GTV")],
        )
        studies.append(Study(id=f"s_{tp}", study_type=st,
                              relative_day=day,
                              image_series=[img], rt_structures=[rt]))
    course = TreatmentCourse(id="rt1", studies=studies,
                              prescription_dose_cgy=6000.0, num_fractions=30)
    p = Patient(id="P1", treatment_courses=[course])
    return Cohort(id="long", name="longitudinal", patients=[p])


class TestManifestSchema:
    def test_columns_canonical(self):
        # mirror what JLR consumes — list of expected names
        assert "patient_id" in MANIFEST_COLUMNS
        assert "image_path" in MANIFEST_COLUMNS
        assert "mask_path" in MANIFEST_COLUMNS
        assert "mask_label" in MANIFEST_COLUMNS

    def test_columns_order_stable(self):
        # first three columns are the minimum required keys
        assert MANIFEST_COLUMNS[0] == "patient_id"
        assert MANIFEST_COLUMNS[1] == "course_id"
        assert MANIFEST_COLUMNS[2] == "study_id"


class TestFlattenCohort:
    def test_empty_cohort(self):
        assert flatten_cohort(Cohort(id="z", name="empty")) == []

    def test_diagnostic_single_row(self):
        rows = flatten_cohort(_make_diagnostic_cohort())
        assert len(rows) == 1
        r = rows[0]
        assert r["patient_id"] == "P1"
        assert r.get("course_id") in (None, "", "n/a")
        assert r["image_path"] == "/x/preCT.nrrd"
        assert r["mask_path"] == "/x/GTV-label.nrrd"

    def test_longitudinal_two_rows_with_course(self):
        rows = flatten_cohort(_make_longitudinal_cohort())
        assert len(rows) == 2
        assert all(r["course_id"] == "rt1" for r in rows)
        # flatten_cohort derives the timepoint label from relative_day
        # (Study.relative_day=0 → 'day+0', 28 → 'day+28').
        tps = sorted(r["timepoint"] for r in rows)
        assert tps == ["day+0", "day+28"]


class TestManifestRoundTrip:
    def test_diagnostic(self, tmp_path):
        cohort = _make_diagnostic_cohort()
        rows = flatten_cohort(cohort)
        path = tmp_path / "m.csv"
        write_manifest(rows, path)
        again = read_manifest(path)
        assert len(again) == len(rows)
        assert again[0]["patient_id"] == rows[0]["patient_id"]
        assert again[0]["image_path"] == rows[0]["image_path"]

    def test_longitudinal(self, tmp_path):
        cohort = _make_longitudinal_cohort()
        rows = flatten_cohort(cohort)
        path = tmp_path / "m.csv"
        write_manifest(rows, path)
        again = read_manifest(path)
        assert len(again) == 2


class TestCohortPersistence:
    @pytest.mark.parametrize("suffix", [".yaml", ".json"])
    def test_round_trip(self, tmp_path, suffix):
        c = _make_longitudinal_cohort()
        path = tmp_path / f"cohort{suffix}"
        save_cohort(c, path)
        c2 = load_cohort(path)
        assert c2.id == c.id
        assert len(c2.patients) == len(c.patients)
        # treatment course preserved
        course = c2.patients[0].treatment_courses[0]
        assert course.id == "rt1"
        assert course.prescription_dose_cgy == 6000.0
        assert course.num_fractions == 30
        assert len(course.studies) == 2

    def test_modality_enum_preserved(self, tmp_path):
        c = _make_longitudinal_cohort()
        save_cohort(c, tmp_path / "c.yaml")
        c2 = load_cohort(tmp_path / "c.yaml")
        modalities = {
            s.image_series[0].modality
            for s in c2.patients[0].treatment_courses[0].studies
        }
        assert Modality.CBCT in modalities
        assert Modality.CT in modalities

    def test_study_type_enum_preserved(self, tmp_path):
        c = _make_longitudinal_cohort()
        save_cohort(c, tmp_path / "c.yaml")
        c2 = load_cohort(tmp_path / "c.yaml")
        study_types = {s.study_type
                       for s in c2.patients[0].treatment_courses[0].studies}
        assert StudyType.PLANNING_CT in study_types
        assert StudyType.ON_TREATMENT_CBCT in study_types
