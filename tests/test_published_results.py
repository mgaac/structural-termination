import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads((ROOT / "results/reference/expected.json").read_text())
MANIFEST = json.loads((ROOT / "artifacts/reference/manifest.json").read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_reference_bundle_hashes_match_manifest() -> None:
    for relative_path, expected_hash in MANIFEST["sha256"].items():
        path = ROOT / relative_path
        assert path.exists(), relative_path
        assert sha256(path) == expected_hash


def test_displayed_reference_metrics_are_fixed() -> None:
    evaluation = EXPECTED["evaluation"]
    assert evaluation["checkpoint_step"] == 500
    assert evaluation["num_graphs"] == 20
    assert evaluation["rollout_mode"] == "teacher_forced"
    assert evaluation["acc/bf_distance"] == 0.8187718391418457
    assert evaluation["acc/bf_predecessor"] == 0.8206279873847961
    assert evaluation["acc/bf_termination"] == 0.8092063069343567
    assert evaluation["acc/bfs_state"] == 1.0
    assert evaluation["acc/bfs_termination"] == 0.6316666603088379


def test_threshold_sweep_has_aligned_series() -> None:
    sweep = EXPECTED["threshold_sweep"]
    lengths = {
        len(sweep["thresholds"]),
        len(sweep["acc_bfs_termination"]),
        len(sweep["acc_bf_termination"]),
    }
    assert lengths == {61}
    assert sweep["thresholds"][0] == 74.0
    assert sweep["thresholds"][-1] == 80.0


def test_convergence_claim_preserves_changing_cohorts() -> None:
    convergence = EXPECTED["latent_convergence"]
    assert convergence["mean"][0] == 33.92076553961243
    assert convergence["mean"][-1] == 2.43073770403862
    assert all(
        current > following
        for current, following in zip(convergence["mean"], convergence["mean"][1:])
    )
    assert convergence["counts"][0] == 99.0
    assert convergence["counts"][-1] == 4.0


def test_unrecoverable_outputs_are_explicitly_archived() -> None:
    archive_note = (
        ROOT / "results/archive_historical_single_seed/README.md"
    ).read_text()
    assert "cannot be independently regenerated" in archive_note
    assert "not verified results" in archive_note


def test_displayed_figures_exist() -> None:
    assert (ROOT / "figures/test_threshold_vs_termination_accuracy.png").is_file()
    assert (ROOT / "figures/latent_convergence.png").is_file()


def test_reference_records_use_portable_paths() -> None:
    for path in (
        ROOT / "results/reference/test_threshold_sweep.json",
        ROOT / "results/reference/latent_convergence_stats.json",
    ):
        assert "/Users/" not in path.read_text()
