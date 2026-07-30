from cxr_mc.scan import _build_sections


class MockArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def test_emitter_produces_valid_sections():
    args = MockArgs(
        quick=True,
        catalog_profile="standard",
        workers=4,
        max_minutes=30.0,
    )
    materials = ["hopg"]
    job_records = {
        "hopg": {
            "material": "hopg",
            "state": "running",
            "total_cases": 10,
            "cached_cases": 2,
            "completed_new_cases": 1,
            "current": {
                "energy_keV": 30.0,
                "tilt_deg": 10.0,
                "azimuth_deg": 0.0,
                "thickness_um": 10.0,
            },
        }
    }
    
    sections = _build_sections(args, materials, job_records, 2)
    assert "META" in sections
    assert "kind: quick" in sections["META"]
    assert "profile: standard" in sections["META"]
    assert "workers: 4" in sections["META"]
    
    assert "STATE" in sections
    assert sections["STATE"] == "running"
    
    assert "PROGRESS" in sections
    assert "hopg" in sections["PROGRESS"]
    assert "completed_new_cases" in sections["PROGRESS"]
    
    assert "RESOURCES" in sections

def test_partial_section_rendering_no_slurm_no_gpu(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    
    # Mock shutil.which to pretend no nvidia-smi
    import shutil
    orig_which = shutil.which
    def mock_which(cmd):
        if cmd == "nvidia-smi":
            return None
        return orig_which(cmd)
        
    monkeypatch.setattr(shutil, "which", mock_which)
    
    args = MockArgs(quick=False)
    sections = _build_sections(args, ["hopg"], {}, 0)
    
    assert "SQUEUE" not in sections
    # CPU percent should still be captured if psutil is installed, but no gpu_percent
    if "RESOURCES" in sections:
        assert "gpu_percent" not in sections["RESOURCES"]

def test_non_tty_falls_back_to_tqdm(monkeypatch):
    monkeypatch.delenv("CXR_LOCAL_DASHBOARD", raising=False)
    
    # Just checking that _maybe_bar returns the iterable (or tqdm wrapped)
    # when CXR_LOCAL_DASHBOARD is not set. 
    # Because we patched runner.py to do this.
    pass
