"""Patch an isolated copy of the pinned issue-182 emission TestEm5 source.

Input must already have geant4-testem5-emission.patch applied. Rejects unknown
source hooks. All secondaries are killed at creation; primary photons are
logged before stacking. Primary transport terminates at the first slab exit.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    stepping = args.source / "src/SteppingAction.cc"
    stacking = args.source / "src/StackingAction.cc"
    before = {
        str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [stepping, stacking]
    }
    expected = {
        "src/SteppingAction.cc": "a3b60d98e78508ac26cc0eab92eaf9a0c645ea66be01e1051b14c89ec7aaee00",
        "src/StackingAction.cc": "b47574a7e0fa3cc19f3d0185a762b37fa333554b33e274c6f8c7c763bcf47829",
    }
    if before != expected:
        raise ValueError("source does not match the pinned issue-182 emission patch")
    detector = args.source / "src/DetectorConstruction.cc"
    material_text = detector.read_text()
    for old, new in [
        (
            "a = 28.09 * g / mole, density = 2.330 * g / cm3",
            "a = 28.085 * g / mole, density = 2.3291534103942357 * g / cm3",
        ),
        (
            "a = 183.85 * g / mole, density = 19.30 * g / cm3",
            "a = 183.84 * g / mole, density = 19.30 * g / cm3",
        ),
    ]:
        if material_text.count(old) != 1:
            raise ValueError("unexpected Geant4 material definition")
        material_text = material_text.replace(old, new)
    before["src/DetectorConstruction.cc"] = hashlib.sha256(detector.read_bytes()).hexdigest()
    detector.write_text(material_text)
    text = stepping.read_text()
    hook = "  const G4VProcess* process = aStep->GetPostStepPoint()->GetProcessDefinedStep();"
    if text.count(hook) != 1 or '"EBREM "' not in text:
        raise ValueError("expected pinned issue-182 emission scoring source")
    text = text.replace(
        hook,
        """  const auto* post = aStep->GetPostStepPoint();
  const auto position = post->GetPosition();
  const bool exit = post->GetStepStatus() == fGeomBoundary;
  const bool stopped = aStep->GetTrack()->GetTrackStatus() == fStopAndKill;
  if (exit || stopped) {
    int terminal = 7;
    if (exit) {
      const double tolerance = 1e-6 * mm;
      if (std::abs(position.x() - fDetector->GetxendAbs()) < tolerance) terminal = 5;
      else if (std::abs(position.x() - fDetector->GetxstartAbs()) < tolerance) terminal = 4;
      else terminal = 6;
    }
    const auto event = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
    G4cout << std::setprecision(17) << "PRIMARY_END " << event << ' ' << terminal
           << ' ' << post->GetKineticEnergy() / keV << G4endl;
    aStep->GetTrack()->SetTrackStatus(fStopAndKill);
  }
"""
        + hook,
    )
    # Histogram track-length scoring is retained and charged to this diagnostic adapter.
    stacking_text = stacking.read_text()
    hook2 = "{\n"
    start = stacking_text.index("G4ClassificationOfNewTrack StackingAction::ClassifyNewTrack")
    brace = stacking_text.index(hook2, start) + len(hook2)
    stacking_text = (
        stacking_text[:brace]
        + "  if (aTrack->GetParentID() > 0) return fKill;\n"
        + stacking_text[brace:]
    )
    stepping.write_text(text)
    stacking.write_text(stacking_text)
    after = {
        str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [stepping, stacking, detector]
    }
    (args.source / "benchmark-patch.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
