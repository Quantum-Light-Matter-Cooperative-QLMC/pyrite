"""Add photon-only stacking and escaped-photon records to TestEm5 (issue #341).

Apply on top of ``../full_track_bremslib/geant4-testem5-emission.patch`` and
``../mev_full_track/patch_tracking.py``. Two changes:

- ``StackingAction.cc``/``.hh``: ``/testem/stack/killSecondaries 3`` kills
  every charged secondary at creation without depositing its energy and keeps
  all photons (the switch becomes ``G4int``; upstream ``G4bool`` maps every
  positive value to 1);
- ``TrackingAction.cc``: every photon whose track leaves the world prints::

      GAM <event> <E eV> <x> <y> <z mm> <ux> <uy> <uz> <creator> <direct>

  ``creator`` is the creating process name; ``direct`` is 1 when the photon's
  energy and direction equal its creation values (no Compton or Rayleigh
  interaction), else 0. The world is vacuum, so the exit point and direction
  define the straight line to any external aperture.

Run: ``python3 patch_testem5.py TestEm5/src``.
"""

import sys
from pathlib import Path

src = Path(sys.argv[1])

stacking = src / "StackingAction.cc"
s = stacking.read_text()
anchor = """  if (0 < fKillSecondary) {
    if (fKillSecondary == 1) {"""
assert s.count(anchor) == 1, "unexpected StackingAction.cc"
s = s.replace(
    anchor,
    """  // Issue #341: photon-only secondaries; charged daughters vanish undeposited.
  if (fKillSecondary == 3) {
    return (charge != 0.) ? fKill : fUrgent;
  }
"""
    + anchor,
)
stacking.write_text(s)
# Upstream stores the switch as G4bool, collapsing every positive value to 1.
header = src.parent / "include" / "StackingAction.hh"
h = header.read_text()
assert h.count("G4bool fKillSecondary = false;") == 1, "unexpected StackingAction.hh"
header.write_text(h.replace("G4bool fKillSecondary = false;", "G4int fKillSecondary = 0;"))

tracking = src / "TrackingAction.cc"
s = tracking.read_text()
include = '#include "G4Track.hh"\n'
anchor = "  // transmitted + reflected particles counter\n"
assert s.count(include) == 1 and s.count(anchor) == 1, "unexpected TrackingAction.cc"
assert "G4EventManager.hh" in s, "apply mev_full_track/patch_tracking.py first"
s = s.replace(include, include + '#include "G4Gamma.hh"\n#include "G4VProcess.hh"\n', 1)
s = s.replace(
    anchor,
    """  // Issue #341: every photon leaving the (vacuum) world.
  if (aTrack->GetDefinition() == G4Gamma::Gamma()
      && aTrack->GetStep()->GetPostStepPoint()->GetStepStatus() == fWorldBoundary)
  {
    const G4int event = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
    const G4ThreeVector u = aTrack->GetMomentumDirection();
    const G4VProcess* creator = aTrack->GetCreatorProcess();
    const G4bool direct = energy == aTrack->GetVertexKineticEnergy()
                          && u == aTrack->GetVertexMomentumDirection();
    G4cout << std::setprecision(12) << "GAM " << event << ' ' << energy / eV << ' '
           << position.x() / mm << ' ' << position.y() / mm << ' ' << position.z() / mm << ' '
           << u.x() << ' ' << u.y() << ' ' << u.z() << ' '
           << (creator ? creator->GetProcessName() : G4String("primary")) << ' ' << direct
           << std::setprecision(6) << G4endl;
  }

"""
    + anchor,
)
tracking.write_text(s)
