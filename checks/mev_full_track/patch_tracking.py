"""Log each primary's exit state from TestEm5 TrackingAction.cc (issue #439).

Apply on top of ``../full_track_bremslib/geant4-testem5-emission.patch``. The
primary (track 1) prints one line when its track ends::

    EXIT <event> <T|R|A> <kinetic energy eV> <ux> <uy> <uz>

``T``/``R`` use TestEm5's own transmit/reflect tests; ``A`` is absorbed (the
primary stopped inside the absorber). Run: ``python3 patch_tracking.py
TestEm5/src/TrackingAction.cc``.
"""

import sys

path = sys.argv[1]
s = open(path).read()
include = '#include "G4Track.hh"\n'
anchor = "  G4bool notabsor = (transmit || reflect);\n"
assert s.count(include) == 1 and s.count(anchor) == 1, "unexpected TrackingAction.cc"
s = s.replace(
    include,
    include + '#include "G4Event.hh"\n#include "G4EventManager.hh"\n\n#include <iomanip>\n',
)
s = s.replace(
    anchor,
    anchor
    + """
  // Issue #439: per-primary exit state for unbinned PyRITE comparison.
  if (aTrack->GetTrackID() == 1) {
    const G4int event = G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
    const G4ThreeVector u = aTrack->GetMomentumDirection();
    G4cout << std::setprecision(17) << "EXIT " << event << ' '
           << (transmit ? 'T' : (reflect ? 'R' : 'A')) << ' ' << energy / eV << ' ' << u.x()
           << ' ' << u.y() << ' ' << u.z() << std::setprecision(6) << G4endl;
  }
""",
)
open(path, "w").write(s)
