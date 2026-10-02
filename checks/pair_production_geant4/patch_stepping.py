"""Insert pair-conversion recording into TestEm5 SteppingAction.cc (issue #275)."""

import sys

path = sys.argv[1]
s = open(path).read()
s = s.replace(
    '#include "G4Step.hh"\n',
    """#include "G4Step.hh"
#include "G4Event.hh"
#include "G4RunManager.hh"
#include "G4VProcess.hh"

#include <cstdlib>
#include <fstream>
#include <string>

namespace
{
// One line per conversion of the primary photon: event, photon kinetic energy
// [eV] and direction before the step, then pdg, kinetic energy [eV] and
// direction of every secondary created in that step (issue #275).
std::ofstream& PairOut()
{
  static std::ofstream out = [] {
    const char* path = std::getenv("PAIR_OUT");
    std::ofstream stream(path ? path : "pairs.txt");
    stream.precision(12);
    return stream;
  }();
  return out;
}
}  // namespace
""",
    1,
)
s = s.replace(
    """void SteppingAction::UserSteppingAction(const G4Step* aStep)
{
""",
    """void SteppingAction::UserSteppingAction(const G4Step* aStep)
{
  if (aStep->GetTrack()->GetTrackID() == 1) {
    const G4VProcess* process = aStep->GetPostStepPoint()->GetProcessDefinedStep();
    if (process != nullptr && process->GetProcessName() == "conv") {
      const G4StepPoint* pre = aStep->GetPreStepPoint();
      const G4ThreeVector d = pre->GetMomentumDirection();
      std::ofstream& out = PairOut();
      out << G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID() << ' '
          << pre->GetKineticEnergy() / CLHEP::eV << ' ' << d.x() << ' ' << d.y() << ' ' << d.z();
      for (const G4Track* secondary : *aStep->GetSecondaryInCurrentStep()) {
        const G4ThreeVector v = secondary->GetMomentumDirection();
        out << ' ' << secondary->GetDefinition()->GetPDGEncoding() << ' '
            << secondary->GetKineticEnergy() / CLHEP::eV << ' ' << v.x() << ' ' << v.y() << ' '
            << v.z();
      }
      out << '\\n';
    }
  }
""",
    1,
)
assert s.count("PairOut()") == 2
open(path, "w").write(s)
