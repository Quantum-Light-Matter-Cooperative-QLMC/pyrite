// Geant4 G4Cerenkov reference for the bulk Frank--Tamm source (issue #587).
//
// One charged primary crosses a slab with a tabulated RINDEX. For every step
// of the primary inside the slab we record the step-mean beta, the step
// length and the Cerenkov photons created in that step; photon energy and
// polar angle about the step direction are histogrammed. All secondaries are
// killed at creation, so only the primary radiates.
//
// usage: cherenkov MATERIAL RINDEX_FILE PARTICLE KINETIC_MEV THICK_MM EVENTS
//                  SEED EMIN_EV EMAX_EV N_EBINS OUT
// RINDEX_FILE: two whitespace columns, photon energy [eV] and n, ascending.

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "G4Box.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4LogicalVolume.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4NistManager.hh"
#include "G4OpticalParameters.hh"
#include "G4OpticalPhoton.hh"
#include "G4OpticalPhysics.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4UserStackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VProcess.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "Randomize.hh"

namespace {

struct Config {
  std::string material, rindex, particle, out;
  double kinetic_MeV, thick_mm, emin_eV, emax_eV;
  long events, seed;
  int n_ebins;
};

G4LogicalVolume* gRadiator = nullptr;

class Detector : public G4VUserDetectorConstruction {
 public:
  explicit Detector(const Config& c) : fC(c) {}
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    auto* vacuum = nist->FindOrBuildMaterial("G4_Galactic");
    auto* base = nist->FindOrBuildMaterial(fC.material);
    auto* radiator = new G4Material("Radiator", base->GetDensity(), base);
    std::vector<G4double> energy, index;
    std::ifstream in(fC.rindex);
    for (double e, n; in >> e >> n;) {
      energy.push_back(e * eV);
      index.push_back(n);
    }
    auto* mpt = new G4MaterialPropertiesTable();
    mpt->AddProperty("RINDEX", energy, index);
    radiator->SetMaterialPropertiesTable(mpt);

    const double half = 0.5 * fC.thick_mm * mm;
    auto* world = new G4LogicalVolume(new G4Box("World", 1 * m, 1 * m, half + 10 * cm),
                                      vacuum, "World");
    gRadiator = new G4LogicalVolume(new G4Box("Slab", 1 * m, 1 * m, half), radiator, "Slab");
    new G4PVPlacement(nullptr, {}, gRadiator, "Slab", world, false, 0);
    return new G4PVPlacement(nullptr, {}, world, "World", nullptr, false, 0);
  }

 private:
  Config fC;
};

class Gun : public G4VUserPrimaryGeneratorAction {
 public:
  explicit Gun(const Config& c) : fGun(1) {
    fGun.SetParticleDefinition(G4ParticleTable::GetParticleTable()->FindParticle(c.particle));
    fGun.SetParticleEnergy(c.kinetic_MeV * MeV);
    fGun.SetParticlePosition({0, 0, -(0.5 * c.thick_mm * mm + 5 * cm)});
    fGun.SetParticleMomentumDirection({0, 0, 1});
  }
  void GeneratePrimaries(G4Event* event) override { fGun.GeneratePrimaryVertex(event); }

 private:
  G4ParticleGun fGun;
};

class KillSecondaries : public G4UserStackingAction {
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* track) override {
    return track->GetParentID() > 0 ? fKill : fUrgent;
  }
};

class Recorder : public G4UserSteppingAction {
 public:
  explicit Recorder(const Config& c)
      : fC(c), fSteps(c.out + ".steps"), fEnergy(c.n_ebins, 0), fTheta(900, 0) {
    fSteps << std::setprecision(17);
  }
  ~Recorder() override {
    std::ofstream h(fC.out + ".hist");
    h << "energy_edges " << fC.emin_eV << ' ' << fC.emax_eV << ' ' << fC.n_ebins << '\n';
    h << "energy_counts";
    for (auto v : fEnergy) h << ' ' << v;
    h << "\ntheta_edges_deg 0 90 900\ntheta_counts";
    for (auto v : fTheta) h << ' ' << v;
    h << "\nphotons " << fPhotons << "\nangle_outside " << fOutside << '\n';
  }
  void UserSteppingAction(const G4Step* step) override {
    if (step->GetTrack()->GetTrackID() != 1) return;
    const auto* pre = step->GetPreStepPoint();
    if (pre->GetPhysicalVolume() == nullptr ||
        pre->GetPhysicalVolume()->GetLogicalVolume() != gRadiator)
      return;
    const auto* post = step->GetPostStepPoint();
    const G4ThreeVector dir = (post->GetPosition() - pre->GetPosition()).unit();
    long n = 0;
    for (const G4Track* t : *step->GetSecondaryInCurrentStep()) {
      if (t->GetDefinition() != G4OpticalPhoton::Definition()) continue;
      if (t->GetCreatorProcess()->GetProcessName() != "Cerenkov") continue;
      ++n;
      const double e = t->GetKineticEnergy() / eV;
      const int ie = int((e - fC.emin_eV) / (fC.emax_eV - fC.emin_eV) * fC.n_ebins);
      if (ie >= 0 && ie < fC.n_ebins) ++fEnergy[ie];
      const double cosv = std::fmin(1.0, std::fmax(-1.0, dir.dot(t->GetMomentumDirection())));
      const int it = int(std::acos(cosv) / deg * 10.0);
      if (it >= 0 && it < 900) ++fTheta[it]; else ++fOutside;
    }
    fPhotons += n;
    fSteps << 0.5 * (pre->GetBeta() + post->GetBeta()) << ' ' << step->GetStepLength() / mm
           << ' ' << n << '\n';
  }

 private:
  Config fC;
  std::ofstream fSteps;
  std::vector<long> fEnergy, fTheta;
  long fPhotons = 0, fOutside = 0;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 12) {
    G4cerr << "usage: cherenkov MATERIAL RINDEX PARTICLE T_MEV THICK_MM EVENTS SEED "
              "EMIN_EV EMAX_EV N_EBINS OUT\n";
    return 2;
  }
  Config c{argv[1], argv[2], argv[3], argv[11], std::atof(argv[4]), std::atof(argv[5]),
           std::atof(argv[8]), std::atof(argv[9]), std::atol(argv[6]), std::atol(argv[7]),
           std::atoi(argv[10])};
  G4Random::setTheSeed(c.seed);
  auto* run = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
  run->SetUserInitialization(new Detector(c));
  auto* physics = new G4VModularPhysicsList();
  physics->RegisterPhysics(new G4EmStandardPhysics_option4());
  physics->RegisterPhysics(new G4OpticalPhysics());
  run->SetUserInitialization(physics);
  auto* optical = G4OpticalParameters::Instance();
  for (const char* p : {"Scintillation", "OpAbsorption", "OpRayleigh", "OpMieHG", "OpBoundary",
                        "OpWLS", "OpWLS2"})
    optical->SetProcessActivation(p, false);
  optical->SetCerenkovMaxPhotonsPerStep(300);
  optical->SetCerenkovMaxBetaChange(1.0);  // percent
  run->SetUserAction(new Gun(c));
  run->SetUserAction(new KillSecondaries());
  auto* recorder = new Recorder(c);
  run->SetUserAction(recorder);
  run->Initialize();
  run->BeamOn(int(c.events));
  delete run;  // deletes the recorder, flushing histograms
  return 0;
}
