// Matched-bounds Geant4 single-scattering benchmark for issue #821.
//
// One primary electron per event enters a 10 x 10 mm slab in vacuum. Physics is
// an electron-only constructor holding exactly the processes PyRITE's frozen
// transport models: eIoni (no delta rays: 1 m electron cut, no fluctuations),
// eBrem (Seltzer-Berger, photon cut exactly 1 keV) and DPWA single Coulomb
// scattering (scattering-power correction on, the counterpart of PyRITE's
// atomic-electron deflection). No ePairProd, PIXE, Auger or fluorescence.
// Every secondary is scored at creation and killed; the primary is killed on
// its first slab exit or below 10 keV.
//
// Per-primary records live in per-thread memory and are written once after the
// timed run, so no logging enters the event loop. main() times
// /run/initialize, an untimed-for-ranking warmup run (seed + 1000000) and the
// measured run separately with a steady clock.

#include "G4Box.hh"
#include "G4Electron.hh"
#include "G4EmParameters.hh"
#include "G4Event.hh"
#include "G4Gamma.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4PhysicsListHelper.hh"
#include "G4Positron.hh"
#include "G4ProcessManager.hh"
#include "G4ProductionCutsTable.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4UserEventAction.hh"
#include "G4UserRunAction.hh"
#include "G4UserStackingAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VPhysicsConstructor.hh"
#include "G4VProcess.hh"
#include "G4VUserActionInitialization.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4CoulombScattering.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eDPWACoulombScatteringModel.hh"
#include "G4eIonisation.hh"
#include "Randomize.hh"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace
{
struct Config
{
  std::string material = "Si";
  double energy_keV = 300.;
  double thickness_um = 100.;
  double angle_deg = 0.;
  long histories = 1000;
  long warmup = 1000;
  long seed = 1;
  int threads = 1;
  std::string output = "g4bench";
  // eIoni step function; <= 0 keeps the Geant4 default (0.2, 1 mm).
  double step_dr = 0.;
  double step_final_um = 0.;
};
Config gConfig;

// Fixed-width per-primary record; numpy dtype in geant4_bench.py.
#pragma pack(push, 1)
struct Record
{
  int32_t event;
  int32_t terminal;  // 4 back, 5 transmitted, 6 side, 7 stopped (PyRITE codes)
  double exit_keV;
  double path_um;
  int32_t photons[4];  // [1,10), [10,50), [50,100), [100,inf) keV
  double photon_keV_ge10;
  int32_t steps_coulomb;
  int32_t steps_ioni;
  int32_t steps_brem;
  int32_t steps_other;
};
#pragma pack(pop)

class Recorder
{
 public:
  std::vector<Record> records;
  Record current{};
  bool done = false;
  void Reset(int event)
  {
    std::memset(&current, 0, sizeof current);
    current.event = event;
    current.terminal = -1;
    done = false;
  }
};

class Detector : public G4VUserDetectorConstruction
{
 public:
  G4VPhysicalVolume* Construct() override
  {
    // Frozen #510 compositions: exact catalog Si number density, W 19.3 g/cm3.
    G4Material* slab = gConfig.material == "Si"
                         ? new G4Material("BenchSi", 14., 28.085 * g / mole,
                                          2.3291534103942357 * g / cm3)
                         : new G4Material("BenchW", 74., 183.84 * g / mole, 19.3 * g / cm3);
    G4Material* vacuum = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");
    const double half_z = 0.5 * gConfig.thickness_um * um;
    auto* world_box = new G4Box("World", 1. * cm, 1. * cm, half_z + 1. * mm);
    auto* world_lv = new G4LogicalVolume(world_box, vacuum, "World");
    auto* world = new G4PVPlacement(nullptr, {}, world_lv, "World", nullptr, false, 0);
    auto* slab_box = new G4Box("Slab", 5. * mm, 5. * mm, half_z);
    fSlabLV = new G4LogicalVolume(slab_box, slab, "Slab");
    fSlab = new G4PVPlacement(nullptr, {}, fSlabLV, "Slab", world_lv, false, 0);
    return world;
  }
  static G4VPhysicalVolume* fSlab;
  G4LogicalVolume* fSlabLV = nullptr;
};
G4VPhysicalVolume* Detector::fSlab = nullptr;

class ElectronOnlyPhysics : public G4VPhysicsConstructor
{
 public:
  ElectronOnlyPhysics() : G4VPhysicsConstructor("ElectronOnlySS") {}
  void ConstructParticle() override { G4Electron::Electron(); }
  void ConstructProcess() override
  {
    auto* ph = G4PhysicsListHelper::GetPhysicsListHelper();
    auto* e = G4Electron::Electron();
    ph->RegisterProcess(new G4eIonisation(), e);
    ph->RegisterProcess(new G4eBremsstrahlung(), e);
    // As G4EmStandardPhysicsSS with Mott correction: DPWA over all angles,
    // scattering-power correction on (constructor defaults).
    auto* ss = new G4CoulombScattering(false);
    ss->SetEmModel(new G4eDPWACoulombScatteringModel());
    ph->RegisterProcess(ss, e);
  }
};

class PhysicsList : public G4VModularPhysicsList
{
 public:
  PhysicsList()
  {
    auto* p = G4EmParameters::Instance();
    p->SetDefaults();
    p->SetLossFluctuations(false);
    p->SetFluo(false);
    p->SetAuger(false);
    p->SetPixe(false);
    p->SetLowestElectronEnergy(10. * keV);
    p->SetMscThetaLimit(0.0);
    p->SetUseMottCorrection(true);
    if (gConfig.step_dr > 0.) p->SetStepFunction(gConfig.step_dr, gConfig.step_final_um * um);
    RegisterPhysics(new ElectronOnlyPhysics());
  }
  void ConstructParticle() override
  {
    G4VModularPhysicsList::ConstructParticle();
    // Secondaries need definitions to be created (then killed at stacking).
    G4Gamma::Gamma();
    G4Positron::Positron();
  }
  void SetCuts() override
  {
    // Gamma cut: 1 nm range maps below the table's lower edge, so the cut is
    // exactly the 1 keV edge = PyRITE's hard-photon cutoff. Electron cut 1 m:
    // far above E0/2, so eIoni never emits a delta ray.
    G4ProductionCutsTable::GetProductionCutsTable()->SetEnergyRange(1. * keV, 100. * GeV);
    SetCutValue(1. * nm, "gamma");
    SetCutValue(1. * m, "e-");
    SetCutValue(1. * m, "e+");
    SetCutValue(1. * m, "proton");
  }
};

class Gun : public G4VUserPrimaryGeneratorAction
{
 public:
  Gun()
  {
    fGun.SetParticleDefinition(G4Electron::Electron());
    fGun.SetParticleEnergy(gConfig.energy_keV * keV);
    const double a = gConfig.angle_deg * deg;
    const double z0 = -0.5 * gConfig.thickness_um * um - 1. * nm;
    fGun.SetParticlePosition({-std::tan(a) * 1. * nm, 0., z0});
    fGun.SetParticleMomentumDirection({std::sin(a), 0., std::cos(a)});
  }
  void GeneratePrimaries(G4Event* event) override { fGun.GeneratePrimaryVertex(event); }

 private:
  G4ParticleGun fGun{1};
};

class Stepping : public G4UserSteppingAction
{
 public:
  explicit Stepping(Recorder* r) : fRec(r) {}
  void UserSteppingAction(const G4Step* step) override
  {
    auto* track = step->GetTrack();
    if (track->GetTrackID() != 1) return;
    const auto* pre = step->GetPreStepPoint();
    if (pre->GetPhysicalVolume() != Detector::fSlab) return;
    const auto* post = step->GetPostStepPoint();
    Record& r = fRec->current;
    r.path_um += step->GetStepLength() / um;
    const G4VProcess* proc = post->GetProcessDefinedStep();
    const G4String& name = proc ? proc->GetProcessName() : G4String("none");
    if (name == "CoulombScat") ++r.steps_coulomb;
    else if (name == "eIoni") ++r.steps_ioni;
    else if (name == "eBrem") ++r.steps_brem;
    else ++r.steps_other;
    for (const G4Track* s : *step->GetSecondaryInCurrentStep()) {
      if (s->GetParticleDefinition() != G4Gamma::Gamma()) continue;
      const double k = s->GetKineticEnergy() / keV;
      const int band = k < 10. ? 0 : k < 50. ? 1 : k < 100. ? 2 : 3;
      ++r.photons[band];
      if (k >= 10.) r.photon_keV_ge10 += k;
    }
    const bool exit = post->GetStepStatus() == fGeomBoundary;
    const bool stopped = track->GetTrackStatus() == fStopAndKill
                         || post->GetKineticEnergy() <= 0.;
    if (!(exit || stopped) || fRec->done) return;
    int terminal = 7;
    if (exit) {
      const double z = post->GetPosition().z();
      const double half = 0.5 * gConfig.thickness_um * um;
      const double tol = 1e-6 * mm;
      terminal = std::abs(z - half) < tol ? 5 : std::abs(z + half) < tol ? 4 : 6;
    }
    r.terminal = terminal;
    r.exit_keV = terminal == 7 ? 0. : post->GetKineticEnergy() / keV;
    fRec->done = true;
    track->SetTrackStatus(fStopAndKill);
  }

 private:
  Recorder* fRec;
};

class Stacking : public G4UserStackingAction
{
 public:
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* t) override
  {
    return t->GetParentID() == 0 ? fUrgent : fKill;
  }
};

class Events : public G4UserEventAction
{
 public:
  explicit Events(Recorder* r) : fRec(r) {}
  void BeginOfEventAction(const G4Event* e) override { fRec->Reset(e->GetEventID()); }
  void EndOfEventAction(const G4Event*) override { fRec->records.push_back(fRec->current); }

 private:
  Recorder* fRec;
};

std::mutex gWriteMutex;

class Runs : public G4UserRunAction
{
 public:
  explicit Runs(Recorder* r) : fRec(r) {}
  void BeginOfRunAction(const G4Run*) override
  {
    if (!fRec) return;
    fRec->records.clear();
    fRec->records.reserve(static_cast<std::size_t>(gConfig.histories));
  }
  void EndOfRunAction(const G4Run* run) override
  {
    // Run 0 is the warmup; only the measured run (1) is written.
    if (!fRec || run->GetRunID() != 1) return;
    const int tid = std::max(0, G4Threading::G4GetThreadId());
    const std::string path = gConfig.output + ".t" + std::to_string(tid) + ".bin";
    std::lock_guard<std::mutex> lock(gWriteMutex);
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(fRec->records.data()),
              static_cast<std::streamsize>(fRec->records.size() * sizeof(Record)));
  }

 private:
  Recorder* fRec;
};

class Actions : public G4VUserActionInitialization
{
 public:
  void BuildForMaster() const override { SetUserAction(new Runs(nullptr)); }
  void Build() const override
  {
    auto* rec = new Recorder();  // one per worker (or the serial thread)
    SetUserAction(new Gun());
    SetUserAction(new Stepping(rec));
    SetUserAction(new Stacking());
    SetUserAction(new Events(rec));
    SetUserAction(new Runs(rec));
  }
};

double Seconds(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
{
  return std::chrono::duration<double>(b - a).count();
}

void SetSeeds(long seed)
{
  long seeds[3] = {seed, 67890, 0};
  G4Random::setTheSeeds(seeds);
}
}  // namespace

int main(int argc, char** argv)
{
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--material") gConfig.material = v;
    else if (k == "--energy-kev") gConfig.energy_keV = std::stod(v);
    else if (k == "--thickness-um") gConfig.thickness_um = std::stod(v);
    else if (k == "--angle-deg") gConfig.angle_deg = std::stod(v);
    else if (k == "--histories") gConfig.histories = std::stol(v);
    else if (k == "--warmup") gConfig.warmup = std::stol(v);
    else if (k == "--seed") gConfig.seed = std::stol(v);
    else if (k == "--threads") gConfig.threads = std::stoi(v);
    else if (k == "--output") gConfig.output = v;
    else if (k == "--step-dr") gConfig.step_dr = std::stod(v);
    else if (k == "--step-final-um") gConfig.step_final_um = std::stod(v);
    else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      return 2;
    }
  }
  if (gConfig.material != "Si" && gConfig.material != "W") return 2;
  const auto t0 = std::chrono::steady_clock::now();
  auto type = gConfig.threads > 1 ? G4RunManagerType::MTOnly : G4RunManagerType::SerialOnly;
  auto* rm = G4RunManagerFactory::CreateRunManager(type, gConfig.threads);
  rm->SetVerboseLevel(0);
  rm->SetUserInitialization(new Detector());
  rm->SetUserInitialization(new PhysicsList());
  rm->SetUserInitialization(new Actions());
  rm->Initialize();
  const auto t1 = std::chrono::steady_clock::now();
  SetSeeds(gConfig.seed + 1000000);
  rm->BeamOn(static_cast<G4int>(gConfig.warmup));
  const auto t2 = std::chrono::steady_clock::now();
  SetSeeds(gConfig.seed);
  rm->BeamOn(static_cast<G4int>(gConfig.histories));
  const auto t3 = std::chrono::steady_clock::now();
  // Cuts and process list for the bounds audit (outside every timer).
  G4ProductionCutsTable::GetProductionCutsTable()->DumpCouples();
  const auto* processes = G4Electron::Electron()->GetProcessManager()->GetProcessList();
  for (std::size_t i = 0; i < processes->size(); ++i) {
    std::printf("G4BENCH_PROCESS e- %s\n", (*processes)[i]->GetProcessName().c_str());
  }
  std::printf("G4BENCH_TIMING init_s=%.6f warmup_s=%.6f measured_s=%.6f threads=%d\n",
              Seconds(t0, t1), Seconds(t1, t2), Seconds(t2, t3), gConfig.threads);
  std::printf("G4BENCH_RECORD_BYTES %zu\n", sizeof(Record));
  delete rm;
  return 0;
}
