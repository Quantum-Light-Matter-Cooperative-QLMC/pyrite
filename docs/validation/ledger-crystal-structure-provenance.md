# Crystal structure & CIF provenance

Part of the [physics validation ledger](physics-validation-ledger.md). See the [validation methodology](methodology.md) for the status lifecycle and the [domain inventories](domain-inventories.md) for a claim-by-claim index.

## `pdte2-debye-waller-001`

- **Claim:** 113(2) K 1T-PdTe2 common basal projection `B33=0.61 Å²` for pinned `(001)`
- **Code:** `data/catalog/crystals/pdte2.toml::B_ang2`
- **Source:** Pell, Mironov & Ibers, *Acta Cryst. C* **52**, 1331–1332 (1996), doi:10.1107/S0108270195016246, COD 2004955
- **Status:** unverified
- **Checks:** source gives Pd `U33=0.0074(4)` and Te `U33=0.0079(3) Å²`; multiplicity-weighted `B33=8π²(0.00773)=0.6106 Å²`; site agreement, unit occupancy, basal-only scope, and 10 keV intensity sensitivity checked
- **Anchor:** catalog golden serialization
- **Notes:** Direction-specific scalar, not `Biso`. New scalar changes `|F001|²` by −0.019% versus placeholder and −0.071% versus site-specific tensors at 10 keV.

## `2h-tas2-debye-waller-002`

- **Claim:** 295 K 2H-TaS2 common basal projection `B33=0.53 Å²` for pinned `(002)`
- **Code:** `data/catalog/crystals/2h_tas2.toml::B_ang2`
- **Source:** Meetsma *et al.*, *Acta Cryst. C* **46**, 1598–1599 (1990), doi:10.1107/S0108270190000014, COD 9007815
- **Status:** unverified
- **Checks:** source gives Ta `U33=0.0065(2)` and S `U33=0.0068(8) Å²`; multiplicity-weighted `B33=8π²(0.00670)=0.5290 Å²`; site agreement, unit occupancy, basal-only scope, and 10 keV intensity sensitivity checked
- **Anchor:** catalog golden serialization
- **Notes:** Direction-specific scalar, not `Biso`. New scalar changes `|F002|²` by +0.096% versus placeholder and −0.024% versus site-specific tensors at 10 keV.

## `diamond-cif-migration`

- **Claim:** bundled diamond CIF preserves the pre-CIF catalog's structure and derived physics
- **Code:** `data/cifs/diamond.cif`; `data/catalog/crystals/diamond.toml`
- **Source:** —
- **Status:** unverified
- **Checks:** lattice, expanded basis, volume, composition, structure factor, and dominant-reflection golden
- **Anchor:** `tests/materials/test_material_catalog.py::test_packaged_catalog_matches_independent_serialized_golden`, `::test_catalog_matches_serialized_physics_for_every_crystal`
- **Notes:** migration-equivalence claim only; no new structure source claimed

## `silicon-cif-migration`

- **Claim:** bundled silicon CIF preserves the pre-CIF catalog's structure and derived physics
- **Code:** `data/cifs/silicon.cif`; `data/catalog/crystals/silicon.toml`
- **Source:** —
- **Status:** unverified
- **Checks:** lattice, expanded basis, volume, composition, structure factor, and dominant-reflection golden
- **Anchor:** `tests/materials/test_material_catalog.py::test_packaged_catalog_matches_independent_serialized_golden`, `::test_catalog_matches_serialized_physics_for_every_crystal`
- **Notes:** migration-equivalence claim only; no new structure source claimed

## `lif-cif-migration`

- **Claim:** bundled LiF CIF preserves the pre-CIF catalog's structure and derived physics
- **Code:** `data/cifs/lif.cif`; `data/catalog/crystals/lif.toml`
- **Source:** —
- **Status:** unverified
- **Checks:** lattice, expanded basis, volume, composition, structure factor, and dominant-reflection golden
- **Anchor:** `tests/materials/test_material_catalog.py::test_packaged_catalog_matches_independent_serialized_golden`, `::test_catalog_matches_serialized_physics_for_every_crystal`
- **Notes:** migration-equivalence claim only; no new structure source claimed

## `hopg-cif-migration`

- **Claim:** bundled HOPG CIF preserves the pre-CIF catalog's structure and derived physics
- **Code:** `data/cifs/hopg.cif`; `data/catalog/crystals/hopg.toml`
- **Source:** —
- **Status:** unverified
- **Checks:** lattice, expanded basis, volume, composition, structure factor, and pinned-reflection golden
- **Anchor:** `tests/materials/test_material_catalog.py::test_packaged_catalog_matches_independent_serialized_golden`, `::test_catalog_matches_serialized_physics_for_every_crystal`
- **Notes:** migration-equivalence claim only; no new structure source claimed

## `mose2-cif-migration`

- **Claim:** bundled 2H-MoSe2 single-crystal lattice and P1-expanded basis
- **Code:** `data/cifs/mose2.cif`; `data/catalog/crystals/mose2.toml`
- **Source:** Bronsema, *Z. Anorg. Allg. Chem.* **540/541**, 15--17 (1986), doi:10.1002/zaac.19865400904
- **Status:** unverified
- **Checks:** lattice, expanded basis, volume, composition, structure factor, and dominant-reflection golden
- **Anchor:** `tests/materials/test_material_catalog.py::test_packaged_catalog_matches_independent_serialized_golden`, `::test_catalog_matches_serialized_physics_for_every_crystal`
- **Notes:** a=3.289 A, c=12.927 A, Se z=0.6210 reproduce the single-crystal refinement; MP mp-1634 remains a secondary provenance pointer; no experimental PXR/CBS agreement claimed.

## `wse2-cif-migration`

- **Claim:** bundled 2H-WSe2 single-crystal lattice and P1-expanded basis
- **Code:** `data/cifs/wse2.cif`; `data/catalog/crystals/wse2.toml`
- **Source:** Schutte, De Boer & Jellinek, *J. Solid State Chem.* **70**, 207--209 (1987), doi:10.1016/0022-4596(87)90057-0; COD 9012193
- **Status:** unverified
- **Checks:** lattice, expanded basis, volume, composition, structure factor, dominant-reflection golden, COD lattice comparison
- **Anchor:** `tests/materials/test_material_catalog.py::test_packaged_catalog_matches_independent_serialized_golden`, `::test_catalog_matches_serialized_physics_for_every_crystal`; `tests/materials/test_crystal_external_db.py`
- **Notes:** a=3.282 A, c=12.960 A, Se z=0.62110 reproduce COD 9012193; MP mp-1821 remains a secondary provenance pointer; no experimental PXR/CBS agreement claimed.

## `ptse2-cif-migration`

- **Claim:** bundled 1T-PtSe2 CIF preserves the pre-CIF catalog's structure and derived physics
- **Code:** `data/cifs/ptse2.cif`; `data/catalog/crystals/ptse2.toml`
- **Source:** —
- **Status:** unverified
- **Checks:** lattice, expanded basis, volume, composition, structure factor, and dominant-reflection golden
- **Anchor:** `tests/materials/test_material_catalog.py::test_packaged_catalog_matches_independent_serialized_golden`, `::test_catalog_matches_serialized_physics_for_every_crystal`
- **Notes:** migration-equivalence claim only; no new structure source claimed

## `pts2-p3m1-structure`

- **Claim:** 1T PtS2 P-3m1 experimental lattice + explicit PtS2 primitive-cell basis, c-axis-normal layered slab
- **Code:** `data/cifs/pts2.cif`; `data/catalog/crystals/pts2.toml`
- **Source:** Furuseth, Selte & Kjekshus, *Acta Chem. Scand.* **19**, 257--258 (1965); COD 1537200 (CC0)
- **Status:** unverified
- **Checks:** hexagonal volume; 1Pt+2S stoichiometry; full source basis; finite basal `(001)` `F_g`, `chi_g`, `U_g`; COD lattice comparison
- **Anchor:** `tests/materials/test_crystallography.py::test_pts2_structure_and_basal_couplings_are_sane`; `tests/materials/test_pd_pt_crystals.py`; `tests/materials/test_crystal_external_db.py`
- **Notes:** a=3.5432 A, c=5.0388 A, S z=0.227 replace the JARVIS/MP relaxation; catalog B=0.6 A2 remains a modeling placeholder; no experimental PXR/CBS agreement claimed.

## `pds2-pbca-structure`

- **Claim:** ambient PdS2 Pbca lattice + explicit 4Pd+8S conventional-cell basis, reciprocal-(001)-normal layered slab
- **Code:** `data/cifs/pds2.cif`; `data/catalog/crystals/pds2.toml`
- **Source:** Gronvold & Rost, Acta Cryst. 10, 329--331 (1957); COD 2310589 (CC0)
- **Status:** unverified
- **Checks:** orthorhombic volume; 4Pd+8S stoichiometry; full expanded source basis; odd-00l extinction handled by pinned finite `(002)`; finite `chi_g`, `U_g`
- **Anchor:** `tests/materials/test_pd_pt_crystals.py`
- **Notes:** a=5.460 A, b=5.541 A, c=7.531 A; source Pbca basis explicitly expanded in P1; catalog B=0.6 A2 is a modeling placeholder, not source-derived; no experimental PXR/CBS agreement claimed.

## `pdte2-p3m1-structure`

- **Claim:** 1T PdTe2 P-3m1 lattice + explicit primitive-cell basis, reciprocal-(001)-normal layered slab
- **Code:** `data/cifs/pdte2.cif`; `data/catalog/crystals/pdte2.toml`
- **Source:** Pell, Mironov & Ibers, Acta Cryst. C52, 1331--1332 (1996); COD 2004955 (CC0)
- **Status:** unverified
- **Checks:** hexagonal volume; 1Pd+2Te stoichiometry; full source basis; finite basal `(001)` `F_g`, `chi_g`, `U_g`
- **Anchor:** `tests/materials/test_pd_pt_crystals.py`
- **Notes:** 113 K black-plate refinement, a=4.024 A, c=5.113 A, Te z=0.26628; catalog uses the source-derived basal projection `B33=0.61 Å²` tracked by `pdte2-debye-waller-001`; no experimental PXR/CBS agreement claimed.

## `ptbi2-p31m-structure`

- **Claim:** layered trigonal beta-PtBi2 P31m lattice + explicit 3Pt+6Bi conventional-cell basis, reciprocal-(001)-normal slab
- **Code:** `data/cifs/ptbi2.cif`; `data/catalog/crystals/ptbi2.toml`
- **Source:** Feng et al., Nature Communications 10, 4765 (2019), Supplementary Tables I--II (CC BY 4.0)
- **Status:** unverified
- **Checks:** hexagonal volume; 3Pt+6Bi stoichiometry; full P31m-expanded basis; finite basal `(001)` `F_g`, `chi_g`, `U_g`
- **Anchor:** `tests/materials/test_pd_pt_crystals.py`
- **Notes:** SI Table I 273 K cell a=6.5657 A, c=6.1601 A is used with SI Table II coordinates labeled 295 K; catalog B=0.6 A2 is a shared modeling placeholder, not either reported displacement parameter; layered trigonal phase, not cubic pyrite PtBi2; no experimental PXR/CBS agreement claimed.

## `ptte2-p3m1-structure`

- **Claim:** 1T PtTe2 P-3m1 lattice + explicit primitive-cell basis, reciprocal-(001)-normal layered slab
- **Code:** `data/cifs/ptte2.cif`; `data/catalog/crystals/ptte2.toml`
- **Source:** Furuseth, Selte & Kjekshus, Acta Chem. Scand. 19, 257--258 (1965); COD 1537197 (CC0)
- **Status:** unverified
- **Checks:** hexagonal volume; 1Pt+2Te stoichiometry; full source basis; finite basal `(001)` `F_g`, `chi_g`, `U_g`
- **Anchor:** `tests/materials/test_pd_pt_crystals.py`
- **Notes:** a=4.0259 A, c=5.2209 A, Te z=0.254; catalog B=0.6 A2 is a modeling placeholder, not source-derived; no experimental PXR/CBS agreement claimed.

## `pdse2-pbca-structure`

- **Claim:** ambient PdSe2 Pbca experimental lattice + explicit 4Pd+8Se conventional-cell basis, c-axis-normal layered slab
- **Code:** `data/cifs/pdse2.cif`; `data/catalog/crystals/pdse2.toml`
- **Source:** Soulard et al., *Inorg. Chem.* **43**, 1943--1949 (2004), doi:10.1021/ic0352396; COD 4310736 (CC0)
- **Status:** unverified
- **Checks:** orthorhombic volume; 4Pd+8Se stoichiometry; full expanded source basis; finite basal `(002)` `F_g`, `chi_g`, `U_g`; COD lattice comparison
- **Anchor:** `tests/materials/test_crystallography.py::test_pdse2_structure_and_basal_couplings_are_sane`; `tests/materials/test_crystal_external_db.py`
- **Notes:** Room-temperature refinement: a=5.7457 A, b=5.8679 A, c=7.6946 A; Se 8c x=0.11125, y=0.11799, z=0.40573 explicitly expanded from Pbca; catalog B=0.6 A2 remains a modeling placeholder; no experimental PXR/CBS agreement claimed.

## `tise2-001-structure`

- **Claim:** ambient normal-state 1T-TiSe2 P-3m1 lattice + explicit one-formula-unit basis, c-axis-normal layered slab
- **Code:** `data/cifs/tise2.cif`; `data/catalog/crystals/tise2.toml`
- **Source:** Vaterlaus, *Helv. Phys. Acta* **57**, 884 (1984), Table 2.4
- **Status:** unverified
- **Checks:** hexagonal volume; 1Ti+2Se stoichiometry; finite basal `(001)` `F_g`, `chi_g`, `U_g`; pinned `[001]` / `±(001)` catalog projection
- **Anchor:** `tests/materials/test_crystallography.py::test_tise2_001_structure_and_couplings_are_sane`; `tests/scan/test_sweep.py::test_oriented_materials_are_registered_as_symmetric_cuts`
- **Notes:** a=3.540 A, c=6.008 A, Se z=0.25504; the low-temperature charge-density-wave supercell is intentionally out of scope; no experimental PXR/CBS agreement claimed.

## `nbte2-c2m-structure`

- **Claim:** ambient distorted 1T'' NbTe2 C2/m conventional cell, reciprocal-normal (001) cleavage surface, and allowed basal (001) reflection
- **Code:** `data/cifs/nbte2.cif`; `data/catalog/crystals/nbte2.toml`
- **Source:** Brown, *Acta Cryst.* **20**, 264--267 (1966); COD 2310357 (CC0)
- **Status:** unverified
- **Checks:** monoclinic volume; 6Nb+12Te stoichiometry; finite `(001)` `F_g`, `chi_g`, `U_g`; pinned surface/reflection
- **Anchor:** `tests/materials/test_nb_ti_v_crystals.py`
- **Notes:** Exact Brown conventional setting with 2a/4i/4j sites expanded in P1; vendor alternate-cell indices are intentionally excluded; B=0.6 A2 is a catalog placeholder; no experimental PXR/CBS agreement claimed.

## `tite2-p3m1-structure`

- **Claim:** ambient 1T-TiTe2 P-3m1 experimental cell + explicit one-formula-unit basis, reciprocal-normal (001) cleavage surface
- **Code:** `data/cifs/tite2.cif`; `data/catalog/crystals/tite2.toml`
- **Source:** Kuznetsova et al., *Phys. Rev. B* **72**, 085418 (2005); NIST JARVIS JVASP-335 cross-check
- **Status:** unverified
- **Checks:** hexagonal volume; 1Ti+2Te stoichiometry; finite `(001)` `F_g`, `chi_g`, `U_g`; pinned surface/reflection
- **Anchor:** `tests/materials/test_nb_ti_v_crystals.py`
- **Notes:** Experimental a=3.777 A, c=6.498 A, Te z=0.2628 replaces the JARVIS-relaxed geometry; B=0.6 A2 is a catalog placeholder; no experimental PXR/CBS agreement claimed.

## `vse2-p3m1-structure`

- **Claim:** ambient normal-state 1T-VSe2 P-3m1 experimental cell + explicit one-formula-unit basis, reciprocal-normal (001) cleavage surface
- **Code:** `data/cifs/vse2.cif`; `data/catalog/crystals/vse2.toml`
- **Source:** Barua et al., *Sci. Rep.* **7**, 10964 (2017), CC BY 4.0; Stahl et al., *Inorganics* **11**, 481 (2023), CC BY 4.0; JVASP-10 cross-check
- **Status:** unverified
- **Checks:** hexagonal volume; 1V+2Se stoichiometry; finite `(001)` `F_g`, `chi_g`, `U_g`; pinned surface/reflection
- **Anchor:** `tests/materials/test_nb_ti_v_crystals.py`
- **Notes:** a=3.357 A, c=6.104 A, Se z=0.257; monolayer 2H and high-pressure phases are out of scope; B=0.6 A2 is a catalog placeholder; no experimental PXR/CBS agreement claimed.

## `vte2-c2m-structure`

- **Claim:** ambient distorted 1T'' VTe2 C2/m conventional cell, reciprocal-normal (001) cleavage surface, and allowed basal (001) reflection
- **Code:** `data/cifs/vte2.cif`; `data/catalog/crystals/vte2.toml`
- **Source:** Bronsema, Bus & Wiegers, *J. Solid State Chem.* **53**, 415--421 (1984); COD 1535594 (CC0)
- **Status:** unverified
- **Checks:** monoclinic volume; idealized 6V+12Te stoichiometry; finite `(001)` `F_g`, `chi_g`, `U_g`; pinned surface/reflection
- **Anchor:** `tests/materials/test_nb_ti_v_crystals.py`
- **Notes:** The source is V1.04Te2; the minor partial-occupancy interstitial is omitted because PyRITE requires full occupancy, while the reported primary sites and distorted cell are retained. B=0.6 A2 is a catalog placeholder; no experimental PXR/CBS agreement claimed.

## `gep-c2m-structure`

- **Claim:** layered ambient-pressure M_L-GeP C2/m conventional cell, reciprocal-normal (10-1) surface orientation, and allowed parallel (20-2) reflection
- **Code:** `data/cifs/gep.cif`; `data/catalog/crystals/gep.toml`
- **Source:** COD 1562070 (CC0); Lee et al., *J. Solid State Chem.* **224**, 62--70 (2015), doi:10.1016/j.jssc.2014.04.021
- **Status:** unverified
- **Checks:** monoclinic volume; Ge12P12 stoichiometry; finite `(20-2)` `F_g`, `chi_g`, `U_g`; pinned surface/reflection
- **Anchor:** `tests/materials/test_germanium_crystals.py::test_germanium_family_structures_and_surface_couplings`
- **Notes:** Exact COD conventional setting and six asymmetric 4i sites; B=0.35 A2 is the composition-average of the 90 K refinement; (10-1) is C-centering extinct. No experimental PXR/CBS agreement claimed.

## `ges-pnma-structure`

- **Claim:** ambient alpha-GeS standard-Pnma cell, reciprocal-normal (100) surface orientation, and allowed parallel (200) reflection
- **Code:** `data/cifs/ges.cif`; `data/catalog/crystals/ges.toml`
- **Source:** COD 8104282 (CC0); Wiedemeier and von Schnering, *Z. Kristallogr.* **148**, 295--303 (1978), doi:10.1524/zkri.1978.148.3-4.295
- **Status:** unverified
- **Checks:** orthorhombic volume; Ge4S4 stoichiometry; finite `(200)` `F_g`, `chi_g`, `U_g`; pinned surface/reflection
- **Anchor:** `tests/materials/test_germanium_crystals.py::test_germanium_family_structures_and_surface_couplings`
- **Notes:** COD/Pbnm source normalized to standard Pnma with matching coordinates; B=1.07 A2 from reported Ge/S isotropic values; (100) is extinct. No experimental PXR/CBS agreement claimed.

## `gese-pnma-structure`

- **Claim:** ambient alpha-GeSe standard-Pnma cell, reciprocal-normal (100) surface orientation, and allowed parallel (200) reflection
- **Code:** `data/cifs/gese.cif`; `data/catalog/crystals/gese.toml`
- **Source:** COD 4003515 (CC0); Murgatroyd et al., *Chem. Mater.* **32** (2020), doi:10.1021/acs.chemmater.0c00453
- **Status:** unverified
- **Checks:** orthorhombic volume; Ge4Se4 stoichiometry; finite `(200)` `F_g`, `chi_g`, `U_g`; pinned surface/reflection
- **Anchor:** `tests/materials/test_germanium_crystals.py::test_germanium_family_structures_and_surface_couplings`
- **Notes:** Exact 250.72 K COD cell and y=3/4 origin choice; B=1.10 A2 from the selected refinement's Ueq values; (100) is extinct. No experimental PXR/CBS agreement claimed.

## `gese2-beta-p21c-structure`

- **Claim:** layered beta-GeSe2 experimental initial structure transformed losslessly to standard P21/c:b1, reciprocal-normal (001) surface orientation, and allowed parallel (002) reflection
- **Code:** `data/cifs/gese2.cif`; `data/catalog/crystals/gese2.toml`
- **Source:** Materials Project mp-540625 initial structure, matminer `mp_all_20181018` snapshot (CC BY 4.0); Dittmar and Schaefer, *Acta Cryst.* B32, 2726--2728 (1976), doi:10.1107/S0567740876008704
- **Status:** unverified
- **Checks:** monoclinic volume; Ge16Se32 stoichiometry; finite `(002)` `F_g`, `chi_g`, `U_g`; pinned surface/reflection
- **Anchor:** `tests/materials/test_germanium_crystals.py::test_germanium_family_structures_and_surface_couplings`
- **Notes:** Axis/coordinate transform and artifact SHA-256 are recorded in the CIF; B=0.6 A2 remains an explicit catalog placeholder because no source B/U table was recovered; (001) is c-glide extinct. Tetragonal high-pressure COD 1521080 is explicitly excluded; no experimental PXR/CBS agreement claimed.

## `res2-mp572758-structure`

- **Claim:** distorted-1T ReS2 doubled-c P-1 DFT model with reciprocal-normal (001) cleavage orientation and dominant single-layer basal (002) reflection (odd 00l are near-extinct supercell harmonics)
- **Code:** `data/cifs/res2.cif`; `data/catalog/crystals/res2.toml`
- **Source:** Materials Project mp-572758, matminer `mp_all_20181018` snapshot (CC BY 4.0); compared against Lamfers et al., *J. Alloys Compd.* **241**, 34--39 (1996), doi:10.1016/0925-8388(96)02313-4
- **Status:** unverified
- **Checks:** triclinic volume; exact Re8S16 full basis; finite `(002)` `F_g`, `chi_g`, `U_g`; pinned surface/reflection
- **Anchor:** `tests/materials/test_re_ta_crystals.py`
- **Notes:** Lamfers is the preferred experimental doubled-cell provenance, but no open full fractional-basis deposit was found; retaining the internally consistent MP model is safer than combining experimental lattice constants with DFT coordinates. Snapshot date/hash and P1 normalization are recorded in the CIF; B=0.6 A2 is a modeling placeholder; no experimental PXR/CBS agreement claimed.

## `rese2-cod1539529-structure`

- **Claim:** ambient distorted-1T ReSe2 P-1 historical model with reciprocal-normal (001) cleavage orientation and allowed basal (001) reflection
- **Code:** `data/cifs/rese2.cif`; `data/catalog/crystals/rese2.toml`
- **Source:** Alcock and Kjekshus, *Acta Chem. Scand.* **19**, 79--94 (1965); COD 1539529 (CC0)
- **Status:** unverified
- **Checks:** triclinic volume; exact Re4Se8 full basis; finite `(001)` `F_g`, `chi_g`, `U_g`; pinned surface/reflection
- **Anchor:** `tests/materials/test_re_ta_crystals.py`
- **Notes:** Provisional structure: later work reports corrected pseudo-centres and revised cell parameters, but no comparably redistributable modern CIF was identified. B=0.6 A2 is a modeling placeholder; no experimental PXR/CBS agreement claimed.

## `2h-tas2-p63mmc-structure`

- **Claim:** room-temperature parent 2H-TaS2 P63/mmc conventional cell, reciprocal-normal (001) cleavage orientation, and first allowed parallel (002) reflection
- **Code:** `data/cifs/2h_tas2.cif`; `data/catalog/crystals/2h_tas2.toml`
- **Source:** Meetsma et al., *Acta Cryst.* C46, 1598--1599 (1990); COD 9007815 (CC0)
- **Status:** unverified
- **Checks:** hexagonal volume; exact Ta2S4 full basis; odd-00l extinction; finite `(002)` `F_g`, `chi_g`, `U_g`; pinned surface/reflection
- **Anchor:** `tests/materials/test_re_ta_crystals.py`
- **Notes:** Low-temperature CDW superstructure is excluded; catalog uses the source-derived basal projection `B33=0.53 Å²` tracked by `2h-tas2-debye-waller-002`; no experimental PXR/CBS agreement claimed.

## `2h-tase2-p63mmc-structure`

- **Claim:** room-temperature parent 2H-TaSe2 P63/mmc conventional cell, reciprocal-normal (001) cleavage orientation, and first allowed parallel (002) reflection
- **Code:** `data/cifs/2h_tase2.cif`; `data/catalog/crystals/2h_tase2.toml`
- **Source:** Brown and Beerntsen, *Acta Cryst.* **18**, 31--36 (1965); COD 2310532 (CC0)
- **Status:** unverified
- **Checks:** hexagonal volume; exact Ta2Se4 full basis; odd-00l extinction; finite `(002)` `F_g`, `chi_g`, `U_g`; pinned surface/reflection
- **Anchor:** `tests/materials/test_re_ta_crystals.py`
- **Notes:** Explicit P1 expansion normalizes rounded COD symops that crystals 1.7 can fail to deduplicate; CDW and 4Hb phases are excluded; B=0.6 A2 is a modeling placeholder; no experimental PXR/CBS agreement claimed.

## `tate2-c2m-structure`

- **Claim:** ambient distorted-1T TaTe2 C2/m conventional cell in the Brown long-a setting, reciprocal-normal (001) cleavage orientation, and allowed basal (001) reflection
- **Code:** `data/cifs/tate2.cif`; `data/catalog/crystals/tate2.toml`
- **Source:** Brown, *Acta Cryst.* **20**, 264--267 (1966); COD 2310358 (CC0)
- **Status:** unverified
- **Checks:** monoclinic volume; exact Ta6Te12 full basis; finite `(001)` `F_g`, `chi_g`, `U_g`; pinned surface/reflection
- **Anchor:** `tests/materials/test_re_ta_crystals.py`
- **Notes:** The additional low-temperature approximately 3b superstructure and alternate reduced settings are excluded; B=0.6 A2 is a modeling placeholder; no experimental PXR/CBS agreement claimed.

## `fete-p4nmm-idealized-structure`

- **Claim:** idealized stoichiometric beta-FeTe PbO-type framework derived from an experimental P4/nmm Fe1.095Te parent, reciprocal-normal (001) cleavage orientation, and allowed basal (001) reflection
- **Code:** `data/cifs/fete.cif`; `data/catalog/crystals/fete.toml`
- **Source:** Rodriguez et al., *J. Am. Chem. Soc.* **132**, 10006--10008 (2010), doi:10.1021/ja104004t; COD 4102703 (CC0)
- **Status:** unverified
- **Checks:** tetragonal volume; exact retained Fe2Te2 full-occupancy framework basis; finite `(001)` `F_g`, `chi_g`, `U_g`; pinned surface/reflection; idealization disclosure
- **Anchor:** `tests/materials/test_fe_w_zr_crystals.py`
- **Notes:** The source's 0.095-occupied interstitial Fe 2c orbit is intentionally omitted because PyRITE supports only full occupancy. This is explicitly not the exact Fe1.095Te refinement; B=0.6 A2 is a modeling placeholder; no experimental PXR/CBS agreement claimed.

## `wte2-td-pnm21-structure`

- **Claim:** ambient orthorhombic Td-WTe2 Pnm21 conventional cell in the Brown axis setting, reciprocal-normal (001) cleavage orientation, and first allowed parallel (002) reflection
- **Code:** `data/cifs/wte2.cif`; `data/catalog/crystals/wte2.toml`
- **Source:** Brown, *Acta Cryst.* **20**, 268--274 (1966); COD 2310355 (CC0)
- **Status:** unverified
- **Checks:** orthorhombic volume; exact W4Te8 full basis; odd-00l extinction; finite `(002)` `F_g`, `chi_g`, `U_g`; pinned surface/reflection
- **Anchor:** `tests/materials/test_fe_w_zr_crystals.py`
- **Notes:** The Brown a=6.282, b=3.496 A setting is retained without the common vendor axis swap; B=0.6 A2 is a modeling placeholder; no experimental PXR/CBS agreement claimed.

## `zrte3-p21m-structure`

- **Claim:** room-temperature type-A ZrTe3 P21/m parent cell, reciprocal-normal (001) cleavage orientation, and allowed basal (001) reflection
- **Code:** `data/cifs/zrte3.cif`; `data/catalog/crystals/zrte3.toml`
- **Source:** Furuseth and Fjellvag, *Acta Chem. Scand.* **45**, 694--697 (1991), doi:10.3891/acta.chem.scand.45-0694; COD 1559502 (CC0)
- **Status:** unverified
- **Checks:** monoclinic volume; exact Zr2Te6 full basis; finite `(001)` `F_g`, `chi_g`, `U_g`; pinned surface/reflection
- **Anchor:** `tests/materials/test_fe_w_zr_crystals.py`
- **Notes:** The 293 K parent structure is used and the low-temperature CDW modulation is excluded; B=0.6 A2 is a modeling placeholder; no experimental PXR/CBS agreement claimed.

## `zrte5-cmcm-structure`

- **Claim:** room-temperature orthorhombic ZrTe5 Cmcm cell, reciprocal-normal (010) cleavage orientation, and first allowed parallel (020) reflection
- **Code:** `data/cifs/zrte5.cif`; `data/catalog/crystals/zrte5.toml`
- **Source:** Fjellvag and Kjekshus, *Solid State Commun.* **60**, 91--93 (1986), ICSD 85506; open values re-tabulated by Facio et al., *SciPost Phys.* **14**, 066 (2023), Table 1 structure B (CC BY 4.0)
- **Status:** unverified
- **Checks:** orthorhombic volume; exact Zr4Te20 full basis; C-centering (010) extinction; finite `(020)` `F_g`, `chi_g`, `U_g`; pinned surface/reflection
- **Anchor:** `tests/materials/test_fe_w_zr_crystals.py`
- **Notes:** The bundled P1 expansion is constructed from the openly licensed 293 K table, not redistributed from proprietary ICSD; B=0.6 A2 is a modeling placeholder; no experimental PXR/CBS agreement claimed.

## `black-phosphorus-020-structure`

- **Claim:** ambient black-phosphorus Cmce lattice + explicit 8P conventional-cell basis, b-axis-normal layered slab
- **Code:** `data/cifs/black_phosphorus.cif`; `data/catalog/crystals/black_phosphorus.toml`
- **Source:** Brown & Rundqvist, Acta Cryst. 19, 684--685 (1965), doi:10.1107/S0365110X65004140
- **Status:** unverified
- **Checks:** orthorhombic volume; 8P stoichiometry; finite basal `(020)` `F_g`, `chi_g`, `U_g`; pinned `[010]` / `±(020)` catalog projection
- **Anchor:** `tests/materials/test_crystallography.py::test_black_phosphorus_020_structure_and_couplings_are_sane`; `tests/scan/test_sweep.py::test_oriented_materials_are_registered_as_symmetric_cuts`
- **Notes:** a=3.3136 A, b=10.478 A, c=4.3763 A; P 8f y=0.10168, z=0.08056 expanded in P1; no experimental PXR/CBS agreement claimed.

## `4h-sic-0004-structure`

- **Claim:** ambient 4H-SiC hexagonal conventional cell, configured as a symmetric basal (0004) cut
- **Code:** `data/cifs/4h_sic.cif`; `data/catalog/crystals/4h_sic.toml`
- **Source:** 4H-SiC room-temperature powder refinement
- **Status:** unverified
- **Checks:** hexagonal volume; 4Si+4C; finite `(0004)` `F_g`, `chi_g`, `U_g`; pinned `[001]` / `±(0004)` catalog projection
- **Anchor:** `tests/materials/test_crystallography.py::test_hexagonal_sic_basal_structures_and_couplings_are_sane`; `tests/scan/test_sweep.py::test_oriented_materials_are_registered_as_symmetric_cuts`
- **Notes:** no experimental PXR/CBS agreement claimed.

## `6h-sic-0006-structure`

- **Claim:** ambient 6H-SiC hexagonal conventional cell, configured as a symmetric basal (0006) cut
- **Code:** `data/cifs/6h_sic.cif`; `data/catalog/crystals/6h_sic.toml`
- **Source:** Capitani et al., *American Mineralogist* **92**, 403–407 (2007)
- **Status:** unverified
- **Checks:** hexagonal volume; 6Si+6C; finite `(0006)` `F_g`, `chi_g`, `U_g`; pinned `[001]` / `±(0006)` catalog projection
- **Anchor:** `tests/materials/test_crystallography.py::test_hexagonal_sic_basal_structures_and_couplings_are_sane`; `tests/scan/test_sweep.py::test_oriented_materials_are_registered_as_symmetric_cuts`
- **Notes:** a=3.0810 A, c=15.1248 A; no experimental PXR/CBS agreement claimed.

## `hfse2-cif-migration`

- **Claim:** bundled 1T-HfSe2 CIF preserves the pre-CIF catalog's structure and derived physics
- **Code:** `data/cifs/hfse2.cif`; `data/catalog/crystals/hfse2.toml`
- **Source:** —
- **Status:** unverified
- **Checks:** lattice, expanded basis, volume, composition, structure factor, and dominant-reflection golden
- **Anchor:** `tests/materials/test_material_catalog.py::test_packaged_catalog_matches_independent_serialized_golden`, `::test_catalog_matches_serialized_physics_for_every_crystal`
- **Notes:** migration-equivalence claim only; no new structure source claimed

## `zrse2-cif-migration`

- **Claim:** bundled 1T-ZrSe2 CIF preserves the pre-CIF catalog's structure and derived physics
- **Code:** `data/cifs/zrse2.cif`; `data/catalog/crystals/zrse2.toml`
- **Source:** —
- **Status:** unverified
- **Checks:** lattice, expanded basis, volume, composition, structure factor, and dominant-reflection golden
- **Anchor:** `tests/materials/test_material_catalog.py::test_packaged_catalog_matches_independent_serialized_golden`, `::test_catalog_matches_serialized_physics_for_every_crystal`
- **Notes:** migration-equivalence claim only; no new structure source claimed

## `ws2-cif-migration`

- **Claim:** bundled 2H-WS2 CIF preserves the pre-CIF catalog's structure and derived physics
- **Code:** `data/cifs/ws2.cif`; `data/catalog/crystals/ws2.toml`
- **Source:** —
- **Status:** unverified
- **Checks:** lattice, expanded basis, volume, composition, structure factor, and dominant-reflection golden
- **Anchor:** `tests/materials/test_material_catalog.py::test_packaged_catalog_matches_independent_serialized_golden`, `::test_catalog_matches_serialized_physics_for_every_crystal`
- **Notes:** migration-equivalence claim only; no new structure source claimed

## `mos2-cif-migration`

- **Claim:** bundled 2H-MoS2 CIF preserves the pre-CIF catalog's structure and derived physics
- **Code:** `data/cifs/mos2.cif`; `data/catalog/crystals/mos2.toml`
- **Source:** —
- **Status:** unverified
- **Checks:** lattice, expanded basis, volume, composition, structure factor, and dominant-reflection golden
- **Anchor:** `tests/materials/test_material_catalog.py::test_packaged_catalog_matches_independent_serialized_golden`, `::test_catalog_matches_serialized_physics_for_every_crystal`
- **Notes:** migration-equivalence claim only; no new structure source claimed

## `mote2-bulk-structure`

- **Claim:** 2H-MoTe2 bulk lattice + basis (a=3.517 Å, c=13.96 Å)
- **Code:** `data/cifs/mote2.cif`; `data/catalog/crystals/mote2.toml`
- **Source:** literature / Materials Project
- **Status:** unverified
- **Checks:** cell volume + stoichiometry check
- **Anchor:** `tests/materials/test_crystallography.py::test_mote2_structure_sane`
- **Notes:** bulk material, used in bare MoTe2 scans

## `mote2-product-structure`

- **Claim:** 2H-MoTe2 product-page lattice + basis used for few-layer MoTe2-on-sapphire scans
- **Code:** `data/cifs/mote2_product.cif`; `data/catalog/crystals/mote2_product.toml`
- **Source:** 2D Semiconductors product page
- **Status:** unverified
- **Checks:** unit conversion nm→Å + cell-volume check
- **Anchor:** `tests/materials/test_crystallography.py::test_mote2_product_structure_sane`
- **Notes:** product-page lattice (a=3.50 Å, c=13.41 Å) differs from bulk for thin films

## `hfs2-structure`

- **Claim:** bulk 1T-HfS2 P-3m1 lattice + three-atom primitive-cell basis
- **Code:** `data/cifs/hfs2.cif`; `data/catalog/crystals/hfs2.toml`
- **Source:** 2D Semiconductors product lattice; Neal et al., npj 2D Mater. Appl. 5, 45 (2021); Iwasaki et al., JPSJ 51, 2233 (1982)
- **Status:** rederived
- **Checks:** V=65.82 Å³; 1 Hf + 2 S; z=0.25 gives d(Hf-S)=2.544 Å and 2.90 Å sheet thickness; odd basal (001) allowed
- **Anchor:** `tests/materials/test_crystallography.py::test_hfs2_structure_sane`
- **Notes:** independent derivation matches; [validation write-up](materials/hfs2-structure.md); a=3.62 Å, c=5.80 Å from product page; 1T phase/space group independently supported; idealized octahedral S z=0.25 (exact regularity would be z=0.2548); no unverified vendor mosaic value encoded

## `hfte2-001-structure`

- **Claim:** 1T-HfTe2 P-3m1 Materials Project relaxed lattice + explicit 1a/2d primitive basis, configured as a symmetric (001) cut
- **Code:** `data/cifs/hfte2.cif`; `data/catalog/crystals/hfte2.toml`
- **Source:** Materials Project mp-32887
- **Status:** unverified
- **Checks:** hexagonal volume; 1Hf+2Te; finite `(001)` `F_g`, `chi_g`, `U_g`; pinned `[001]` / `±(001)` catalog projection
- **Anchor:** `tests/materials/test_crystallography.py::test_hfte2_structure_and_001_couplings_are_sane`; `tests/scan/test_sweep.py::test_oriented_materials_are_registered_as_symmetric_cuts`
- **Notes:** MP-relaxed a=4.01826066 A, c=7.628794 A, Te z=0.228088; this is not the vendor lattice previously stubbed in the TOML; no experimental PXR/CBS agreement claimed.

## `sapphire-corundum-structure`

- **Claim:** α-Al2O3/sapphire corundum lattice + explicit conventional-cell basis, B_ang2=0.25
- **Code:** `data/cifs/sapphire.cif`; `data/catalog/crystals/sapphire.toml`
- **Source:** Newnham & de Haan 1962; B_ang2 literature ~0.25
- **Status:** unverified
- **Checks:** cell volume + stoichiometry check
- **Anchor:** `tests/materials/test_crystallography.py::test_sapphire_structure_sane`
- **Notes:** R-3c Wyckoff sites are explicit in the bundled P1 CIF; the adapter also supports symmetry expansion for non-P1 CIFs; Debye-Waller B_ang2 updated from 0.5 Ų to 0.25 Ų (literature range 0.20–0.30)

## `hbn-structure`

- **Claim:** h-BN P6_3/mmc layered/eclipsed lattice + explicit four-atom conventional-cell basis
- **Code:** `data/cifs/hbn.cif`; `data/catalog/crystals/hbn.toml`
- **Source:** Pease, Acta Cryst 5, 356 (1952)
- **Status:** rederived
- **Checks:** V=36.17 Å³ ✓; 2B+2N ✓; basis ≡ Pease Wyckoff under shift (2/3,1/3,3/4) ✓; AA′ registry ✓ (anchor green)
- **Anchor:** `tests/materials/test_crystallography.py::test_hbn_structure_sane`
- **Notes:** a=2.504 A, c=6.661 A; B/N sites swap across the half-cell so B lies above N; write-up `materials/hbn-structure.md`

## `hbn-debye-waller-00l`

- **Claim:** room-temperature h-BN basal-reflection Debye–Waller coefficient `B33=3.45 Å²` (`U33=0.0437 Å²`) used for pinned `(002)/(004)`
- **Code:** `data/catalog/crystals/hbn.toml::B_ang2`
- **Source:** Pease, *Acta Cryst.* **5**, 356–361 (1952), Table 1 and Fig. 3; doi:10.1107/S0365110X52001064
- **Status:** unverified
- **Checks:** weighted fit of `ln[(Fobs/Fcalc)²]` vs `l²` for `002/004/006/008`: slope `−0.03888(834)`, then `B33=−2c²·slope=3.45(74) Å²`; `U33=B33/(8π²)=0.0437(94) Å²`; catalog golden pins chosen value
- **Anchor:** `tests/materials/test_material_catalog.py::test_packaged_catalog_matches_independent_serialized_golden`, `::test_catalog_matches_serialized_physics_for_every_crystal`
- **Notes:** Pease reports room-temperature `00l` temperature-factor data and a 410 K directional Debye temperature, not atom-specific isotropic ADPs. h-BN motion is strongly anisotropic (`U33 ≫ U11`); scalar schema cannot represent its tensor. Value is deliberately a **direction-specific `B33` approximation**, valid here because catalog pins only basal `(002)/(004)` reflections; do not reuse as `Biso` or for non-basal h-BN reflections. Zotero key `MFWKLT5G`, citation key `peaseXrayStudyBoron1952`.

## `nbs2-2ha-structure`

- **Claim:** 2H-a NbS2 P6_3/mmc lattice + explicit 2b/4f conventional-cell basis
- **Code:** `data/cifs/nbs2.cif`; `data/catalog/crystals/nbs2.toml`
- **Source:** supplier a/c via El Youbi et al., PRB 103, 155105 (2021) supplement; z via Heil et al., PRB 98, 075120 (2018); AFLOW `AB2_hP6_194_b_f-002`
- **Status:** unverified
- **Checks:** cell volume; 2Nb+4S; aligned Nb columns; finite `F_g`, `chi_g`, `U_g`
- **Anchor:** `tests/materials/test_crystallography.py::test_2ha_niobium_dichalcogenide_structure`, `::test_2ha_niobium_dichalcogenide_couplings_are_finite`
- **Notes:** [validation write-up](materials/2ha-niobium-dichalcogenides.md)

## `nbse2-2ha-structure`

- **Claim:** 2H-a NbSe2 P6_3/mmc lattice + explicit 2b/4f conventional-cell basis
- **Code:** `data/cifs/nbse2.cif`; `data/catalog/crystals/nbse2.toml`
- **Source:** a/c via Wang et al., APL 123, 153505 (2023); experimental z range via Johannes et al., PRB 73, 205102 (2006) and primary refinements; AFLOW `AB2_hP6_194_b_f-002`
- **Status:** unverified
- **Checks:** cell volume; 2Nb+4Se; aligned Nb columns; finite `F_g`, `chi_g`, `U_g`
- **Anchor:** `tests/materials/test_crystallography.py::test_2ha_niobium_dichalcogenide_structure`, `::test_2ha_niobium_dichalcogenide_couplings_are_finite`
- **Notes:** room-temperature unmodulated cell; [validation write-up](materials/2ha-niobium-dichalcogenides.md)

## `fes2-pyrite-structure`

- **Claim:** cubic pyrite FeS2 Pa-3 lattice + explicit 4a(Fe)/8c(S) conventional-cell basis
- **Code:** `data/cifs/fes2.cif`; `data/catalog/crystals/fes2.toml`
- **Source:** Finklea, Cathey & Amma, Acta Crystallogr. A32, 529 (1976); a=5.4166 A, x(S)=0.386
- **Status:** unverified
- **Checks:** cell volume; 4 Fe + 8 S; idealized x gives S-S dimer ~2.14 A and Fe-S ~2.27 A (lit. ~2.16 A / ~2.26 A); finite `(200)` `F_g`, `chi_g`, `U_g`
- **Anchor:** `tests/materials/test_material_catalog.py::test_packaged_catalog_matches_independent_serialized_golden`, `::test_catalog_matches_serialized_physics_for_every_crystal`
- **Notes:** cubic, non-vdW pyrite structure catalogued as a thin/"2D-FeS2" entry alongside the layered dichalcogenides; Pa-3 systematic absences (`h00: h=2n`) motivate the `(200)` reference beam direction; no independent Wyckoff-generator rederivation against a second source.

## `v2o5-010-structure`

- **Claim:** alpha-V2O5 standard-Pmmn lattice + explicit two-formula-unit basis for the layered cut conventionally indexed as (010)
- **Code:** `data/cifs/v2o5.cif`; `data/catalog/crystals/v2o5.toml`
- **Source:** McColl et al. 2018; Sipr et al. 1999
- **Status:** unverified
- **Checks:** orthorhombic volume; 4V+10O; finite `(001)` `F_g`, `chi_g`, `U_g`; pinned `[001]` / `±(001)` catalog projection
- **Anchor:** `tests/materials/test_crystallography.py::test_v2o5_010_structure_and_couplings_are_sane`; `tests/scan/test_sweep.py::test_oriented_materials_are_registered_as_symmetric_cuts`
- **Notes:** 10 micrometre perfect single-crystal preset; historical Pmnm `(010)` equals standard-Pmmn `(001)`; no asymmetric-reflection model; [validation write-up](materials/oriented-v2o5-tis2.md)

## `tis2-003-structure`

- **Claim:** 1T-TiS2 P-3m1 lattice + explicit one-formula-unit basis for a symmetric (003) cut
- **Code:** `data/cifs/tis2.cif`; `data/catalog/crystals/tis2.toml`
- **Source:** Brown et al. 1998
- **Status:** unverified
- **Checks:** hexagonal volume; 1Ti+2S; finite `(003)` `F_g`, `chi_g`, `U_g`; pinned `[001]` / `±(003)` catalog projection
- **Anchor:** `tests/materials/test_crystallography.py::test_tis2_003_structure_and_couplings_are_sane`; `tests/scan/test_sweep.py::test_oriented_materials_are_registered_as_symmetric_cuts`
- **Notes:** 10 micrometre perfect single-crystal preset; odd basal order allowed by the 1T basis; [validation write-up](materials/oriented-v2o5-tis2.md)
