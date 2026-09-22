      INCLUDE 'radial.f'
      INCLUDE 'elsepa2020.f'

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C
C    +++++++++++++++++++++++++++++
C    +++    PROGRAM ELSCATA    +++
C    +++++++++++++++++++++++++++++
C
C
C                               F. Salvat, A. Jablonski and C.J. Powell
C                               November 30, 2003
C
C  Updated in February 2020 by F. Salvat, to work with the subroutine
C  package RADIAL.
C
C  Ref.: F. Salvat and J. M. Fernandez-Varea,
C        'RADIAL: a Fortran subroutine package for the solution of
C        radial Schrodinger and Dirac wave equations',
C        Comput. Phys. Commun. 240 (2019) 165-177.
C
C
C  ELastic SCATtering of electrons and positrons by Atoms (and ions)
C  --      ----                                     -
C
C     This program does relativistic (Dirac) partial wave calculations
C  of elastic scattering of electrons and positrons by neutral atoms and
C  positive ions with Z=1-103. The interaction between the projectile
C  and the target system is assumed to be equal to the electrostatic
C  potential energy, plus a local exchange potential in the case of
C  projectile electrons. For projectiles with relatively low energies,
C  approximate correlation-polarization and absorption corrections can
C  be introduced.
C
C     The code calculates the phase shifts, the direct and spin-flip
C  scattering amplitudes, the differential cross section (DCS) and the
C  Sherman function (spin polarization) for electrons and positrons with
C  kinetic energies larger than 10 eV. For kinetic energies larger than
C  10 MeV, the convergence of the partial-wave series is too slow and
C  the differential cross section is calculated using approximate high-
C  energy factorization methods.
C
C
C  **** The input data file.
C
C     Data are read from a formatted input file (unit 5). Each line in
C  this file consists of a 6-character keyword (columns 1-6) followed by
C  a numerical value (in free format) that is written on the columns
C  8-19. Keywords are explicitly used/verified by the program (which is
C  case sensitive!). The text after column 20 describes the input
C  quantity and its default value (in square brackets). This text is a
C  reminder for the user and is not read by the program.
C
C     Lines defining default values can be omitted from the input file.
C  The program assigns default values to input parameters that are not
C  explicitly defined, and also to those that are manifestly wrong. The
C  code stops when it finds an inconsistent input datum. The conflicting
C  quantity appears in the last line written on the screen.
C
C ----+----1----+----2----+----3----+----4----+----5----+----6----+----7
C IZ      80         atomic number                               [none]
C MNUCL   3          rho_n (1=P, 2=U, 3=F, 4=Uu)                  [  3]
C NELEC   80         number of bound electrons                    [ IZ]
C MELEC   4          rho_e (1=TFM, 2=TFD, 3=DHFS, 4=DF, 5=file)   [  4]
C MUFFIN  0          0=free atom, 1=muffin-tin model              [  0]
C RMUF   -1.0        muffin-tin radius (cm)                  [measured]
C IELEC  -1          -1=electron, +1=positron                     [ -1]
C MEXCH   1          V_ex (0=none, 1=FM, 2=TF, 3=RT)              [  1]
C MCPOL   2          V_cp (0=none, 1=B, 2=LDA)                    [  0]
C VPOLA  -1.0        atomic polarizability (cm^3)            [measured]
C VPOLB  -1.0        b_pol parameter                          [default]
C MABS    2          W_abs (0=none, 1=LDA-I, 2=LDA-II)            [  0]
C VABSA  -1.0        absorption-potential strength, Aabs      [default]
C VABSD  -1.0        energy gap DELTA (eV)                    [default]
C IHEF    2          high-E factorization (0=no, 1=yes, 2=Born)   [  1]
C EV      1.000E2    kinetic energy (eV)                         [none]
C EV      1.000E3    optionally, more energies...
C ----+----1----+----2----+----3----+----4----+----5----+----6----+----7
C
C  NOTE: The absorption potential W_abs is obtained from the local-
C  density approximation (LDA) by using the Lindhard dielectric function
C  of the free-electron gas. The LDA-I option only accounts for
C  electron-hole excitations, while the LDA-II option uses the full
C  dielectric function, i.e., it includes plasmon-like excitations.
C
C     The computed information is delivered in various files with the
C  extension '.dat'. The output file 'dcs_xpyyyezz.dat' contains the
C  calculated DCS for the energy x.yyyEzz (E format, in eV) in a format
C  ready for visualization with a plotting program. The output files
C  'scfield.dat' and 'scatamp.dat' contain tables of the scattering
C  potential and the scattering amplitudes of the last calculated case;
C  they are overwritten (i.e. lost) when a new case is calculated.
C
C     This program uses the subroutine packages 'elsepa2020.f' and
C  'radial.f', which are inserted into this source file by means of
C  two INCLUDE statements (see above).
C
C
      USE CONSTANTS
C
      IMPLICIT DOUBLE PRECISION (A-B,D-H,O-Z), COMPLEX*16 (C),
     1   INTEGER*4 (I-N)
      CHARACTER*6 KWORD
      CHARACTER*12 BUFFER,OFILE
C
      PARAMETER (A0B2=A0B*A0B)
      PARAMETER (PI=3.1415926535897932D0)
C  ****  Results from the partial wave calculation.
      PARAMETER (NGT=650)
      COMMON/DCSTAB/ECS,TCS1,TCS2,TH(NGT),XT(NGT),DCST(NGT),SPOL(NGT),
     1              ERROR(NGT),NTAB
      COMMON/CTOTCS/TOTCS,ABCS
C
C  ****  Atomic polarizabilities of free atoms (in cm**3), from
C    Thomas M. Miller, 'Atomic and molecular polarizabilities' in
C    CRC Handbook of Chemistry and Physics, Editor-in-chief David
C    R. Linde. 79th ed., 1998-1999, pp. 10-160 to 10-174.
      DIMENSION ATPOL(103)
      DATA ATPOL/   0.666D-24, 0.205D-24, 24.30D-24, 5.600D-24,
     A   3.030D-24, 1.760D-24, 1.100D-24, 0.802D-24, 0.557D-24,
     1   3.956D-25, 24.08D-24, 10.06D-24, 6.800D-24, 5.380D-24,
     A   3.630D-24, 2.900D-24, 2.180D-24, 1.641D-24, 43.40D-24,
     2   22.80D-24, 17.80D-24, 14.60D-24, 12.40D-24, 11.60D-24,
     A   9.400D-24, 8.400D-24, 7.500D-24, 6.800D-24, 6.100D-24,
     3   7.100D-24, 8.120D-24, 6.070D-24, 4.310D-24, 3.770D-24,
     A   3.050D-24, 2.484D-24, 47.30D-24, 27.60D-24, 22.70D-24,
     4   17.90D-24, 15.70D-24, 12.80D-24, 11.40D-24, 9.600D-24,
     A   8.600D-24, 4.800D-24, 7.200D-24, 7.200D-24, 10.20D-24,
     5   7.700D-24, 6.600D-24, 5.500D-24, 5.350D-24, 4.044D-24,
     A   59.60D-24, 39.70D-24, 31.10D-24, 29.60D-24, 28.20D-24,
     6   31.40D-24, 30.10D-24, 28.80D-24, 27.70D-24, 23.50D-24,
     A   25.50D-24, 24.50D-24, 23.60D-24, 22.70D-24, 21.80D-24,
     7   21.00D-24, 21.90D-24, 16.20D-24, 13.10D-24, 11.10D-24,
     A   9.700D-24, 8.500D-24, 7.600D-24, 6.500D-24, 5.800D-24,
     8   5.100D-24, 7.600D-24, 6.800D-24, 7.400D-24, 6.800D-24,
     A   6.000D-24, 5.300D-24, 48.70D-24, 38.30D-24, 32.10D-24,
     9   32.10D-24, 25.40D-24, 24.90D-24, 24.80D-24, 24.50D-24,
     A   23.30D-24, 23.00D-24, 22.70D-24, 20.50D-24, 19.70D-24,
     1   23.80D-24, 18.20D-24, 17.50D-24, 20.00D-24/
C
C  ****  Ionization energies of neutral atoms (in eV).
C        NIST Physical Reference Data.
C        http://sed.nist.gov/PhysRefData/IonEnergy/tblNew.html
C  For astatine (Z=85), the value given below was calculated with
C  the DHFXA code (Salvat and Fernandez-Varea, UBIR-2003).
      DIMENSION EIONZ(103)
      DATA EIONZ/  13.5984D0, 24.5874D0, 5.39170D0, 9.32270D0,
     A  8.29800D0, 11.2603D0, 14.5341D0, 13.6181D0, 17.4228D0,
     1  21.5646D0, 5.13910D0, 7.64620D0, 5.98580D0, 8.15170D0,
     A  10.4867D0, 10.3600D0, 12.9676D0, 15.7596D0, 4.34070D0,
     2  6.11320D0, 6.56150D0, 6.82810D0, 6.74620D0, 6.76650D0,
     A  7.43400D0, 7.90240D0, 7.88100D0, 7.63980D0, 7.72640D0,
     3  9.39420D0, 5.99930D0, 7.89940D0, 9.78860D0, 9.75240D0,
     A  11.8138D0, 13.9996D0, 4.17710D0, 5.69490D0, 6.21710D0,
     4  6.63390D0, 6.75890D0, 7.09240D0, 7.28000D0, 7.36050D0,
     A  7.45890D0, 8.33690D0, 7.57620D0, 8.99380D0, 5.78640D0,
     5  7.34390D0, 8.60840D0, 9.00960D0, 10.4513D0, 12.1298D0,
     A  3.89390D0, 5.21170D0, 5.57690D0, 5.53870D0, 5.47300D0,
     6  5.52500D0, 5.58200D0, 5.64360D0, 5.67040D0, 6.15010D0,
     A  5.86380D0, 5.93890D0, 6.02150D0, 6.10770D0, 6.18430D0,
     7  6.25420D0, 5.42590D0, 6.82510D0, 7.54960D0, 7.86400D0,
     A  7.83350D0, 8.43820D0, 8.96700D0, 8.95870D0, 9.22550D0,
     8  10.4375D0, 6.10820D0, 7.41670D0, 7.28560D0, 8.41700D0,
     A  9.50000D0, 10.7485D0, 4.07270D0, 5.27840D0, 5.17000D0,
     9  6.30670D0, 5.89000D0, 6.19410D0, 6.26570D0, 6.02620D0,
     A  5.97380D0, 5.99150D0, 6.19790D0, 6.28170D0, 6.42000D0,
     1  6.50000D0, 6.58000D0, 6.65000D0, 4.90000D0/
C
C  ****  First excitation energies of neutral atoms (in eV).
C        NIST Physical Reference Data.
C  The value 0.0D0 indicates that the experimental value for
C  the atom was not available.
      DIMENSION EEX1Z(103)
      DATA EEX1Z/ 10.20D0, 19.82D0,  1.85D0,  2.73D0,
     A    3.58D0,  1.26D0,  2.38D0,  1.97D0, 12.70D0,
     1   16.62D0,  2.10D0,  2.71D0,  3.14D0,  0.78D0,
     A    1.41D0,  1.15D0,  8.92D0, 11.55D0,  1.61D0,
     2    1.88D0,  1.43D0,  0.81D0,  0.26D0,  0.94D0,
     A    2.11D0,  0.86D0,  0.43D0,  0.01D0,  1.38D0,
     3    4.00D0,  3.07D0,  0.88D0,  1.31D0,  1.19D0,
     A    7.87D0,  9.91D0,  0.00D0,  0.00D0,  0.00D0,
     4    0.00D0,  0.00D0,  1.34D0,  0.00D0,  0.00D0,
     A    0.00D0,  0.00D0,  0.00D0,  3.73D0,  0.00D0,
     5    0.00D0,  0.00D0,  0.00D0,  0.00D0,  8.31D0,
     A    0.00D0,  0.00D0,  0.00D0,  0.00D0,  0.00D0,
     6    0.00D0,  0.00D0,  0.00D0,  0.00D0,  0.00D0,
     A    0.00D0,  0.00D0,  0.00D0,  0.00D0,  0.00D0,
     7    0.00D0,  0.00D0,  0.00D0,  0.00D0,  0.00D0,
     A    0.00D0,  0.00D0,  0.00D0,  0.00D0,  0.00D0,
     8    4.67D0,  0.00D0,  0.00D0,  0.00D0,  0.00D0,
     A    0.00D0,  0.00D0,  0.00D0,  0.00D0,  0.00D0,
     9    0.00D0,  0.00D0,  0.00D0,  0.00D0,  0.00D0,
     A    0.00D0,  0.00D0,  0.00D0,  0.00D0,  0.00D0,
     1    0.00D0,  0.00D0,  0.00D0,  0.00D0/
C
C  ****  Nearest-neighbour distances (in cm) of the elements,
C    from Ch. Kittel, 'Introduction to Solid State Physics'. 5th
C    ed. (John Wiley and Sons, New York, 1976).
C  The value -1.0D-8 indicates that the experimental value for
C  the element was not available.
      DIMENSION DNNEL(103)
      DATA DNNEL/    -1.000D-8,-1.000D-8, 3.124D-8, 2.220D-8,
     A     -1.000D-8, 1.540D-8,-1.000D-8,-1.000D-8, 1.440D-8,
     1     -1.000D-8, 3.822D-8, 3.200D-8, 2.860D-8, 2.350D-8,
     A     -1.000D-8,-1.000D-8,-1.000D-8,-1.000D-8, 4.752D-8,
     2      3.950D-8, 3.250D-8, 2.890D-8, 2.620D-8, 2.500D-8,
     A      2.240D-8, 2.480D-8, 2.500D-8, 2.490D-8, 2.560D-8,
     3      2.660D-8, 2.440D-8, 2.450D-8, 3.160D-8, 2.320D-8,
     A     -1.000D-8,-1.000D-8, 4.837D-8, 4.300D-8, 3.550D-8,
     4      3.170D-8, 2.860D-8, 2.720D-8, 2.710D-8, 2.650D-8,
     A      2.690D-8, 2.750D-8, 2.890D-8, 2.980D-8, 3.250D-8,
     5      2.810D-8, 2.910D-8, 2.860D-8, 3.540D-8,-1.000D-8,
     A      5.235D-8, 4.350D-8, 3.730D-8, 3.650D-8, 3.630D-8,
     6      3.660D-8,-1.000D-8, 3.590D-8, 3.960D-8, 3.580D-8,
     A      3.520D-8, 3.510D-8, 3.490D-8, 3.470D-8, 3.540D-8,
     7      3.880D-8, 3.430D-8, 3.130D-8, 2.860D-8, 2.740D-8,
     A      2.740D-8, 2.680D-8, 2.710D-8, 2.770D-8, 2.880D-8,
     8     -1.000D-8, 3.460D-8, 3.500D-8, 3.070D-8, 3.340D-8,
     A     -1.000D-8,-1.000D-8,-1.000D-8,-1.000D-8, 3.760D-8,
     9      3.600D-8, 3.210D-8, 2.750D-8, 2.620D-8, 3.100D-8,
     A      3.610D-8,-1.000D-8,-1.000D-8,-1.000D-8,-1.000D-8,
     1     -1.000D-8,-1.000D-8,-1.000D-8,-1.000D-8/
C
C  ****  Atomic densities (number of atoms per cm**3) of the
C    elements in their natural state (from the PENELOPE database).
      DIMENSION VMOLE(103)
      DATA VMOLE/    5.0039E+19, 2.5024E+19, 4.6331E+22, 1.2349E+23,
     A   1.3202E+23, 1.0028E+23, 5.0100E+19, 5.0119E+19, 5.0093E+19,
     1   2.5024E+19, 2.5435E+22, 4.3113E+22, 6.0237E+22, 4.9959E+22,
     A   4.2774E+22, 3.7561E+22, 5.0869E+19, 2.5055E+19, 1.3277E+22,
     2   2.3290E+22, 4.0040E+22, 5.7102E+22, 7.2230E+22, 8.3158E+22,
     A   8.1555E+22, 8.4908E+22, 9.0946E+22, 9.1343E+22, 8.4912E+22,
     3   6.5692E+22, 5.0994E+22, 4.4148E+22, 4.6057E+22, 3.4321E+22,
     A   5.3301E+19, 2.4996E+19, 1.0795E+22, 1.7457E+22, 3.0271E+22,
     4   4.2949E+22, 5.5551E+22, 6.4151E+22, 7.0735E+22, 7.3944E+22,
     A   7.2621E+22, 6.8019E+22, 5.8619E+22, 4.6341E+22, 3.8340E+22,
     5   3.7084E+22, 3.3096E+22, 2.9450E+22, 2.3396E+22, 2.5161E+19,
     A   8.4865E+21, 1.5348E+22, 2.6679E+22, 2.8611E+22, 2.8677E+22,
     6   2.8808E+22, 3.0005E+22, 2.9878E+22, 2.0778E+22, 3.0256E+22,
     A   3.1181E+22, 3.1686E+22, 3.2113E+22, 3.2642E+22, 3.3228E+22,
     7   2.3422E+22, 3.3867E+22, 4.4907E+22, 5.5426E+22, 6.3219E+22,
     A   6.7980E+22, 7.1461E+22, 7.0241E+22, 6.6216E+22, 5.9069E+22,
     8   4.0668E+22, 3.4533E+22, 3.2988E+22, 2.8088E+22, 2.6857E+22,
     A   2.6728E+22, 2.4591E+19, 2.7003E+21, 1.3322E+22, 2.6711E+22,
     9   3.0417E+22, 4.0062E+22, 4.7943E+22, 5.1444E+22, 4.9981E+22,
     A   3.3869E+22, 3.2930E+22, 3.4124E+22, 3.3579E+22, 3.3446E+22,
     1  -1.0000D+00,-1.0000D+00,-1.0000D+00,-1.0000D+00/
C
C  ****  Phase shifts.
      PARAMETER (NDM=25000)
      COMMON/PHASES/DP(NDM),DM(NDM),NPH,ISUMP
      COMMON/PHASEI/DPJ(NDM),DMJ(NDM)
C
C  ************  Input data.
C
C  ****  Default model.
      IELEC =-1        ! electron
      IZ    = 0        ! no default
      NELEC = 1000     ! =Z (the present value is a flag)
      MNUCL = 3        ! Fermi nuclear charge distribution
      MELEC = 4        ! DF electron density
      MUFFIN= 0        ! free atom
      RMUF  =-1.0D0    ! free atom
      VMOL  =-1.0D0  ! free atom
      MEXCH = 1        ! FM exchange potential
      MCPOL = 0        ! no correlation-polarization
      VPOLA =-1.0D0    ! atomic polarizability
      VPOLB =-1.0D0    ! polariz. cutoff parameter
      MABS  = 0        ! no absorption
      VABSA =-1.0D0    ! absorption potential strength
      VABSD =-1.0D0    ! energy gap
      IHEF  = 1        ! high-energy factorization on
C
  100 CONTINUE
      READ(5,'(A6,1X,A12)') KWORD,BUFFER
      IF(KWORD.EQ.'IZ    ') THEN
        READ(BUFFER,*) IZ
        IF(IZ.LT.1.OR.IZ.GT.103) THEN
          WRITE(6,*) 'IZ =',IZ
          STOP 'Wrong atomic number.'
        ENDIF
      ELSE IF(KWORD.EQ.'MNUCL ') THEN
        READ(BUFFER,*) MNUCL
        IF(MNUCL.LT.1.OR.MNUCL.GT.4) MNUCL=3
      ELSE IF(KWORD.EQ.'NELEC ') THEN
        READ(BUFFER,*) NELEC
        IF(NELEC.LT.0) NELEC=IZ
        IF(NELEC.GT.IZ) THEN
          WRITE(6,*) 'NELEC =',NELEC
          STOP 'Negative ion.'
        ENDIF
      ELSE IF(KWORD.EQ.'MELEC ') THEN
        READ(BUFFER,*) MELEC
        IF(MELEC.LT.1.OR.MELEC.GT.5) MELEC=4
      ELSE IF(KWORD.EQ.'MUFFIN') THEN
        READ(BUFFER,*) MUFFIN
        IF(MUFFIN.NE.1) MUFFIN=0
      ELSE IF(KWORD.EQ.'RMUF  ') THEN
        READ(BUFFER,*) RMUF
      ELSE IF(KWORD.EQ.'IELEC ') THEN
        READ(BUFFER,*) IELEC
        IF(IELEC.NE.+1) IELEC=-1
      ELSE IF(KWORD.EQ.'MEXCH ') THEN
        READ(BUFFER,*) MEXCH
        IF(MEXCH.LT.0.OR.MEXCH.GT.3) MEXCH=1
      ELSE IF(KWORD.EQ.'MCPOL ') THEN
        READ(BUFFER,*) MCPOL
        IF(MCPOL.LT.0.OR.MCPOL.GT.2) MCPOL=0
      ELSE IF(KWORD.EQ.'VPOLA ') THEN
        READ(BUFFER,*) VPOLA
        IF(VPOLA.LT.-1.0D-35) VPOLA=ATPOL(IZ)
      ELSE IF(KWORD.EQ.'VPOLB ') THEN
        READ(BUFFER,*) VPOLB
        IF(VPOLB.LT.1.0D-10) VPOLB=-10.0D0
      ELSE IF(KWORD.EQ.'MABS  ') THEN
        READ(BUFFER,*) MABS
        IF(MABS.LT.0.OR.MABS.GT.2) MABS=0
      ELSE IF(KWORD.EQ.'VABSA ') THEN
        READ(BUFFER,*) VABSAI
        IF(VABSAI.GT.0.0D0) VABSA=VABSAI
      ELSE IF(KWORD.EQ.'VABSD ') THEN
        READ(BUFFER,*) VABSD
        IF(VABSD.LT.-1.0D-35) VABSD=-1.0D0
      ELSE IF(KWORD.EQ.'IHEF') THEN
        READ(BUFFER,*) IHEF
        IF(IHEF.NE.0.AND.IHEF.NE.2) IHEF=1
      ELSE IF(KWORD.EQ.'EV    ') THEN
        IF(VMOL.LT.-1.0D-35) VMOL=VMOLE(IZ)
        GO TO 200
      ELSE
        WRITE(6,*) 'Unrecognized keyword.'
        GO TO 100
      ENDIF
      GO TO 100
C
C  ****  Potential model parameters.
C
  200 CONTINUE
      IF(NELEC.EQ.1000) NELEC=IZ
C
      IF(NELEC.NE.IZ) MUFFIN=0
      IF(MUFFIN.EQ.1) THEN
        IF(RMUF.LT.1.0D-9) RMUF=0.5D0*DNNEL(IZ)
        IF(RMUF.LT.1.0D-9) THEN
          WRITE(6,*) 'RMUF = ',RMUF
          STOP 'RMUF is too small.'
        ENDIF
        VMOL=VMOLE(IZ)
        IF(VMOL.LT.1.0D0) VMOL=1.0D0/(2.0D0*RMUF)**3
C  ****  If the volume of the muffin-tin sphere is larger than that of
C  the Wigner-Seitz cell (= 1/VMOL), a cubic lattice is assumed.
        RWS=(0.75D0/(PI*VMOL))**(1.0D0/3.0D0)
        IF(RMUF.GT.RWS) THEN
          VMOL=1.0D0/(2.0D0*RMUF)**3
        ENDIF
        IHEF=0
      ELSE
        MUFFIN=0
        RMUF=200.0D-8
      ENDIF
C
      IF(IELEC.EQ.+1) MEXCH=0
C
      IF(MCPOL.EQ.1) THEN
        IF(VPOLA.LT.-1.0D-35) VPOLA=ATPOL(IZ)  ! Default value.
        IF(VPOLA.LT.-1.0D-35) THEN
          WRITE(6,*) 'VPOLA = ',VPOLA
          STOP 'VPOLA must be positive.'
        ENDIF
        IF(VPOLB.LT.1.0D-10) VPOLB=-10.0D0
      ELSE IF(MCPOL.EQ.2) THEN
        IF(VPOLA.LT.-1.0D-35) VPOLA=ATPOL(IZ)
        IF(VPOLA.LT.-1.0D-35) THEN
          WRITE(6,*) 'VPOLA = ',VPOLA
          STOP 'VPOLA must be positive.'
        ENDIF
        IF(VPOLB.LT.1.0D-10) VPOLB=-10.0D0
      ELSE
        MCPOL=0
        VPOLA=0.0D0
        VPOLB=0.0D0
      ENDIF
C
      IF(MABS.EQ.1.OR.MABS.EQ.2) THEN
        IF(VABSA.LT.-1.0D-35) THEN
          IF(MABS.EQ.1) THEN
            VABSA=2.0D0
          ELSE
            VABSA=0.75D0
          ENDIF
        ENDIF
        IF(VABSD.LT.-1.0D-35) THEN
          IF(IELEC.EQ.-1) THEN  ! Default (experimental) value.
            VABSD=EEX1Z(IZ)
            IF(VABSD.LT.-1.0D-35) VABSD=0.0D0
          ELSE
            VABSD=MAX(0.0D0,EIONZ(IZ)-6.8D0)
          ENDIF
        ENDIF
      ELSE
        MABS=0
        VABSA=0.0D0
        VABSD=1.0D0
      ENDIF
C
      OPEN(25,FILE='tcstable.dat')
      WRITE(25,1001)
 1001 FORMAT(1X,'#  Total cross section table (last run of',
     1  ' ''elscata'').', /1X,'#')
      IF(IELEC.EQ.-1) THEN
        WRITE(25,1002) IZ
 1002   FORMAT(1X,'#  Z =',I4,',   projectile: electron')
      ELSE
        WRITE(25,1003) IZ
 1003   FORMAT(1X,'#  Z =',I4,',   projectile: positron')
      ENDIF
      IF((MUFFIN.EQ.1).AND.(MABS.EQ.1.OR.MABS.EQ.2)) THEN
        WRITE(25,1004)
 1004   FORMAT(1X,'#',/1X,'#   Energy',9X,'ECS',9X,'TCS1',9X,'TCS2',
     1    9X,'ABSCS',6X,'inel-MFP',5X,'error',/1X,'#',4X,'(eV)',
     2    8X,'(cm**2)',6X,'(cm**2)',6X,'(cm**2)',6X,'(cm**2)',
     3    7X,'(cm)',/1X,'#',87('-'))
      ELSE
        WRITE(25,1005)
 1005   FORMAT(1X,'#',/1X,'#   Energy',9X,'ECS',9X,'TCS1',9X,'TCS2',
     1    9X,'ABSCS',6X,'error',/1X,'#',4X,'(eV)',8X,'(cm**2)',6X,
     2    '(cm**2)',6X,'(cm**2)',6X,'(cm**2)',/1X,'#',74('-'))
      ENDIF
C
C  ************  Partial-wave analysis.
C
  300 CONTINUE
      READ(BUFFER,*) EV
      WRITE(6,*) '   '
      WRITE(6,'(A,1P,E15.7)') 'E (eV)=',EV
C  ****  You may wish to comment off the next condition to run the
C  program for kinetic energies less that 5 eV. However, the results
C  for these energies may be highly inaccurate.
      IF(EV.LT.4.999D0) STOP 'The kinetic energy is too small.'
C
      IF(MCPOL.NE.0) THEN
        IF(EV.GT.1.0D4) THEN
          WRITE(6,*) 'WARNING: For E>10 keV the correlation-polari',
     1      'zation correction is'
          WRITE(6,*) '         switched off.'
          MCPOLC=0
          VPOLBC=0.0D0
        ELSE
          MCPOLC=MCPOL
          IF(VPOLB.LT.0.0D0) THEN
            VPOLBC=SQRT(MAX((EV-50.0D0)/16.0D0,1.0D0))
          ELSE
            VPOLBC=VPOLB
          ENDIF
        ENDIF
      ELSE
        MCPOLC=0
        VPOLBC=0.0D0
      ENDIF
      IF(MABS.NE.0) THEN
        IF(EV.GT.1.0D6) THEN
          WRITE(6,*) 'WARNING: For E>1 MeV, the absorption correc',
     1      'tion is switched off.'
          MABSC=0
        ELSE
          MABSC=MABS
        ENDIF
      ELSE
        MABSC=0
      ENDIF
C
      WRITE(BUFFER,'(1P,E12.5)') EV
      OFILE=BUFFER(2:2)//'p'//BUFFER(4:6)//'e'//BUFFER(11:12)
      OPEN(8,FILE='dcs_'//OFILE(1:8)//'.dat')
      CALL ELSEPA(IELEC,EV,IZ,NELEC,MNUCL,MELEC,MUFFIN,RMUF,VMOL,
     1  MEXCH,MCPOLC,VPOLA,VPOLBC,MABSC,VABSA,VABSD,IHEF,8)
      CLOSE(8)
C
      ERRM=0.0D0
      OPEN(8,FILE='dcs.dat')
      WRITE(8,'(1X,''#'')')
      WRITE(8,'(1X,''# Differential cross section:'',15X,
     1  ''MU=(1-COS(THETA))/2'')')
      WRITE(8,'(1X,''#'',/1X,''#  THETA'',8X,''MU'',10X,''DCS'',
     1  10X,''DCS'',8X,''Sherman''/1X,''#  (deg)'',
     2  17X,''(cm**2/sr)'',3X,''(a0**2/sr)'',4X,''function'',
     3  /1X,''#'',62(''-''))')
      DO I=1,NTAB
        ERRM=MAX(ERRM,ERROR(I))
        WRITE(8,1006) TH(I),XT(I),DCST(I),DCST(I)/A0B2,SPOL(I)
 1006   FORMAT(1X,1P,E10.3,E13.5,3E13.5,E9.1)
      ENDDO
      CLOSE(8)
C
      IF((MUFFIN.EQ.1).AND.(MABS.EQ.1.OR.MABS.EQ.2)) THEN
        AMFP=1.0D0/(VMOL*ABCS)
        WRITE(25,1007) EV,ECS,TCS1,TCS2,ABCS,AMFP,ERRM
 1007   FORMAT(1X,1P,6E13.5,E9.1)
      ELSE
        WRITE(25,1008) EV,ECS,TCS1,TCS2,ABCS,ERRM
 1008   FORMAT(1X,1P,5E13.5,E9.1)
      ENDIF
C
      IF(MUFFIN.EQ.0.AND.MCPOL.EQ.0.AND.MABSC.EQ.0) THEN
        OPEN(8,FILE='dcsMott.dat')
        CALL MOTTSC(IELEC,IZ,EV,8)
        CLOSE(8)
      ENDIF
C
 400  CONTINUE
      READ(5,'(A6,1X,A12)',END=9999) KWORD,BUFFER
      IF(KWORD.EQ.'EV    ') GO TO 300
      GO TO 400
C
 9999 CONTINUE
      CLOSE(25)
      END
