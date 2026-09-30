C
C     **************************
C     **    Program SBETHE    **
C     **************************
C
C     Calculates the electronic stopping power of materials for fast
C  charged particles from the corrected Bethe formula with the DHFS
C  shell correction and the density-effect correction. In the case of
C  projectiles much heavier than the electron, the Lindhard-Sorensen
C  correction and the Barkas correction are included. The density-
C  effect and the Barkas corrections are computed from the DHFS-model
C  OOS of the material, which is determined by the adopted I value.
C  For projectile electrons and positrons, the program also provides
C  the radiative stopping power calculated from the tables of electron
C  bremmstrahlung atomic cross sections of Seltzer and Berger.
C
C     The composition and physical parameters of the material are
C  entered by the user in response to the prompts from the program.
C
C     The file 'pdcompos.pen' in the subdirectory './sdbase' contains
C  composition data and physical parameters for 280 materials, taken
C  from the database of the ESTAR program of Berger (NISTIR 4999, 1992).
C  The first 99 entries are the elements Z=1-99, ordered by atomic
C  number Z. Materials 100 to 280 are compounds and mixtures, in
C  alphabetical order. For these materials, the I value assigned
C  tentatively by the program is read from the 'pdcompos.pen' file; for
C  other materials, the proposed I value is estimated by means of the
C  additivity approximation. The user is allowed to modify the proposed
C  I value.
C
C     The corrected Bethe formula is valid only for projectiles with
C  energies higher than ECUT (about 1 keV, 150 keV, 0.75 MeV and 5.0 MeV
C  for electrons/positrons, muons/antimuons, protons/antiprotons, and
C  alpha particles, respectively). For energies less than ECUT, the
C  stopping power is estimated from an empirical extrapolation of the
C  corrected Bethe formula. The generated table of the stopping power as
C  a function of the kinetic energy of the projectile is used to
C  evaluate the CSDA range of projectile particles, and the CSDA depth-
C  dose distribution from particle beams with various initial energies
C  (disregaring the effect of elastic scattering).
C
C                               Francesc Salvat, Barcelona, April 2024.
C


C  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      MODULE CONSTANTS  ! Physical constants.
      SAVE  ! Saves all items in the module.
C  ----  Speed of light (1/alpha).
      DOUBLE PRECISION, PARAMETER :: SL=137.035999084D0
C  ----  Bohr radius (cm).
      DOUBLE PRECISION, PARAMETER :: A0B=5.29177210903D-9
C  ----  Hartree energy (eV).
      DOUBLE PRECISION, PARAMETER :: HREV=27.211386245988D0
C  ----  Electron rest energy (eV).
      DOUBLE PRECISION, PARAMETER :: REV=510.9989500D3
C  ----  Avogadro's number (1/mol).
      DOUBLE PRECISION, PARAMETER :: AVOG=6.02214076D23
C  ----  Classical electron radius (cm).
      DOUBLE PRECISION, PARAMETER :: RECL=2.8179403262D-13
      END MODULE CONSTANTS
C  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


C  *********************************************************************
C                       MAIN PROGRAM
C  *********************************************************************
      USE CONSTANTS
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      CHARACTER BUFFER*12,CZS*2
      CHARACTER CPART*10,MNAME*15
C  ****  Composition data.
      COMMON/COMPOS/STF(30),AW,IZ(30),NELEM
C  ****  OOS and dielectric function tables.
      PARAMETER (NR=750)
      COMMON/COOST/ZT,EXPOT,RHO,VMOL,OMEGA2,WOD,ZTOT
      COMMON/CETAS/WP(NR),FT(NR),SFT(NR),WL(NR),FL(NR),ETA2(NR),ETA1(NR)
C  ****  Energy mesh points.
      PARAMETER (NEGP=1500)
      COMMON/CEGRID/ET(NEGP),STPE(NEGP),STPR(NEGP),RANGE(NEGP),NE
      DIMENSION A0(NEGP),B0(NEGP),C0(NEGP),D0(NEGP),STPT(NEGP)
      DIMENSION A1(NEGP),B1(NEGP),C1(NEGP),D1(NEGP),RR(NEGP),ENT(NEGP)
      DIMENSION A2(NEGP),B2(NEGP),C2(NEGP),D2(NEGP)
C  ****  Element data.
      CHARACTER LASYMB*2
      COMMON/CADATA/ATW(99),EPX(99),RSCR(99),ETA(99),EB(99,30),
     1  ALW(99,30),CP0(99,30),IFI(99,30),IKS(99,30),NSHT(99),LASYMB(99)
C  ****  Kinematical parameters.
      COMMON/PKINEM/RMASEV,ECC,ETT,GAMM1,GAMMA,GAMMA2,BETA2,BETA,
     1  CP2,CP,CONS,ZEF,ICHAR
C
      COMMON/CBRIDGE/PMASS,WMAX,IZT,IZ1
      COMMON/CBETHE/RL0,FCOR,CSHC,CDEN,CLS,CBAR
C
      DIMENSION S0T(99)
      DATA S0T   /   1.00000E+00, 1.99981E+00, 2.99933E+00, 3.99866E+00,
     A  4.99782E+00, 5.99663E+00, 6.99510E+00, 7.99323E+00, 8.99098E+00,
     1  9.98835E+00, 1.09853E+01, 1.19820E+01, 1.29783E+01, 1.39743E+01,
     A  1.49698E+01, 1.59649E+01, 1.69594E+01, 1.79535E+01, 1.89466E+01,
     2  1.99402E+01, 2.09331E+01, 2.19254E+01, 2.29173E+01, 2.39087E+01,
     A  2.48996E+01, 2.58900E+01, 2.68799E+01, 2.78693E+01, 2.88581E+01,
     3  2.98463E+01, 3.08340E+01, 3.18211E+01, 3.28077E+01, 3.37937E+01,
     A  3.47793E+01, 3.57643E+01, 3.67476E+01, 3.77331E+01, 3.87169E+01,
     4  3.96998E+01, 4.06822E+01, 4.16640E+01, 4.26452E+01, 4.36258E+01,
     A  4.46058E+01, 4.55852E+01, 4.65640E+01, 4.75423E+01, 4.85195E+01,
     5  4.94961E+01, 5.04721E+01, 5.14474E+01, 5.24220E+01, 5.33957E+01,
     A  5.43671E+01, 5.53465E+01, 5.63837E+01, 5.73460E+01, 5.83066E+01,
     6  5.92663E+01, 6.02255E+01, 6.11844E+01, 6.21430E+01, 6.31112E+01,
     A  6.40787E+01, 6.50456E+01, 6.60117E+01, 6.69770E+01, 6.79416E+01,
     7  6.89048E+01, 6.98687E+01, 7.08314E+01, 7.17933E+01, 7.27544E+01,
     A  7.37147E+01, 7.46742E+01, 7.56329E+01, 7.65910E+01, 7.75482E+01,
     8  7.85045E+01, 7.94595E+01, 8.04138E+01, 8.13671E+01, 8.23196E+01,
     A  8.32711E+01, 8.42217E+01, 8.51719E+01, 8.61033E+01, 8.70819E+01,
     9  8.80604E+01, 8.90154E+01, 8.99581E+01, 9.08935E+01, 9.18245E+01,
     A  9.27528E+01, 9.36957E+01, 9.46376E+01, 9.55786E+01, 9.65185E+01/
C
C  ****  Other arrays.
      PARAMETER (NT=5000)
      DIMENSION W0(NT),F0(NT)
      COMMON/CSHCOR/ESHC(NT),SHC(NT),NSHC
      DIMENSION AUX1(NT),AUX2(NT),X1(NT),X2(NT)
      DIMENSION E0T(10),ZMAX(10),DDOSE(10)
C
      PARAMETER (NEGRT=66)
      DIMENSION EGRID(NEGRT)
      DATA EGRID/1.00D0,1.05D0,1.10D0,1.15D0,1.20D0,1.25D0,1.30D0,
     1    1.35D0,1.40D0,1.45D0,1.50D0,1.60D0,1.70D0,1.80D0,1.90D0,
     2    2.00D0,2.10D0,2.20D0,2.30D0,2.40D0,2.50D0,2.60D0,2.70D0,
     3    2.80D0,2.90D0,3.00D0,3.10D0,3.20D0,3.30D0,3.40D0,3.50D0,
     4    3.60D0,3.70D0,3.80D0,3.90D0,4.00D0,4.10D0,4.20D0,4.30D0,
     5    4.40D0,4.50D0,4.60D0,4.70D0,4.80D0,4.90D0,5.00D0,5.25D0,
     6    5.50D0,5.75D0,6.00D0,6.25D0,6.50D0,6.75D0,7.00D0,7.25D0,
     7    7.50D0,7.75D0,8.00D0,8.25D0,8.50D0,8.75D0,9.00D0,9.25D0,
     8    9.50D0,9.750D0,1.00D1/
C
      DIMENSION EGRT(17)
      DATA EGRT/1.0D0,1.25D0,1.50D0,1.75D0,2.00D0,2.50D0,3.00D0,
     1  3.50D0,4.00D0,4.50D0,5.00D0,5.50D0,6.00D0,7.00D0,8.00D0,
     2  9.00D0,1.00D1/
C
C  ****  Define the material.
C
      WRITE(6,*) ' '
      WRITE(6,*) ' Enter the material name (up to 15 chars, ',
     1  'no blanks) ...'
      READ(5,1001) MNAME
 1001 FORMAT(A15)
      CALL STMATW(MNAME,W0,F0,NTAB,ID)
C
C  ****  'Cleaning' the OOS table.
C
      CALL ODTRAN(MNAME,ZT,W0,F0,NTAB)
      IZT=ZT+0.5D0
C
C  ****  Density-effect correction.
C
      CALL DENSIT
C
C  ****  Projectile particle.
C
 10   CONTINUE
      write(6,*) ' '
      WRITE(6,'(/1X,'' Projectile particle (IPROJ)?'')')
      WRITE(6,'(5X,''1=electron,  2=positron,  3=muon,  4=antimuon,'')')
      WRITE(6,'(5X,''5=proton,  6=antiproton,  7=alpha,  8=other  '')')
      READ(5,*) IPROJ
      IF(IPROJ.LT.1.OR.IPROJ.GT.8) THEN
        WRITE(6,'(2X,''Wrong projectile type.'')')
        GO TO 10
      ENDIF
C
      ICHAR=0
      IF(IPROJ.EQ.1) THEN
        CPART='electron  '
        IZ1=-1; PMASS=1.0D0
        ELOW=10.0D0
        ECUT=1.0D3
        EMAX=100.0D0
        EUP=1.0D9
      ELSE IF(IPROJ.EQ.2) THEN
        CPART='positron  '
        IZ1=+1; PMASS=1.0D0
        ELOW=10.0D0
        ECUT=1.0D3
        EMAX=100.0D0
        EUP=1.0D9
      ELSE IF(IPROJ.EQ.3) THEN
        CPART='muon      '
        IZ1=-1; PMASS=206.7682830D0
        ELOW=1.0D3
        ECUT=1.5D5
        EMAX=2.0D4
        EUP=1.0D15
      ELSE IF(IPROJ.EQ.4) THEN
        CPART='antimuon  '
        IZ1=+1; PMASS=206.7682830D0
        ELOW=1.0D3
        ECUT=1.5D5
        EMAX=2.0D4
        EUP=1.0D15
      ELSE IF(IPROJ.EQ.5) THEN
        CPART='proton    '
        IZ1=+1; PMASS=1836.15267343D0
        ELOW=1.0D3
        ECUT=0.75D6
        EMAX=5.0D4
        ICHAR=1
        EUP=1.0D10
      ELSE IF(IPROJ.EQ.6) THEN
        CPART='antiproton'
        IZ1=-1; PMASS=1836.15267343D0
        ELOW=1.0D3
        ECUT=0.75D6
        EMAX=5.0D4
        EUP=1.0D10
      ELSE IF(IPROJ.EQ.7) THEN
        CPART='alpha     '
        IZ1=2; PMASS=7294.29954142D0
        ELOW=5.0D3
        ECUT=5.0D6
        EMAX=8.0D5
        ICHAR=1
        EUP=1.0D10
      ELSE IF(IPROJ.EQ.8) THEN
        WRITE(6,*) ' Enter particle name (up to 10 characters) ...'
        READ(5,'(A10)') CPART
        WRITE(6,*) ' Enter rest energy (eV) ...'
        READ(5,*) RENERG
        PMASS=RENERG/REV
        WRITE(6,*) ' Particle charge (/elementary charge) ...'
        READ(5,*) IZ1
        IF(PMASS.LT.1000.0D0) THEN
          ELOW=5.0D2
        ELSE
          IF(IZ1.GT.1) ICHAR=1
          ELOW=5.0D3
        ENDIF
        ECUT=PMASS*500.0D0
        EMAX=PMASS*50.0D0
        EUP=1.0D10
      ELSE
        STOP 'Incorrect particle number.'
      ENDIF
C
C  ****  Energy mesh points (to produce nice tables).
C
      NE=0
      IGRID=0
      FGRID=10.0D0
 1    IGRID=IGRID+1
      EV=EGRID(IGRID)*FGRID
      IF(IGRID.EQ.NEGRT) THEN
        IGRID=1
        FGRID=10.0D0*FGRID
      ENDIF
      IF(EV.LT.0.99D0) GO TO 1
      NE=NE+1
      ET(NE)=EV
      IF(EV.LT.EUP.AND.NE.LT.NEGP) GO TO 1
C
      CB=1.0D0
      ZAV=0.0D0
      RAT=0.0D0
      DO J=1,NELEM
        ZAV=ZAV+STF(J)*IZ(J)
        RAT=RAT+STF(J)
      ENDDO
      ZAV=ZAV/RAT
      CB=MAX(1.0D0,0.1D0*ZAV)
      CALL KINPAR(IZ1,PMASS,1.0D3)  ! Needed at initialization.
C
      OPEN(25, FILE='gnuinfo.dat')
      WRITE(25,'(A,A)') ' cpart="',TRIM(CPART)//'"'
      WRITE(25,*) 'mname="'//TRIM(MNAME)//'"'
      CLOSE(25)
C
C  ****  Asymptotic formulas for the total cross section, the stopping
C        cross section and the energy-straggling cross section of
C        neutral DHFS atoms (uncorrected).
C
      OPEN(12,FILE='asymptotic.dat')
      WRITE(12,1002) CPART,PMASS*REV,IZ1,MNAME,ZT,AW,RHO
 1002 FORMAT('#  Asymptotic formulas for integrated cross sections.'
     1  /,'#',/,'#  Projectile particle: ',A10,'   rest energy =',
     2  1P,E12.5,' eV',/,'#',41X,'charge =',I3,' e',/,'#',
     3  /,'#  Material filename:   ',A,
     4  /,'#  Electrons/molecule ........... ',1P,E12.5,
     5  /,'#  Molecular weight ............. ',E12.5,
     6  /,'#  Density ...................... ',E12.5,' g/cm^3',/,'#')
      WRITE(12,1005)
 1005 FORMAT('#  1 mtu = 1 g/cm^2',/,'#')
      WRITE(12,1003) VMOL/RHO
 1003 FORMAT('#  N_Avog/A ..................... ',1P,E12.5,' 1/g',
     1  /,'#')
      WRITE(12,1004)
 1004 FORMAT('#   Energy',7X,'sigma^0',6X,'sigma^1',6X,'sigma^2',
     1  8X,'MFP',7X,'Stopping',4X,'Straggling',/,'#',4X,'(eV)',
     2  8X,'(cm^2)',6X,'(eV*cm^2)',3X,'((eV*cm)^2)',
     3  5X,'(mtu)',6X,'(MeV/mtu)',3X,'(MeV^2/mtu)',/,'# ',90('-'))
      DO IE=1,NE
        E=ET(IE)
        IF(E.GT.ELOW-1.0D0) THEN
          CS0=0.0D0; CS1=0.0D0; CS2=0.0D0
          DO I=1,NELEM
            CALL ASACSS(E,PMASS,CS0A,CS1A,CS2A,IZ1,IZ(I))
            CS0=CS0+STF(I)*CS0A
            CS1=CS1+STF(I)*CS1A
            CS2=CS2+STF(I)*CS2A
          ENDDO
          IF(CS0.GT.1.0D-35) THEN
            RMFP=RHO/(VMOL*CS0)
          ELSE
            RMFP=1.0D35
          ENDIF
          IF(CS1.GT.1.0D-35) THEN
            RSTP=1.0D-6*VMOL*CS1/RHO
          ELSE
            RSTP=1.0D-35
          ENDIF
          IF(CS2.GT.1.0D-35) THEN
            RSTR=1.0D-12*VMOL*CS2/RHO
          ELSE
            RSTR=1.0D-35
          ENDIF
          WRITE(12,'(1P,10E13.5)') E,CS0,CS1,CS2,RMFP,RSTP,RSTR
        ENDIF
      ENDDO
      CLOSE(12)
C
C  ****  Read the DHFS shell-correction tables, and include the
C        relativistic correction.
C
      DO JZ=1,NELEM
        IZE=IZ(JZ)
        WRITE(BUFFER,'(1P,I3)') IZE
        IF(IZE.LE.9) THEN
          CZS=''//'0'//BUFFER(3:3)//''
        ELSE
          CZS=BUFFER(2:3)
        ENDIF
        IF(IZE.LT.100) THEN
          IF(IPROJ.EQ.1) THEN
            OPEN(15,FILE='./sdbase/eshcor-'//CZS//'.tab')
          ELSE IF(IPROJ.EQ.2) THEN
            OPEN(15,FILE='./sdbase/pshcor-'//CZS//'.tab')
          ELSE
            OPEN(15,FILE='./sdbase/shcor-'//CZS//'.tab')
          ENDIF
        ELSE
          STOP 'Wrong atomic number.'
        ENDIF
        IF(JZ.EQ.1) THEN
          NSHC=0
          DO II=1,10000
 101        CONTINUE
            READ(15,*,ERR=101,END=102) GAM1,SHELL
            NSHC=NSHC+1
            RC=DLOG(GAM1*(GAM1+2.0D0))-GAM1*(GAM1+2.0D0)/(GAM1+1.0D0)**2
            SHELL=SHELL+((DBLE(IZE)-S0T(IZE))/(2.0D0*DBLE(IZE)))*RC
            ESHC(NSHC)=GAM1*PMASS*REV
            SHC(NSHC)=STF(JZ)*SHELL
          ENDDO
 102      CONTINUE
          CLOSE(15)
        ELSE
          N1=0
          DO II=1,10000
 103        CONTINUE
            READ(15,*,ERR=103,END=104) GAM1,SHELL
            N1=N1+1
            RC=DLOG(GAM1*(GAM1+2.0D0))-GAM1*(GAM1+2.0D0)/(GAM1+1.0D0)**2
            SHELL=SHELL+((DBLE(IZE)-S0T(IZE))/(2.0D0*DBLE(IZE)))*RC
            X1(N1)=GAM1*PMASS*REV
            X2(N1)=STF(JZ)*SHELL
          ENDDO
 104      CONTINUE
          CLOSE(15)
          CALL MERGE2(X1,X2,ESHC,SHC,AUX1,AUX2,N1,NSHC,NT,N,2)
          NSHC=N
          DO I=1,N
            ESHC(I)=AUX1(I)
            SHC(I)=AUX2(I)
          ENDDO
        ENDIF
      ENDDO
C
      CBF=CB
      WMAX=1.0D35
      OPEN(12,FILE='stplogb.dat')
      WRITE(12,2001) CPART,PMASS*REV,IZ1,MNAME,ZT,AW,RHO,EXPOT,CBF
 2001 FORMAT('#  Corrected Bethe formula. Stopping logarithms.'
     1  /,'#',/,'#  Projectile particle: ',A10,'   rest energy =',
     2  1P,E12.5,' eV',/,'#',41X,'charge =',I3,' e',/,'#',
     3  /,'#  Material filename:   ',A,
     4  /,'#  Electrons/molecule ........... ',1P,E12.5,
     5  /,'#  Molecular weight ............. ',E12.5,
     6  /,'#  Density ...................... ',E12.5,' g/cm^3',
     7  /,'#  Mean excitation energy ....... ',E12.5,' eV',
     8  /,'#  CB parameter ................. ',E12.5,/,'#')
      WRITE(12,2002) VMOL/RHO,SQRT(OMEGA2)
 2002 FORMAT('#  N_Avog/A ..................... ',1P,E12.5,' 1/g'
     1  /,'#  Plasma resonance energy ...... ',E12.5,' eV',/,'#')
      WRITE(12,2003)
 2003 FORMAT('#   Energy',7X,'STP-CS',8X,'L_0',7X,'f(GAMMA)/2',
     1  3X,'Shell corr.',4X,'Density',8X,'DL',11X,'DL',/,'#',4X,'(eV)',
     2  7X,'(eV*cm^2)',31X,'(C/Z)',7X,'(delta/2)',
     3  4X,'(LinSor)',5X,'(Barkas)',/,'# ',103('-'))
      DO IE=1,NE
        EK=ET(IE)
        IF(EK.GT.ECUT-1.0D0) THEN
          BSTP=BETHE(EXPOT,CBF,EK)
          WRITE(12,'(1P,10E13.5)') EK,BSTP,RL0,FCOR,CSHC,CDEN,CLS,CBAR
        ENDIF
      ENDDO
      CLOSE(12)
C
C
      ICASE=1
      IF((IPROJ.EQ.5).OR.(IPROJ.EQ.7)) THEN
        IF(ID.LT.1000) ICASE=2
      ENDIF
C
      IF(ICASE.EQ.2) THEN  ! Extension.
        OPEN(8,FILE='./sdbase/exp-param.tab')
        DO I=1,999
 50       CONTINUE
          READ(8,*,ERR=50,END=51) IZPR,IZTR,AR,BR,CR,DR,FR
          IF((IZPR.EQ.IZ1).AND.(IZTR.EQ.ID)) THEN
            AF=AR; BF=BR; CF=CR; DF=DR; FF=FR
            CLOSE(8)
            GO TO 52
          ENDIF
        ENDDO
 51     CONTINUE
        CLOSE(8)
        ICASE=1
 52     CONTINUE
      ENDIF
C
C  ************  Table of stopping powers from the corrected Bethe
C                formula, with and without the shell correction.
C
      OPEN(8,FILE='stp.dat')
      WRITE(8,2004) CPART,PMASS*REV,IZ1,MNAME,ZT,AW,RHO,EXPOT
 2004 FORMAT('#  Stopping power, corrected Bethe formula.',
     1  /,'#',/,'#  Projectile particle: ',A10,'   rest energy =',
     2  1P,E12.5,' eV',/,'#',41X,'charge =',I3,' e',/,'#',
     3  /,'#  Material filename:   ',A,
     4  /,'#  Electrons/molecule ........ ',1P,E12.5,
     5  /,'#  Molecular weight .......... ',E12.5,
     6  /,'#  Density ................... ',E12.5,' g/cm^3',
     7  /,'#  Mean excitation energy .... ',E12.5,' eV',/,'#')
      WRITE(8,2005)
 2005 FORMAT('#  1 A = 1.0E-8 cm,     1 mtu = 1 g/cm^2',/,'#')
      WRITE(8,2006)
 2006 FORMAT('#   Energy',10X,'Stopping power',10X,'Without shell corr.'
     1  ,4X,'Stopping CS',/,'#    (eV)',8X,'(eV/A)',6X,'(MeV/mtu)',5X,
     2  '(eV/A)',6X,'(MeV/mtu)',4X,'(eV*cm^2)',/,'# ',77('-'))
      ETST=ECUT-1.0D0
      IF(ICASE.EQ.2) ETST=ECUT+ETST
      IELAST=0
      DO IE=1,NE
        EK=ET(IE)
        IF(EK.GT.ETST) THEN
          BSTP=BETHE(EXPOT,CBF,EK)*VMOL
          IF(BSTP.LT.1.0D-75) BSTP=0.0D0
          BSTPM=BSTP/(1.0D6*RHO)
          BSNS=BSTP+(2.0D0*CONS*ZT*VMOL*CSHC)
          IF(BSNS.LT.1.0D-75) BSNS=0.0D0
          BSNSM=BSNS/(1.0D6*RHO)
          WRITE(8,'(1P,10E13.5)')
     1      EK,BSTP*1.0D-8,BSTPM,BSNS*1.0D-8,BSNSM,BSTP/VMOL
          STPE(IE)=BSTPM
        ELSE
          IELAST=IE
        ENDIF
      ENDDO
      CLOSE(8)
C
C  ************  Table of stopping powers estimated from the low-energy
C                extrapolation, and extension if available.
C
      N1=IELAST+1
      E1=ET(N1)
      T1=LOG(E1/EMAX)
      S1=STPE(N1)
      N2=N1+1
      E2=ET(N2)
      S2=STPE(N2)
      SP1=(S2-S1)/(E2-E1)
      CE=1.50D0
      BE=(-SP1*E1/S1)/(CE*T1**(CE-1.0D0))
      AE=LOG(S1)+BE*T1**CE
      DE=CE*EXP(AE)
C
      OPEN(8,FILE='stp-low.dat')
      WRITE(8,2007) CPART,PMASS*REV,IZ1,MNAME,ZT,AW,RHO,EXPOT
 2007 FORMAT('#  Stopping power from the low-E extrapolation, and',
     1  ' extension if available.',
     1  /,'#',/,'#  Projectile particle: ',A10,'   rest energy =',
     2  1P,E12.5,' eV',/,'#',41X,'charge =',I3,' e',/,'#',
     3  /,'#  Material filename:   ',A,
     4  /,'#  Electrons/molecule ........ ',1P,E12.5,
     5  /,'#  Molecular weight .......... ',E12.5,
     6  /,'#  Density ................... ',E12.5,' g/cm^3',
     7  /,'#  Mean excitation energy .... ',E12.5,' eV',/,'#')
      WRITE(8,2008) EMAX,AE,BE,CE
 2008 FORMAT('#  Extrapolation function:    T=E/Emax',/,
     1  '#    STP=EXP(a-b*LOG(T)**c)             if T>1',/,
     2  '#       =1.5*EXP(a)*SQRT(T)*(1-T/3.0)   if T<1',/,
     3  '#   Emax=',1P,E11.4,',  a=',E11.4,/,
     4  '#      b=',E11.4,',  c=',E11.4,/,'#')
      IF(ICASE.EQ.2) THEN
        WRITE(8,2108) AF,BF,CF,DF,FF
 2108   FORMAT('#  Low-energy fitted function:    T=E/keV or MeV',/,
     1    '#     SL=a1*T**a2; SH=(a3/T)*LOG(1.0D0+a4/T+a5*T)',/,
     2    '#    STP=1.0D-15*SL*SH/(SL+SH)',/,
     3    '#     a1=',1P,E11.4,',  a2=',E11.4,/,
     4    '#     a3=',E11.4,',  a4=',E11.4,',  a5=',E11.4,/,'#')
      ENDIF
      WRITE(8,2009)
 2009 FORMAT('#  1 A = 1.0E-8 cm,     1 mtu = 1 g/cm^2',/,'#')
      IF(ICASE.EQ.1) THEN
        WRITE(8,2010)
 2010   FORMAT('#   Energy',9X,'STP extrapolated',6X,'Stopping CS',
     1    /,'#    (eV)',8X,'(eV/A)',6X,'(MeV/mtu)',4X,'(eV*cm^2)',
     2    /,'# ',51('-'))
      ELSE
        WRITE(8,2110)
 2110   FORMAT('#   Energy',11X,'STP low-E fit',7X,'Stopping CS',
     1    6X,'STP extrapolated',6X,'STP CS extr',
     2    /,'#    (eV)',8X,'(eV/A)',6X,
     2    '(MeV/mtu)',4X,'(eV*cm^2)',5X,'(eV/A)',6X,'(MeV/mtu)',
     4    4X,'(eV*cm^2)',/,'# ',90('-'))
      ENDIF
      SCB=0.0D0
      SLOWC=0.0D0
      IF(ICASE.EQ.2) THEN
        SCB=STPE(IELAST+1)
        XA=ECUT+ECUT
        IF(IZ1.EQ.1) THEN
          T=XA*1.0D-3
        ELSE
          T=XA*1.0D-6
        ENDIF
        S1=AF*T**BF
        S2=(CF/T)*LOG(1.0D0+DF/T+FF*T)
        FITL=S1*S2/(S1+S2)
        SLOWC=VMOL*(FITL*1.0D-15)/(1.0D6*RHO)
      ENDIF
C
      DO IE=1,IELAST+1
        EK=ET(IE)
        IF(EK.GT.ETST) GO TO 105
        T=EK/EMAX
        IF(EK.LT.EMAX) THEN
          STPM=DE*SQRT(T)*(1.0D0-T/3.0D0)
        ELSE
          STPM=EXP(AE-BE*LOG(T)**CE)
        ENDIF
        STPL=STPM*(1.0D6*RHO)
C
        IF(ICASE.EQ.2) THEN
          STP1M=STPM
          STP1L=STPL
          IF(IZ1.EQ.1) THEN
            T=EK*1.0D-3
          ELSE
            T=EK*1.0D-6
          ENDIF
          S1=AF*T**BF
          S2=(CF/T)*LOG(1.0D0+DF/T+FF*T)
          FITL=S1*S2/(S1+S2)
          STPM=VMOL*(FITL*1.0D-15)/(1.0D6*RHO)
          IF(EK.GT.ECUT)
     1      STPM=(1.0D0-(EK/ECUT-1.0D0)*(1.0D0-SCB/SLOWC))*STPM
          STPL=STPM*(1.0D6*RHO)
          IF(EK.GT.ELOW-0.01D0) WRITE(8,'(1P,10E13.5)') EK,STPL*1.0D-8,
     1      STPM,STPL/VMOL,STP1L*1.0D-8,STP1M,STP1L/VMOL
        ELSE
          IF(EK.GT.ELOW-0.01D0) WRITE(8,'(1P,10E13.5)')
     1      EK,STPL*1.0D-8,STPM,STPL/VMOL
        ENDIF
        STPE(IE)=STPM
      ENDDO
 105  CONTINUE
      CLOSE(8)
C
C  ****  Radiative stopping of electrons and muons.
C
      IF(IPROJ.LT.3) THEN
        CALL EBRIN
        DO IE=1,NE
          EK=ET(IE)
          CALL EBRSTP(IPROJ,EK,RSTP,RSTPU)
          STPR(IE)=RSTPU
          AUX1(IE)=1.0D0/(STPE(IE)+STPR(IE))
        ENDDO
        RADLEN=ET(NE)/(STPR(NE)*1.0D6*RHO)
      ELSE IF(IPROJ.LT.5) THEN
        CALL RMRIN
        DO IE=1,NE
          EK=ET(IE)
          IF(EK.GT.500.0E6) THEN
C  ----  Bremsstrahlung high-energy correction.
            IF(ABS(EK-ECC).GT.1.0D-12*EK) THEN
              CALL KINPAR(IZ1,PMASS,EK)
            ENDIF
            WRIDGE=2.0D0*GAMMA2*BETA2*REV
     1        /(1.0D0+2.0D0*GAMMA/PMASS+1.0D0/PMASS**2)
            QL=LOG(2.0D0*WRIDGE/REV)
            DD=VMOL*(RECL**2*REV*ZT/SL)*(LOG(2.0D0*ETT/RMASEV)-QL/3.0D0)
     1        *QL**2
            STPE(IE)=STPE(IE)+DD/(1.0D6*RHO)
          ENDIF
          CALL RMRSTP(IPROJ,EK,RSTP,RSTPU)
          STPR(IE)=RSTPU
          AUX1(IE)=1.0D0/(STPE(IE)+STPR(IE))
        ENDDO
        RADLEN=0.0D0
      ELSE
        DO IE=1,NE
          STPR(IE)=0.0D0
          AUX1(IE)=1.0D0/STPE(IE)
        ENDDO
        RADLEN=0.0D0
      ENDIF
      RANGE(1)=0.0D0
      DO IE=2,NE
        JE=IE
        IF(ET(IE).GT.ELOW) THEN
          RANGE(IE)=RANGE(IE-1)
     1      +SMOMLL(ET,AUX1,ET(JE-1),ET(JE),NE,0,0)
        ELSE
          RANGE(IE)=0.0D0
        ENDIF
      ENDDO
C
C  ************  Table of mass stopping powers and particle CSDA range.
C
      OPEN(8,FILE='mstp.dat')
      WRITE(8,3001) CPART,PMASS*REV,IZ1,MNAME,ZT,AW,RHO,EXPOT
 3001 FORMAT('#  Mass stopping powers and CSDA range.',
     1  /,'#',/,'#  Projectile particle: ',A10,'   rest energy =',
     2  1P,E12.5,' eV',/,'#',41X,'charge =',I3,' e',/,'#',
     3  /,'#  Material filename:   ',A,
     4  /,'#  Electrons/molecule ........... ',1P,E12.5,
     5  /,'#  Molecular weight ............. ',E12.5,
     6  /,'#  Density ...................... ',E12.5,' g/cm^3',
     7  /,'#  Mean excitation energy ....... ',E12.5,' eV')
      IF(IPROJ.LT.3) WRITE(8,3101) RADLEN,RADLEN*RHO
 3101 FORMAT('#  Radiation length ............. ',1P,E12.5,' cm',
     1  /,'#                              or ',E12.5,' g/cm^2')
      WRITE(8,'(''#'')')
      IF(ICASE.EQ.1) THEN
        WRITE(8,2008) EMAX,AE,BE,CE
      ELSE
        WRITE(8,2108) AF,BF,CF,DF,FF
      ENDIF
      WRITE(8,3002)
 3002 FORMAT('#  1 mtu = 1 g/cm^2',/,'#')
      WRITE(8,3003)
 3003 FORMAT('#   Energy',5X,'Elect. STP',4X,'Rad. STP',
     1  5X,'Total STP',5X,'Range',/,'#    (eV)',7X,
     2  '(MeV/mtu)',4X,'(MeV/mtu)',4X,'(MeV/mtu)',4X,'(g/cm^2)',
     3  /,'# ',64('-'))
      DO IE=1,NE
        IF(ET(IE).GT.ELOW-1.0D0) WRITE(8,'(1P,10E13.5)')
     1    ET(IE),STPE(IE),STPR(IE),STPE(IE)+STPR(IE),RANGE(IE)*1.0D-6
      ENDDO
      CLOSE(8)
C
C  ************  Table of stopping powers and particle CSDA range.
C
      OPEN(8,FILE='lstp.dat')
      WRITE(8,3004) CPART,PMASS*REV,IZ1,MNAME,ZT,AW,RHO,EXPOT
 3004 FORMAT('#  Stopping powers and CSDA range.',
     1  /,'#',/,'#  Projectile particle: ',A10,'   rest energy =',
     2  1P,E12.5,' eV',/,'#',41X,'charge =',I3,' e',/,'#',
     3  /,'#  Material filename:   ',A,
     4  /,'#  Electrons/molecule ........... ',1P,E12.5,
     5  /,'#  Molecular weight ............. ',E12.5,
     6  /,'#  Density ...................... ',E12.5,' g/cm^3',
     7  /,'#  Mean excitation energy ....... ',E12.5,' eV')
      IF(IPROJ.LT.3) WRITE(8,3101) RADLEN,RADLEN*RHO
      WRITE(8,'(''#'')')
      IF(ICASE.EQ.1) THEN
        WRITE(8,2008) EMAX,AE,BE,CE
      ELSE
        WRITE(8,2108) AF,BF,CF,DF,FF
      ENDIF
      WRITE(8,3005)
 3005 FORMAT('#  1 A = 1.0E-8 cm',/,'#')
      WRITE(8,3006)
 3006 FORMAT('#   Energy',5X,'Elect. STP',4X,'Rad. STP',4X,
     1  'Total STP',6X,'Range'/,'#    (eV)',8X,'(eV/A)',7X,'(eV/A)',
     2  7X,'(eV/A)',8X,'(cm)'/,'# ',64('-'))
      FU=1.0D-2*RHO
      DO IE=1,NE
        IF(ET(IE).GT.ELOW-1.0D0) WRITE(8,'(1P,10E13.5)')
     1    ET(IE),STPE(IE)*FU,STPR(IE)*FU,
     1    (STPE(IE)+STPR(IE))*FU,RANGE(IE)*1.0D-6/RHO
      ENDDO
      CLOSE(8)
C
C  ----  CSDA depth-dose distribution (without elastic scattering).
C
      OPEN(8,FILE='depth-dose.dat')
      WRITE(8,5001) CPART,PMASS*REV,IZ1,MNAME,ZT,AW,RHO,EXPOT
 5001 FORMAT('#  Depth-dose distribution.',
     1  /,'#',/,'#  Projectile particle: ',A10,'   rest energy =',
     2  1P,E12.5,' eV',/,'#',41X,'charge =',I3,' e',/,'#',
     3  /,'#  Material filename:   ',A,
     4  /,'#  Electrons/molecule ........... ',1P,E12.5,
     5  /,'#  Molecular weight ............. ',E12.5,
     6  /,'#  Density ...................... ',E12.5,' g/cm^3',
     7  /,'#  Mean excitation energy ....... ',E12.5,' eV')
      IF(IPROJ.LT.3) WRITE(8,3101) RADLEN,RADLEN*RHO
      WRITE(8,'(''#'')')
      WRITE(8,3002)
      DO IE=1,NE
        STPT(IE)=(STPE(IE)+STPR(IE))*FU*1.0D8
      ENDDO
      CALL SPLINE(ET,STPT,A0,B0,C0,D0,0.0D0,0.0D0,NE)
      NRT=0
      IZERO=0
      DO IE=1,NE
        IF(IZERO.EQ.0) THEN
          IF(RANGE(IE+1).GT.1.0D-10) IZERO=1
        ELSE
          NRT=NRT+1
          RR(NRT)=RANGE(IE)*1.0D-6/RHO
          ENT(NRT)=ET(IE)
        ENDIF
      ENDDO
C
      CALL SPLINE(ENT,RR,A1,B1,C1,D1,0.0D0,0.0D0,NRT)
      CALL SPLINE(RR,ENT,A2,B2,C2,D2,0.0D0,0.0D0,NRT)
C
      E0T(1)=5.0D5
      E0T(2)=2.5D6
      E0T(3)=1.0D7
      E0T(4)=5.0D7
      E0T(5)=2.5D8
      NED=5
      DO I=1,NED
        E0=E0T(I)
        CALL FINDI(E0,ENT,NRT,I1)
        ZMAX(I)=A1(I1)+E0*(B1(I1)+E0*(C1(I1)+E0*D1(I1)))
      ENDDO
      NZP=1001
      ZLOW=0.1D0*ZMAX(1)
      ZUP=1.5D0*ZMAX(NED)
      FACT=(ZUP/ZLOW)**(1.0D0/DBLE(NZP))
      ZZ=ZLOW/FACT
      WRITE(8,5003)
 5003 FORMAT('#    Depth',19X,'Depth-dose, D(z)=S(E(z)), in eV/cm',
     1  /,'#    (cm)',6X,'E0=0.5 MeV',5X,'2.5 MeV',6X,'10 MeV',7X,
     2  '50 MeV',7X,'250 MeV',/,'# ',77('-'))
      NPZ=0
      NTST=NED
      DO IR=1,NZP+1
        ZZ=ZZ*FACT
        DO I=1,NED
          Z=ZMAX(I)-ZZ
          IF(Z.GT.0.0D0) THEN
            CALL FINDI(Z,RR,NRT,I2)
            EZ=A2(I2)+Z*(B2(I2)+Z*(C2(I2)+Z*D2(I2)))
            CALL FINDI(EZ,ET,NE,I0)
            DDOSE(I)=A0(I0)+EZ*(B0(I0)+EZ*(C0(I0)+EZ*D0(I0)))
          ELSE
            EZ=0.0D0
            DDOSE(I)=0.0D0
          ENDIF
        ENDDO
        WRITE(8,'(1P,20E13.5)') ZZ,(DDOSE(I),I=1,NED)
        NPZ=NPZ+1
        AUX1(NPZ)=ZZ
        AUX2(NPZ)=DDOSE(NTST)
      ENDDO
      SUMD=SMOMLL(AUX1,AUX2,0.0D0,ZUP,NPZ,0,0)+ELOW
      WRITE(8,'(A,1P,E12.5)') '#  Integrated dose/E0=',SUMD/E0T(NTST)
      WRITE(6,'(A,1P,E12.5)') '  Integrated dose/E0=',SUMD/E0T(NTST)
      CLOSE(8)
C
C  ****  Generation of PENELOPE tables.
C
      OPEN(8,FILE='PENstp.dat')
      WRITE(8,4004) CPART,PMASS*REV,IZ1,MNAME,ZT,AW,RHO,EXPOT
 4004 FORMAT('#  Stopping power, corrected Bethe formula.',
     1  /,'#',/,'#  Projectile particle: ',A10,'   rest energy =',
     2  1P,E12.5,' eV',/,'#',41X,'charge =',I3,' e',/,'#',
     3  /,'#  Material filename:   ',A,
     4  /,'#  Electrons/molecule ........... ',1P,E12.5,
     5  /,'#  Molecular weight ............. ',E12.5,
     6  /,'#  Density ...................... ',E12.5,' g/cm^3',
     7  /,'#  Mean excitation energy ....... ',E12.5,' eV')
      IF(IPROJ.LT.3) WRITE(8,3101) RADLEN,RADLEN*RHO
      WRITE(8,'(''#'')')
      WRITE(8,4005)
 4005 FORMAT('#  1 A = 1.0E-8 cm,   1 mtu = 1 g/cm^2',/,'#')
      WRITE(8,4006)
 4006 FORMAT('#   Energy',10X,'Stopping power',12X,'Stopping CS',/,
     1  '#    (eV)',8X,'(eV/A)',6X,'(MeV/mtu)',9X,'(eV cm^2)',/,
     2  '# ',56('-'))
C
      IF(IPROJ.LT.5) THEN
        ELOW=50.0D0-1.0D0
      ELSE
        ELOW=1.0D4-1.0D0
      ENDIF
      EHIGH=1.0D9-1.0D0

      IGRID=0
      FGRID=1.0D1
  201 IGRID=IGRID+1
      EV=EGRT(IGRID)*FGRID
      IF(IGRID.EQ.17) THEN
        IGRID=1
        FGRID=10.0D0*FGRID
      ENDIF
      IF(EV.GT.ELOW) THEN
        BSTPM=FINTRP(EV,ET,STPE,NE,7)
        IF(BSTPM.LT.1.0D-75) BSTPM=0.0D0
        BSTP=BSTPM*(1.0D6*RHO)
        WRITE(8,'(1P,3E13.5,5X,E13.5)') EV,BSTP*1.0D-8,BSTPM,
     1    BSTP/VMOL
      ENDIF
      IF(EV.LT.EHIGH) GO TO 201
      CLOSE(8)
C
C
      WRITE(6,'(/,''  Projectile: '',A)') CPART
      WRITE(6,'(''    Charge/e ='',I3)') IZ1
      WRITE(6,'(''    Mass/m_e ='',1P,E11.4)') PMASS
      WRITE(6,'(''  Done!'')')
      STOP
      END


C  *********************************************************************
C                       SUBROUTINE STMATW
C  *********************************************************************
      SUBROUTINE STMATW(MNAME,W0,F0,NTAB,ID)
C
C  Generates the material definition file. When this file exists, the
C  input of the program is largely simplified, because material data
C  are read directly from the material definition file.
C
      USE CONSTANTS
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      LOGICAL LOG1
      CHARACTER MNAME*15,IFILE0*40,IFILE*40,BUFFER*10,TITLE*99
      CHARACTER LASYMB*2
C  ****  Composition data.
      COMMON/COMPOS/STF(30),AW,IZ(30),NELEM
C  ****  OOS and dielectric function tables.
      PARAMETER (NR=750)
      COMMON/COOST/ZT,EXPOT,RHO,VMOL,OMEGA2,WOD,ZTOT
      COMMON/CETAS/WP(NR),FT(NR),SFT(NR),WL(NR),FL(NR),ETA2(NR),ETA1(NR)
      COMMON/CLOREN/WR,WG,SG,MOM
C  ****  Element data.
      COMMON/CADATA/ATW(99),EPX(99),RSCR(99),ETA(99),EB(99,30),
     1  ALW(99,30),CP0(99,30),IFI(99,30),IKS(99,30),NSHT(99),LASYMB(99)
C  ****  Other arrays.
      PARAMETER (NT=5000)
      DIMENSION W0(NT),F0(NT)
C
      IFILE0=MNAME//'.mat'
      IFILE=' '
      NC=0
      DO I=1,40  ! Copy the filename with blanks removed.
        IF(IFILE0(I:I).NE.' ') THEN
          NC=NC+1
          IFILE(NC:NC)=IFILE0(I:I)
        ENDIF
      ENDDO
      INQUIRE(FILE=IFILE,EXIST=LOG1)
C
C  ****  The material file does exist.
C
      OPEN(17,FILE=IFILE)
      NTAB=0
      IF(LOG1) THEN
        READ(17,'(A15)',ERR=11,END=11) MNAME
        WRITE(6,1001) MNAME
        READ(17,'(A99)',ERR=11,END=11) TITLE
 1001   FORMAT(/2X,'****      Material name = ',A)
        READ(17,'(I5)',ERR=11,END=11) ID
        WRITE(6,1101) ID
 1101   FORMAT(2X,'  Identification number =',I5)
        READ(17,*,ERR=11,END=11) NELEM
        WRITE(6,1002) NELEM
 1002   FORMAT(2X,'     Number of elements =',I3)
        DO IEL=1,NELEM
          READ(17,'(I3,4X,E23.14)',ERR=11) IZ(IEL),STF(IEL)
          WRITE(6,1003) LASYMB(IZ(IEL)),IZ(IEL),STF(IEL)
 1003     FORMAT(28X,A2,2X,I2,1P,E12.5)
        ENDDO
        READ(17,*,ERR=11) ZT
        WRITE(6,1004) ZT
 1004   FORMAT(2X,'    Number of electrons =',1P,E12.5)
        READ(17,*,ERR=11) AW
        WRITE(6,1005) AW
 1005   FORMAT(2X,'       Molecular weight =',1P,E12.5,' g/mol')
        READ(17,*,ERR=11) RHO
        WRITE(6,1006) RHO
 1006   FORMAT(2X,'           Mass density =',1P,E12.5,' g/cm^3')
        READ(17,*,ERR=11) EXPOT
        WRITE(6,1007) EXPOT
 1007   FORMAT(2X,' Mean excitation energy =',1P,E12.5,' eV')
        READ(17,*,ERR=11) VMOL
        WRITE(6,1008) VMOL
 1008   FORMAT(2X,'      Molecular density =',1P,E12.5,' 1/cm^3')
        READ(17,*,ERR=11) OMEGA
        WRITE(6,1009) OMEGA
 1009   FORMAT(2X,'          Plasma energy =',1P,E12.5,' eV')
        OMEGA2=OMEGA*OMEGA
        READ(17,*,ERR=11) WG
        WRITE(6,1109) WG
 1109   FORMAT(2X,'             Gap energy =',1P,E12.5,' eV')
        READ(17,*,ERR=11) WOD
        WRITE(6,1010) WOD
 1010   FORMAT(2X,'          Switch energy =',1P,E12.5,' eV')
        READ(17,'(A)',ERR=11) BUFFER
        DO K=1,NT
          READ(17,*,END=11,ERR=11) W0(K),F0(K)
          NTAB=K
        ENDDO
 11     CONTINUE
C
        CLOSE(17)
        IF(NTAB.GT.10) RETURN
        WRITE(6,'(/2X,A,A/)') 'Corrupted material file. Continue ',
     1    'entering data ...'
        OPEN(17,FILE=IFILE)
      ENDIF
C
C  ****  Generate a new material data file.
C
C  ----  Calculated OOSs of DHFS atoms.
      CALL MATER(W0,F0,NTAB,MNAME,ID)
      TITLE='OOS built from DHFS atomic subshell OOSs'
      WRITE(6,1011) TITLE
 1011 FORMAT(/2X,'****  ',A)
      WRITE(6,1012) ID
 1012 FORMAT(2X,'  Identification number =',I3)
C  ----  Number of electrons per molecule.
      WRITE(6,1004) ZT
      IF(ZT.LT.0.9999D0) STOP
C  ----  Molecular weight.
      WRITE(6,1005) AW
      IF(AW.LT.1.0D-35) STOP
C  ----  Mass density.
      WRITE(6,1006) RHO
      IF(RHO.LT.1.0D-35) STOP
C  ----  Mean excitation energy.
      WRITE(6,1007) EXPOT
      IF(EXPOT.LT.1.0D-35) STOP
C  ----  Molecular density
      WRITE(6,1008) VMOL
C  ----  Plasma energy.
      WRITE(6,1009) SQRT(OMEGA2)
C  ----  Gap energy.
      WRITE(6,1109) WG
C  ----  Switch energy.
      IF(WOD.LT.0.0D0.OR.WOD.GT.100.0D0) WOD=75.0D0
      WRITE(6,1010) WOD
C
      ZTOT=SMOMLL(W0,F0,W0(1),W0(NTAB),NTAB,0,0)
      EXPT=SMOMLL(W0,F0,W0(1),W0(NTAB),NTAB,0,1)
      WRITE(6,'(A,1P,E13.5)')   '         new f sum =',ZTOT
      WRITE(6,'(A,1P,E13.5,/)') '         new EXPOT =',EXP(EXPT/ZTOT)
C
      WRITE(17,'(A15)') MNAME
      WRITE(17,'(A99)') TITLE
      WRITE(17,'(I5,5X,''Identification number'')') ID
      WRITE(17,'(I3,5X,''NELEM'')') NELEM
      DO IEL=1,NELEM
        WRITE(17,'(I3,2X,A2,1P,E23.14,5X,''Element, stiochiometric '',
     1    ''index'')') IZ(IEL),LASYMB(IZ(IEL)),STF(IEL)
      ENDDO
      WRITE(17,'(1P,E21.14,5X,''No. of electrons'')') ZT
      WRITE(17,'(1P,E21.14,5X,''Molecular weight (g/mol)'')') AW
      WRITE(17,'(1P,E21.14,5X,''Mass density (g/cm^3)'' )') RHO
      WRITE(17,'(1P,E21.14,5X,''Mean excitation energy (eV)'' )') EXPOT
      WRITE(17,'(1P,E21.14,5X,''Molecular density (1/cm^3)'')') VMOL
      WRITE(17,'(1P,E21.14,5X,''Plasma resonance energy (eV)'' )')
     1  SQRT(OMEGA2)
C
      IF(WG.LT.1.0D-45) THEN
        DO I=1,NTAB
          IF(F0(I).LT.1.0D-45) THEN
            WG=W0(I)
          ELSE
            GO TO 12
          ENDIF
        ENDDO
      ENDIF
 12   CONTINUE
      WRITE(17,'(1P,E21.14,5X,''Gap energy (eV)'' )') WG
      WRITE(17,'(1P,E21.14,5X,''Cutoff energy (eV)'')') WOD
      WRITE(17,'(A)') '#        W (eV)             OOS (1/eV)'
      DO K=1,NTAB
        IF(W0(K).LT.100.0D0.AND.F0(K).LT.1.0D-30) F0(K)=0.0D0
        WRITE(17,'(1P,E21.14,1X,E21.14)') W0(K),F0(K)
      ENDDO
      CLOSE(17)
      WRITE(6,'(/2X,A,//)') 'The material file has been created.'
C
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE MATER
C  *********************************************************************
      SUBROUTINE MATER(WTAB,FTAB,NTAB,ATITLE,ID)
C
C  Defines the material parameters and the optical oscillator strength
C  using atomic OOSs precalculated with the DHFS self-consistent poten-
C  tial.
C
      USE CONSTANTS
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      CHARACTER YN*1,LASYMB*2,CH2*2,CH5*5,NAME*62,ATITLE*15
      PARAMETER (PI=3.1415926535897932D0)
      PARAMETER (NT=5000)
      DIMENSION WTAB(NT),FTAB(NT)
C  ****  Composition data.
      COMMON/COMPOS/STF(30),AW,IZ(30),NELEM
      DIMENSION FBW(30)
C  ****  OOS and dielectric function tables.
      COMMON/COOST/ZT,EXPOT,RHO,VMOL,OMEGA2,WOD,ZTOT
C  ****  Storage for subshell OOSs of atoms.
      PARAMETER(NWM=15000)
      COMMON/CPHXS/EBIN(30),W(30,NWM),SHOOS(30,NWM),ISH(30),NW(30),NSH
      DIMENSION X(NWM),Y(NWM),X1(NWM),Y1(NWM),X2(NWM),Y2(NWM)
      DIMENSION W0(NWM),F0(NWM)
C  ****  Element data.
      COMMON/CADATA/ATW(99),EPX(99),RSCR(99),ETA(99),EB(99,30),
     1  ALW(99,30),CP0(99,30),IFI(99,30),IKS(99,30),NSHT(99),LASYMB(99)
C
      COMMON/CLOREN/WR,WG,SG,MOM
      EXTERNAL FLOR
C
      WRITE(6,*) '      '
      WRITE(6,*) ' Select one option (1 or 2):'
      WRITE(6,*) '   1: Enter composition data from the keyboard'
      WRITE(6,*) '   2: Read them from the file pdcompos.pen'
      READ(5,*) IREAD
      IF(IREAD.EQ.2) GO TO 901
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C  ************  Data entered from keyboard.
C
      WRITE(6,'(1X,'' Material: '',A)') ATITLE
C
      ZT=0.0D0
      AW=0.0D0
      XPOT=0.0D0
      EXPOT0=-1.0D0
C  ****  Chemical formula or fractions by weight.
 800  CONTINUE
      WRITE(6,*) '      '
      WRITE(6,*) ' Chemical formula:'
      WRITE(6,*) ' Number of elements in the molecule...'
      READ(5,*) NELEM
      IF(NELEM.LT.1.OR.NELEM.GT.30) THEN
        WRITE(6,*) ' NELEM must be positive and less than 31.'
        WRITE(6,*) ' Please, enter an allowed value.'
        GO TO 800
      ENDIF
      WRITE(6,'(1X,'' Number of elements = '',I2)') NELEM
      WRITE(6,*) '      '
C
      IF(NELEM.EQ.1) THEN
 810    CONTINUE
        WRITE(6,230)
 230    FORMAT(/1X,' Enter atomic number of the element...')
        READ(5,*) IZZ
        IF(IZZ.LT.1.OR.IZZ.GT.99) THEN
          WRITE(6,*) ' The atomic number must be in the range 1-99.'
          WRITE(6,*) ' Please, enter an allowed value.'
          GO TO 810
        ENDIF
        IZ(1)=IZZ
        STF(1)=1.0D0
        WRITE(6,'(1X,'' Element: '',A2,'' (Z='',I2,''), atoms/'',
     1    ''molecule ='',1P,E12.5)') LASYMB(IZZ),IZZ,STF(1)
        ZT=IZZ
        AW=ATW(IZZ)
        XPOT=IZZ*LOG(EPX(IZZ))
      ELSE
C
        WRITE(6,*) ' Select one option (1 or 2):'
        WRITE(6,*) '   1: Enter chemical (stoichiometric) formula'
        WRITE(6,*) '   2: Enter fraction by weight of each element'
        READ(5,*) IREAD2
C
        IF(IREAD2.EQ.2) THEN
          WRITE(6,*) '      '
          WRITE(6,*) ' Weight fractions...'
          DO I=1,NELEM
 801        CONTINUE
            IF(I.EQ.1) THEN
              WRITE(6,231)
 231          FORMAT(/1X,' Enter atomic number and fraction by weight',
     1          ' of the first element ...')
            ELSE IF(I.EQ.2) THEN
              WRITE(6,232)
 232          FORMAT(/1X,' Enter atomic number and fraction by weight',
     1          ' of the second element ...')
            ELSE IF(I.EQ.3) THEN
              WRITE(6,233)
 233          FORMAT(/1X,' Enter atomic number and fraction by weight',
     1          ' of the third element ...')
            ELSE
              WRITE(6,234) I
 234          FORMAT(/1X,' Enter atomic number and fraction by weight',
     1          ' of the',I3,'-th element ...')
            ENDIF
            READ(5,*) IZZ,FBW(I)
            IF(IZZ.LT.1.OR.IZZ.GT.99) THEN
              WRITE(6,*) ' The atomic number must be in the range 1-99.'
              WRITE(6,*) ' Please, enter an allowed value.'
              GO TO 801
            ENDIF
            IF(FBW(I).LT.1.0D-35) THEN
              WRITE(6,*) ' The fraction by weight must be positive.'
              WRITE(6,*) ' Please, enter a positive value.'
              GO TO 801
            ENDIF
            WRITE(6,*) '      '
            IZ(I)=IZZ
            WRITE(6,'(1X,'' Element: '',A2,'' (Z='',I2,''), fraction '',
     1      ''by weight ='',1P,E12.5)') LASYMB(IZZ),IZZ,FBW(I)
            IF(I.GT.1) THEN
              DO K=1,I-1
                IF(IZZ.EQ.IZ(K)) THEN
                  STOP 'This element has been declared twice.'
                ENDIF
              ENDDO
            ENDIF
            STF(I)=FBW(I)/ATW(IZZ)
          ENDDO
C
          STFM=0.0D0
          DO I=1,NELEM
            IF(STF(I).GT.STFM) THEN
              STFM=STF(I)
            ENDIF
          ENDDO
          IF(STFM.LT.1.0D-16) THEN
            STOP 'Fractions by weight are too small.'
          ENDIF
          DO I=1,NELEM
            STF(I)=STF(I)/STFM
          ENDDO
C
          WRITE(6,*) '      '
          DO I=1,NELEM
            IZZ=IZ(I)
            WRITE(6,'(1X,'' Element: '',A2,'' (Z='',I2,''), atoms/'',
     1        ''molecule ='',1P,E12.5)') LASYMB(IZZ),IZZ,STF(I)
            ZT=ZT+IZZ*STF(I)
            AW=AW+ATW(IZZ)*STF(I)
            XPOT=XPOT+IZZ*LOG(EPX(IZZ))*STF(I)
          ENDDO
        ELSE
C
          WRITE(6,*) '      '
          WRITE(6,*) ' Stoichiometric indexes...'
          DO I=1,NELEM
 802        CONTINUE
            IF(I.EQ.1) THEN
              WRITE(6,2001)
 2001       FORMAT(/1X,' Enter atomic number and number of atoms/molec',
     1        'ule of the first element ...')
            ELSE IF(I.EQ.2) THEN
              WRITE(6,2002)
 2002       FORMAT(/1X,' Enter atomic number and number of atoms/molec',
     1        'ule of the second element ...')
            ELSE IF(I.EQ.3) THEN
              WRITE(6,2003)
 2003       FORMAT(/1X,' Enter atomic number and number of atoms/molec',
     1        'ule of the third element ...')
            ELSE
              WRITE(6,2004) I
 2004       FORMAT(/1X,' Enter atomic number and number of atoms/molec',
     1        'ule of the',I3,'-th element ...')
            ENDIF
            READ(5,*) IZZ,STF(I)
            IF(IZZ.LT.1.OR.IZZ.GT.99) THEN
              WRITE(6,*) ' The atomic number must be in the range 1-99.'
              WRITE(6,*) ' Please, enter an allowed value.'
              GO TO 802
            ENDIF
            IF(STF(I).LT.1.0D-35) THEN
              WRITE(6,*) ' Stoichiometric fractions must be positive.'
              WRITE(6,*) ' Please, enter a positive value.'
              GO TO 802
            ENDIF
            IZ(I)=IZZ
            WRITE(6,'(1X,'' Element: '',A2,'' (Z='',I2,''), atoms/'',
     1        ''molecule ='',1P,E12.5)') LASYMB(IZZ),IZZ,STF(I)
            IF(I.GT.1) THEN
              DO K=1,I-1
                IF(IZZ.EQ.IZ(K)) THEN
                  STOP 'This element has been declared twice.'
                ENDIF
              ENDDO
            ENDIF
            ZT=ZT+IZZ*STF(I)
            AW=AW+ATW(IZZ)*STF(I)
            XPOT=XPOT+IZZ*LOG(EPX(IZZ))*STF(I)
          ENDDO
        ENDIF
      ENDIF
      EXPOT=EXP(XPOT/ZT)
C
 803  CONTINUE
      WRITE(6,*) '      '
      WRITE(6,*) ' Enter mass density (g/cm^3) ...'
      READ(5,*) RHO
      WRITE(6,'(1X,'' Density = '',1P,E12.5,'' g/cm^3'')') RHO
      IF(RHO.LT.1.0D-35) THEN
        WRITE(6,*) ' The mass density must be positive.'
        WRITE(6,*) ' Please, enter a positive value.'
        GO TO 803
      ENDIF
      VMOL=AVOG*RHO/AW
      ID=1001
      GO TO 904
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C  ************  Material data read from file 'pdcompos.pen'.
C
 901  CONTINUE
      WRITE(6,*) ' Enter material identification number ...'
      READ(5,*) IDNUM
      IF(IDNUM.LT.1.OR.IDNUM.GT.300) THEN
        STOP 'The allowed material ID numbers are 1-300.'
      ENDIF
      ID=IDNUM
C
      OPEN(7,FILE='./sdbase/pdcompos.pen')
      DO I=1,15
        READ(7,'(A62)') NAME
      ENDDO
      DO K1=1,300
        READ(7,'(I3,2X,A62)',END=902) IORD,NAME
        READ(7,*) NELEM,HOLLOW,EXPOT,RHO
        IF(NELEM.GT.30) THEN
          STOP 'NELEM cannot be larger than 30.'
        ENDIF
        IF(NELEM.LT.1) THEN
          STOP 'NELEM must be positive.'
        ENDIF
        DO I=1,NELEM
          READ(7,*) IZ(I),HOLLOW,STF(I)
        ENDDO
        IF(IORD.EQ.IDNUM) GO TO 903
      ENDDO
 902  CONTINUE
      STOP 'Abnormal termination of file ''pdcompos.pen''.'
 903  CONTINUE
      CLOSE(UNIT=7)
      WRITE(6,'(/2X,I3,1X,A62)') IORD,NAME
      EXPOT0=EXPOT
C
      ZT=0.0D0
      AW=0.0D0
      DO I=1,NELEM
        IZZ=IZ(I)
        IF(IZZ.LT.1.OR.IZZ.GT.99) THEN
          WRITE(6,'(1X,'' Element:    (Z='',I2,''), atoms/'',
     1    ''molecule ='',1P,E12.5)') LASYMB(IZZ),IZZ,STF(I)
          STOP 'Wrong atomic number.'
        ENDIF
        WRITE(6,'(1X,'' Element: '',A2,'' (Z='',I2,''), atoms/'',
     1    ''molecule ='',1P,E12.5)') LASYMB(IZZ),IZZ,STF(I)
        IF(STF(I).LT.1.0D-35) THEN
          STOP 'STF must be positive.'
        ENDIF
        IF(I.GT.1) THEN
          DO K=1,I-1
            IF(IZZ.EQ.IZ(K)) THEN
              STOP 'Element has been declared twice.'
            ENDIF
          ENDDO
        ENDIF
        ZT=ZT+IZZ*STF(I)
        AW=AW+ATW(IZZ)*STF(I)
      ENDDO
      WRITE(6,'(/1X,'' Density = '',1P,E12.5,'' g/cm^3'')') RHO
      WRITE(6,'(/1X,'' Number of electrons per molecule = '',
     1  1P,E12.5)') ZT
      IF(RHO.LT.1.0D-35) THEN
        STOP 'The density must be positive.'
      ENDIF
      WRITE(6,'(1X,'' Mean excitation energy ='',1P,E12.5,
     1  '' eV'')') EXPOT
      VMOL=AVOG*RHO/AW
 904  CONTINUE
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  ****  Square of the nominal plasmon energy (eV^2).
      OMEGA2=4.0D0*PI*(VMOL*A0B**3)*ZT*HREV**2
C
C  ************  Atomic configuration.
C
      DO I=1,99
        NSHT(I)=0
        DO J=1,30
          EB(I,J)=0.0D0
          IFI(I,J)=0
          IKS(I,J)=0
        ENDDO
      ENDDO
C
      DO I=1,NELEM
        IZZ=IZ(I)
C  ****  Loads element data only once. NSHT(IZZ) is used as a status
C        indicator.
        IF(NSHT(IZZ).EQ.0) THEN
          OPEN(7,FILE='./sdbase/pdatconf.p14')
          DO J=1,22
            READ(7,'(A5)') CH5
          ENDDO
          NS=0
          IZZT=0
          DO J=1,150000
            READ(7,2005,END=905) IIZ,IS,CH2,CH5,IIF,EIE
 2005       FORMAT(I3,I4,1X,A2,1X,A5,I3,E13.5)
            IF(IIZ.EQ.IZZ) THEN
              NS=NS+1
              IF(NS.GT.30) THEN
                WRITE(6,'(/1X,''NS ='',I4)') NS
                STOP 'Too many shells.'
              ENDIF
              IF(IS.LT.1.OR.IS.GT.30) THEN
                WRITE(6,'(/1X,''IS ='',I4)') IS
                STOP 'Wrong shell number.'
              ENDIF
              IZZT=IZZT+IIF
              EB(IZZ,IS)=EIE
              IFI(IZZ,IS)=IIF
              IKS(IZZ,NS)=IS
            ENDIF
          ENDDO
 905      CONTINUE
          NSHT(IZZ)=NS
          IF(IZZ.NE.IZZT) THEN
            STOP 'Unbalanced charges (element).'
          ENDIF
          CLOSE(7)
        ENDIF
      ENDDO
C
C  ****  Set the optical oscillator strength table.
C
      SLT=ZT*LOG(EXPOT)
      IF(RHO.LT.0.1D0) THEN
        ECUT=0.0D0  ! Gas.
      ELSE
        ECUT=50.0D0  ! Dense material.
      ENDIF
      WOD=1.0D6
C
      FIN=0.0D0
      FOUT=0.0D0
      EBCUT=0.0D0
      DO IEL=1,NELEM
        IZZ=IZ(IEL)
        DO J=1,NSHT(IZZ)
          IS=IKS(IZZ,J)
          IF(IS.EQ.1) EBCUT=MAX(EBCUT,EB(IZZ,IS))
          IF(EB(IZZ,IS).GT.ECUT) THEN
            WOD=MIN(WOD,EB(IZZ,IS))
            FIN=FIN+IFI(IZZ,IS)*STF(IEL)
          ELSE
            FOUT=FOUT+IFI(IZZ,IS)*STF(IEL)
          ENDIF
        ENDDO
      ENDDO
      IF(ECUT.GT.1.0D0) WOD=ECUT
C
      N2=0
      S0N=0.0D0
      SLN=0.0D0
      S0I=0.0D0
      SLI=0.0D0
      S0O=0.0D0
      SLO=0.0D0
      DO IEL=1,NELEM
        IZZ=IZ(IEL)
        WGHT=STF(IEL)
        NSH=NSHT(IZZ)
        CALL OOSTRD(IZZ,NSH)
C
        DO J=1,NSH
          IS=ISH(J)
          N1=NW(IS)
          IF(ECUT.LT.1.0D-5) THEN
            NN=0
          ELSE
            NN=1
          ENDIF
          DO I=1,N1
            IF(W(IS,I).GT.ECUT) THEN
              NN=NN+1
              X1(NN)=W(IS,I)
              Y1(NN)=SHOOS(IS,I)*WGHT
            ENDIF
          ENDDO
          IF(ECUT.GT.0.9999D-5) THEN
            X1(1)=ECUT
            Y1(1)=Y1(2)
          ENDIF
          N1=NN
          SS0=SMOMLL(X1,Y1,X1(1),X1(N1),N1,0,0)
          SSL=SMOMLL(X1,Y1,X1(1),X1(N1),N1,0,1)
          S0N=S0N+SS0
          SLN=SLN+SSL
          IF(EB(IZZ,IS).GT.ECUT) THEN
            IF(N2.EQ.0) THEN
              N2=N1
              N=N2  ! Needed when there is only one inner subshell.
              DO I=1,N2
                X2(I)=X1(I)
                Y2(I)=Y1(I)
                X(I)=X1(I)
                Y(I)=Y1(I)
              ENDDO
            ELSE
              CALL MERGE2(X2,Y2,X1,Y1,X,Y,N2,N1,NWM,N,3)
              N2=N
              DO I=1,N2
                X2(I)=X(I)
                Y2(I)=Y(I)
              ENDDO
            ENDIF
            S0I=S0I+SS0
            SLI=SLI+SSL
          ELSE
            S0O=S0O+SS0
            SLO=SLO+SSL
          ENDIF
        ENDDO
      ENDDO
C
C  ****  Recalculate the discontinuities.
C
      EPS=1.0D-8
      EPS10=1.0D1*EPS
      DO I=1,N
        XC=X(I)
        IF(I.LT.N.AND.I.GT.1) THEN
          IF(X(I+1).LT.XC+EPS*ABS(XC)) XC=XC-EPS10*ABS(XC)
          IF(X(I-1).GT.XC-EPS*ABS(XC)) XC=XC+EPS10*ABS(XC)
        ENDIF
        Y(I)=FINTRP(XC,X,Y,N,3)
      ENDDO
C
C  ****  OOS of inner subshells.
C
      IF(X(1).GT.1.0D-3) THEN
        W0(1)=X(1)*0.5D0
        F0(1)=0.0D0
        W0(2)=X(1)
        F0(2)=0.0D0
        K=2
      ELSE
        K=0
      ENDIF
      DO I=1,N
        K=K+1
        W0(K)=X(I)
        F0(K)=Y(I)
      ENDDO
      N=K
C
      IF(EXPOT0.GT.0.0D0) THEN
        EXPOT=EXPOT0
        WRITE(6,*) '      '
        WRITE(6,'(1X,'' The assigned mean excitation energy I'',
     1    '' is '',1P,E12.5,'' eV'')') EXPOT
        WRITE(6,*) ' Do you want to change it?  (Y/N)'
        READ(5,*) YN
        IF(YN.EQ.'Y'.OR.YN.EQ.'y') THEN
 704      CONTINUE
          WRITE(6,*) '      '
          WRITE(6,*) ' Enter mean excitation energy (eV) ...'
          READ(5,*) EXPOT
          WRITE(6,'(1X,'' Mean excitation energy ='',1P,E12.5,
     1      '' eV'')') EXPOT
          IF(EXPOT.LT.1.0D0) THEN
            WRITE(6,*) ' The mean exc. energy must be larger than 1 eV.'
            WRITE(6,*) ' Please, enter a valid value.'
            GO TO 704
          ENDIF
        ENDIF
      ELSE
        WRITE(6,*) '      '
        WRITE(6,'(1X,'' The calculated mean excitation energy I'',
     1    '' is '',1P,E12.5,'' eV'')') EXPOT
        WRITE(6,*) ' Do you want to change it?  (Y/N)'
        READ(5,*) YN
        IF(YN.EQ.'Y'.OR.YN.EQ.'y') THEN
 804      CONTINUE
          WRITE(6,*) '      '
          WRITE(6,*) ' Enter mean excitation energy (eV) ...'
          READ(5,*) EXPOT
          WRITE(6,'(1X,'' Mean excitation energy ='',1P,E12.5,
     1      '' eV'')') EXPOT
          IF(EXPOT.LT.1.0D0) THEN
            WRITE(6,*) ' The mean exc. energy must be larger than 1 eV.'
            WRITE(6,*) ' Please, enter a valid value.'
            GO TO 804
          ENDIF
        ENDIF
      ENDIF
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C  ****  Gases. Possible rescaling.
C
      IF(FOUT.LT.1.0D-6) THEN
        SUMZ=SMOMLL(W0,F0,W0(1),W0(N),N,0,0)
        SUMI=SMOMLL(W0,F0,W0(1),W0(N),N,0,1)
        WRITE(6,'(A,I5)') '    N=',N
        WRITE(6,'(A,1P,10E15.7)') '   ZT=',ZT
        WRITE(6,'(A,1P,10E15.7)') ' SUMZ=',SUMZ
        WRITE(6,'(A,1P,10E15.7)') '  S0I=',S0I
        WRITE(6,'(A,1P,10E15.7)') '  S0N=',S0N
        WRITE(6,'(A,1P,10E15.7)') ' SUMI=',SUMI
        WRITE(6,'(A,1P,10E15.7)') '  SLI=',SLI
        WRITE(6,'(A,1P,10E15.7)') 'EXPOT=',EXPOT
        RA1=ZT/SUMZ
        RA2=EXP((RA1*SUMI-ZT*LOG(EXPOT))/ZT)
        NTAB=N
        DO I=1,NTAB
          WTAB(I)=W0(I)/RA2
          FTAB(I)=F0(I)*RA1*RA2
        ENDDO
        WOD=WOD/RA2
        WRITE(6,*) ' '
        WRITE(6,'(A,1P,2E14.6)') ' RA1, RA2 =',RA1,RA2
C
        ZTOT=SMOMLL(WTAB,FTAB,WTAB(1),WTAB(NTAB),NTAB,0,0)
        EXPT=SMOMLL(WTAB,FTAB,WTAB(1),WTAB(NTAB),NTAB,0,1)
        WRITE(6,'(A,1P,2E14.6)') '  Z, ZNUM =',ZT,ZTOT
        WRITE(6,'(A,1P,2E14.6)') '  I, INUM =',EXPOT,EXP(EXPT/ZTOT)
        RETURN
      ENDIF
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C  ****  Dense materials.
C
      WRITE(6,*) ' '
      WRITE(6,'(A,A)') '  Is the material an insulator ',
     1  'or semiconductor? (Y/N)'
      READ(5,*) YN
      IF(YN.EQ.'Y'.OR.YN.EQ.'y') THEN
 907    CONTINUE
        WRITE(6,*) '      '
        WRITE(6,*) ' Enter the gap energy (eV) ...'
        READ(5,*) WG
        WRITE(6,'(/1X,'' Gap energy ='',1P,E12.5,'' eV'')') WG
        IF(WG.LT.0.1D0) THEN
          WRITE(6,*) ' The gap energy must be larger than 0.1 eV.'
          WRITE(6,*) ' Please, enter a valid value.'
          GO TO 907
        ENDIF
      ELSE
        WG=0.0D0
      ENDIF
C
C  ****  Contribution from outer subshells, normalized to ZOUT.
C
      WTH=50.0D0
      SUMZI=SMOMLL(W0,F0,WTH,W0(N),N,0,0)
      SUMLI=SMOMLL(W0,F0,WTH,W0(N),N,0,1)
      CALL FINDI(WTH,W0,N,IC)
      IF(IC.GT.1) THEN
        FTH=FINTRP(WTH,W0,F0,N,3)
        W0(IC)=WTH
        F0(IC)=FTH
        IC=IC-1
        N=N-IC
        DO I=1,N
          W0(I)=W0(I+IC)
          F0(I)=F0(I+IC)
        ENDDO
      ENDIF
C
      SLT=ZT*LOG(EXPOT)
      S0OUT=ZT-SUMZI
      SLOUT=SLT-SUMLI
      WR=SQRT(OMEGA2*S0OUT/ZT)
 908  CONTINUE
      WRITE(6,1010) WR
 1010 FORMAT(2X,'  Outer-shell resonance =',1P,E12.5,' eV')
      WU=EBCUT
C
      TOL=1.0D-10
      WL=MAX(WG,1.0D-3)
      SG=5.0D0
 10   CONTINUE
      MOM=0
      S0O=SUMGA(FLOR,WL,WU,TOL)
      MOM=1
      RA1=S0OUT/S0O
      SLO=SUMGA(FLOR,WL,WU,TOL)*RA1
      IF(SLO.GT.SLOUT) THEN
        IF(SG.LT.0.5D0) THEN
          WR=WR*0.8D0
          GO TO 908
        ENDIF
        SG=0.5D0*SG
        GO TO 10
      ENDIF
      SGL=SG
C
      SG=5.0D0
 20   CONTINUE
      MOM=0
      S0O=SUMGA(FLOR,WL,WU,TOL)
      MOM=1
      RA1=S0OUT/S0O
      SLO=SUMGA(FLOR,WL,WU,TOL)*RA1
      IF(SLO.LT.SLOUT) THEN
        SG=2.0D0*SG
        GO TO 20
      ENDIF
      SGU=SG
C
 30   CONTINUE
      SG=0.5D0*(SGL+SGU)
      MOM=0
      S0O=SUMGA(FLOR,WL,WU,TOL)
      MOM=-1
      RA1=S0OUT/S0O
      SLO=SUMGA(FLOR,WL,WU,TOL)*RA1
      IF(SLO.LT.SLOUT) THEN
        SGL=SG
      ELSE
        SGU=SG
      ENDIF
      IF(SGU-SGL.GT.TOL*SGU) GO TO 30
      MOM=0
C
      NPM=NWM
      NFIX=0
      DW=0.01D0
      DO I=1,10
        NFIX=NFIX+1
        X1(NFIX)=WL+I*DW
      ENDDO
      DO I=1,N
        IF((W0(I).GT.WL).AND.(W0(I).LT.WU)) THEN
          NFIX=NFIX+1
          X1(NFIX)=W0(I)
        ENDIF
      ENDDO
      NU=0
      TOL=1.0D-8
      N1=MAX(NFIX+NU+1,1500)
      CALL TABLEF(FLOR,WL,WU,X1,Y1,TOL,ERR,NPM,NFIX,NU,N1,3)
      DO I=1,N1
        Y1(I)=Y1(I)*S0OUT/S0O
      ENDDO
C
      IF(WG.GT.1.0D-3) THEN
        N1=N1+2
        DO I=N1,3,-1
          X1(I)=X1(I-2)
          Y1(I)=Y1(I-2)
        ENDDO
        X1(2)=X1(3)
        Y1(2)=0.0D0
        X1(1)=X1(3)*0.5D0
        Y1(1)=0.0D0
      ENDIF
      N1=N1+1
      X1(N1)=X1(N1-1)
      Y1(N1)=0.0D0
      N1=N1+1
      X1(N1)=2.0D0*X1(N1-1)
      Y1(N1)=0.0D0
C
      N2=N
      CALL MERGE2(X1,Y1,W0,F0,X,Y,N1,N2,NWM,N,3)
      DO I=1,N
        W0(I)=X(I)
        F0(I)=Y(I)
      ENDDO
C
C  ****  Re-calculate discontinuities.
C
      DO I=1,N
        XC=W0(I)
        IF(I.LT.N.AND.I.GT.1) THEN
          IF(W0(I+1).LT.XC+EPS*ABS(XC)) XC=XC-EPS10*ABS(XC)
          IF(W0(I-1).GT.XC-EPS*ABS(XC)) XC=XC+EPS10*ABS(XC)
        ENDIF
        F0(I)=FINTRP(XC,W0,F0,N,3)
      ENDDO
C
C  ************  The OOS table is 'cleaned' so as to have exactly NPS
C        data points, from which the OOS will be calculated by linear
C        log-log interpolation.
C
      ERCUT=1.0D-5
      NPS=750
      CALL TCLEAN(W0,F0,ERCUT,ERRM,N,NPS,3)
      N=NPS
C
C  ****  Renormalization to satisfy the dipole sum rule.
C
      SUMZ=SMOMLL(W0,F0,W0(1),W0(N),N,0,0)
      RA1=ZT/SUMZ
      NTAB=N
      DO I=1,NTAB
        WTAB(I)=W0(I)
        FTAB(I)=F0(I)*RA1
      ENDDO
      ZTOT=SMOMLL(WTAB,FTAB,WTAB(1),WTAB(NTAB),NTAB,0,0)
      EXPT=SMOMLL(WTAB,FTAB,WTAB(1),WTAB(NTAB),NTAB,0,1)
      WOD=WTH
      RETURN
      END
C  *********************************************************************
      FUNCTION FLOR(W)
C  OOS of a damped classical oscillator, with an optional energy gap.
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4(I-N)
      PARAMETER (PI=3.1415926535897932D0, TWOOPI=2.0D0/PI)
      COMMON/CLOREN/WR,WG,SG,MOM
C
      IF(W.LT.WG) THEN
        FLOR=0.0D0
        RETURN
      ELSE
        T2=W*W-WG*WG
        T=SQRT(T2)
        FLOR=TWOOPI*W*SG*T/((WR*WR-T2)**2+SG*SG*T2)
      ENDIF
      IF(MOM.EQ.0) RETURN
      FLOR=FLOR*LOG(W)
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE OOSTRD
C  *********************************************************************
      SUBROUTINE OOSTRD(IZ,NSHZ)
C
C  Reads the optical oscillator strength of atomic electron subshells
C  from the database files.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4(I-N)
      CHARACTER CH1*1,CH2*2
      PARAMETER(NWM=15000)
      COMMON/CPHXS/EBIN(30),W(30,NWM),SHOOS(30,NWM),ISH(30),NW(30),NSH
C
      NSH=NSHZ
      WRITE(CH2,'(I2)') IZ
      IF(IZ.LT.10) CH2(1:1)='0'
      OPEN(7,FILE='./sdbase/oos'//CH2//'.tab')
      READ(7,'(17X,I3)') IZT ! Atomic number.
      IF(IZT.NE.IZ) THEN
        WRITE(6,*) '   IZ =',IZ
        WRITE(6,*) '  IZT =',IZT
        STOP 'OOSTRD: Corrupted oosZZ.dat file.'
      ENDIF
      READ(7,'(A1)') CH1
      READ(7,'(A1)') CH1
      NSHR=0
      DO I=1,31
        READ(7,'(A1)',END=1) CH1
        READ(7,'(A1)',END=1) CH1
        READ(7,'(1X,I4,I4,12X,E17.9,5X,I4)',END=1)
     1    IZT,ISH(I),EBINI,NWI
        IS=ISH(I)
        EBIN(IS)=EBINI
        NW(IS)=NWI
        READ(7,'(A1)') CH1
        NSHR=NSHR+1
        DO IW=1,NWI
          READ(7,*) W(IS,IW),SHOOS(IS,IW)
        ENDDO
      ENDDO
 1    CONTINUE
      CLOSE(7)
      IF(NSHR.NE.NSH) THEN
        WRITE(6,*) '   IZ =',IZ
        WRITE(6,*) '  NSH =',NSH
        WRITE(6,*) ' NSHR =',NSHR
        STOP 'OOSTRD: Corrupted oosZZ.dat file.'
      ENDIF
      RETURN
      END
C  *********************************************************************
C                       BLOCK DATA PENDAT
C  *********************************************************************
      BLOCK DATA PENDAT
C
C  Physical data for the elements Z=1-99.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      CHARACTER LASYMB*2
C
      COMMON/CADATA/ATW(99),EPX(99),RSCR(99),ETA(99),EB(99,30),
     1  ALW(99,30),CP0(99,30),IFI(99,30),IKS(99,30),NSHT(99),LASYMB(99)
C
C  ************  Chemical symbols of the elements.
C
      DATA LASYMB/' H','He','Li','Be',' B',' C',' N',' O',' F',
     1       'Ne','Na','Mg','Al','Si',' P',' S','Cl','Ar',' K',
     2       'Ca','Sc','Ti',' V','Cr','Mn','Fe','Co','Ni','Cu',
     3       'Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr',' Y',
     4       'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In',
     5       'Sn','Sb','Te',' I','Xe','Cs','Ba','La','Ce','Pr',
     6       'Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm',
     7       'Yb','Lu','Hf','Ta',' W','Re','Os','Ir','Pt','Au',
     8       'Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac',
     9       'Th','Pa',' U','Np','Pu','Am','Cm','Bk','Cf','Es'/
C
C  ************  Atomic weights (mean relative atomic masses).
C         R.D. Vocke Jr, Pure Appl. Chem. 71 (1999) 1593–1607.
      DATA ATW   /1.00794D0,4.00260D0,6.94100D0,9.01218D0,
     A  1.08110D1,1.20107D1,1.40067D1,1.59994D1,1.89984D1,
     1  2.01797D1,2.29900D1,2.43050D1,2.69815D1,2.80855D1,
     A  3.09738D1,3.20606D1,3.54527D1,3.99480D1,3.90983D1,
     2  4.00780D1,4.49559D1,4.78670D1,5.09415D1,5.19961D1,
     A  5.49380D1,5.58450D1,5.89332D1,5.86934D1,6.35460D1,
     3  6.53900D1,6.97230D1,7.26100D1,7.49216D1,7.89600D1,
     A  7.99040D1,8.38000D1,8.54678D1,8.76200D1,8.89058D1,
     4  9.12240D1,9.29064D1,9.59400D1,9.79070D1,1.01070D2,
     A  1.02906D2,1.06420D2,1.07868D2,1.12411D2,1.14818D2,
     5  1.18710D2,1.21760D2,1.27600D2,1.26905D2,1.31290D2,
     A  1.32905D2,1.37327D2,1.38906D2,1.40116D2,1.40908D2,
     6  1.44240D2,1.44910D2,1.50360D2,1.51964D2,1.57250D2,
     A  1.58925D2,1.62500D2,1.64930D2,1.67260D2,1.68934D2,
     7  1.73040D2,1.74967D2,1.78490D2,1.80948D2,1.83840D2,
     A  1.86207D2,1.90230D2,1.92217D2,1.95078D2,1.96967D2,
     8  2.00590D2,2.04383D2,2.07200D2,2.08980D2,2.08980D2,
     A  2.09990D2,2.22020D2,2.23020D2,2.26030D2,2.27030D2,
     9  2.32038D2,2.31036D2,2.38029D2,2.37050D2,2.39050D2,
     A  2.43060D2,2.47070D2,2.47070D2,2.51080D2,2.52080D2/
C
C  ************  Mean excitation energies of the elements (eV).
C
      DATA EPX / 19.2D0, 41.8D0, 40.0D0, 63.7D0, 76.0D0, 81.0D0,   ! 1
     1   82.0D0, 95.0D0,115.0D0,137.0D0,149.0D0,156.0D0,166.0D0,   ! 7
     2  173.0D0,173.0D0,180.0D0,174.0D0,188.0D0,190.0D0,191.0D0,   ! 14
     3  216.0D0,233.0D0,245.0D0,257.0D0,272.0D0,286.0D0,297.0D0,   ! 21
     4  311.0D0,322.0D0,330.0D0,334.0D0,350.0D0,347.0D0,348.0D0,   ! 28
     5  343.0D0,352.0D0,363.0D0,366.0D0,379.0D0,393.0D0,417.0D0,   ! 35
     6  424.0D0,428.0D0,441.0D0,449.0D0,470.0D0,470.0D0,469.0D0,   ! 42
     7  488.0D0,488.0D0,487.0D0,485.0D0,491.0D0,482.0D0,488.0D0,   ! 49
     8  491.0D0,501.0D0,523.0D0,535.0D0,546.0D0,560.0D0,574.0D0,   ! 56
     9  580.0D0,591.0D0,614.0D0,628.0D0,650.0D0,658.0D0,674.0D0,   ! 63
     A  684.0D0,694.0D0,705.0D0,718.0D0,727.0D0,736.0D0,746.0D0,   ! 70
     B  757.0D0,790.0D0,790.0D0,800.0D0,810.0D0,823.0D0,823.0D0,   ! 77
     C  830.0D0,825.0D0,794.0D0,827.0D0,826.0D0,841.0D0,847.0D0,   ! 84
     D  878.0D0,890.0D0,902.0D0,921.0D0,934.0D0,939.0D0,952.0D0,   ! 91
     E  966.0D0,980.0D0/                                           ! 98
C
      END


C  *********************************************************************
C                       SUBROUTINE ODTRAN
C  *********************************************************************
      SUBROUTINE ODTRAN(MNAME,Z,W,F,NTB)
C
C     This subroutine transforms the input optical oscillator strength
C  (OOS) data to allow the calculation of inelastic interaction cross
C  sections of charged particles.
C
C  Input parameters:
C    OMEGA2 .... plasma energy, in eV.
C    Z ... ..... number of electrons per molecule.
C    W(K) ...... energy loss values, in eV and in strictly non-decreas-
C                  ing order.
C    F(K) ...... OOS at W(K).
C    NTB ....... number of input data points.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      CHARACTER MNAME*15
      PARAMETER (ZERO=1.0D-90)
      PARAMETER (NT=5000,TOL=0.001D0)
      DIMENSION W(NT),F(NT),Y(NT)
      DIMENSION W0L(NT),F0L(NT)
      PARAMETER (NR=750)
      COMMON/COOST/ZT,EXPOT,RHO,VMOL,OMEGA2,WOD,ZTOT
      COMMON/CETAS/WP(NR),FT(NR),SFT(NR),WL(NR),FL(NR),ETA2(NR),ETA1(NR)
C
      ZT=Z
C
      IF(W(1).LT.1.0D-6) THEN
        WRITE(6,*) 'W(1) is too small'
        STOP 'W(1) is too small'
      ENDIF
      IF(NTB.LT.10) THEN
        WRITE(6,*) 'NTB is too small'
        STOP 'NTB is too small'
      ENDIF
C
      W0L(1)=LOG(W(1))
      F0L(1)=LOG(MAX(F(1),ZERO))
      DO K=2,NTB
        IF(W(K).GT.1.0D0) THEN
          IF(W(K).LT.W(K-1)+1.0D-5) THEN
            IF(K.LT.NTB) THEN
              IF(W(K).LT.W(K+1)-2.0D-5) THEN
                W(K)=W(K-1)+1.0D-5
              ELSE
                WRITE(6,'(A,I3,1PE13.5)') 'K, W(K) =',K-1,W(K-1)
                WRITE(6,'(A,I3,1PE13.5)') 'K, W(K) =',K,W(K)
                WRITE(6,'(A,I3,1PE13.5)') 'K, W(K) =',K+1,W(K+1)
                STOP 'Input resonance energies are too close.'
              ENDIF
            ELSE
              W(K)=W(K)*1.00001D0
            ENDIF
          ENDIF
        ENDIF
        W0L(K)=LOG(W(K))
        F0L(K)=LOG(MAX(F(K),ZERO))
      ENDDO
C
C  ************  The OOS table is 'cleaned' so as to have exactly NR
C        data points, from which the OOS will be calculated by linear
C        log-log interpolation.
      NTAB=NTB
C
      DWL=W0L(2)-W0L(1)
      IF(DWL.LT.1.0D-8) THEN
        WRITE(6,'(2X,''The second data point is too small'')')
        STOP 'The second data point is too small'
      ENDIF
C
      IF(W(1).GT.1.0D-3) THEN
        WCL=LOG(1.0D-3)
        F0L(1)=F0L(1)+(F0L(2)-F0L(1))*((WCL-W0L(1))/DWL)
        F0L(1)=MAX(F0L(1),LOG(ZERO))
        W0L(1)=WCL
      ENDIF
C
C  ****  Points that can be interpolated are removed from the grid.
 1    CONTINUE
      ERRMIN=1.0D0
      KC=0
      DO K=3,NTAB-3
        DWL=W0L(K+1)-W0L(K-1)
        DFL=F0L(K+1)-F0L(K-1)
        FINT=EXP(F0L(K-1)+DFL*(W0L(K)-W0L(K-1))/DWL)
        ERR=ABS(EXP(F0L(K))-FINT)/FINT
        IF(ERR.LT.ERRMIN) THEN
          ERRMIN=ERR
          KC=K
        ENDIF
      ENDDO
C
      IF(ERRMIN.LT.TOL.AND.KC.GT.0) THEN
        DO K=KC,NTAB-1
          W0L(K)=W0L(K+1)
          F0L(K)=F0L(K+1)
        ENDDO
        NTAB=NTAB-1
      ENDIF
      IF(ERRMIN.LT.TOL.AND.NTAB.GT.700) GO TO 1
C
C  ****  Additional grid points added to obtain a sufficiently dense
C        grid for resonance energies near the primary energy.
C
      IF(NTAB.GT.NR-30) THEN
        WRITE(6,'(/2X,''Not enough memory to define the grid.'',
     1    /2X,''Increase the parameter NR.'')')
        STOP
      ENDIF
C
      IF(EXP(W0L(NTAB)).LT.1.0D6) THEN
        NTAB=NTAB+1
        WC=1.0D7
        CALL FINDI(WC,W,NTB,I)
        IF(I.EQ.NTB) I=NTB-1
        WCL=LOG(WC)
        DWL=LOG(W(I+1))-LOG(W(I))
        DFL=LOG(F(I+1))-LOG(F(I))
        FCL=LOG(F(I))+DFL*(WCL-LOG(W(I)))/DWL
        W0L(NTAB)=WCL
        F0L(NTAB)=FCL
      ENDIF
C
      NADDED=0
      FLCUT=LOG(ZERO)
 2    CONTINUE
      IF(NTAB.LT.NR) THEN
        TST=0.0D0
        KC=NTAB
        DO K=1,NTAB-1
          TSTF=MAX(F0L(K+1),F0L(K))
          IF(TSTF.GT.-81.0D0) THEN
            TSTC=W0L(K+1)-W0L(K)
            DW=EXP(W0L(K+1))-EXP(W0L(K))
            IF(TSTC.GT.TST.AND.DW.GT.0.05D0) THEN
              TST=TSTC
              KC=K
            ENDIF
          ENDIF
        ENDDO
        IF(KC.LT.NTAB) THEN
          WC=0.5D0*(EXP(W0L(KC))+EXP(W0L(KC+1)))
        ELSE
          WC=EXP(W0L(NTAB))*1.01D0
        ENDIF
        CALL FINDI(WC,W,NTB,I)
        IF(I.EQ.NTB) I=NTB-1
        WCL=LOG(WC)
        DWL=LOG(W(I+1))-LOG(W(I))
        DFL=LOG(F(I+1))-LOG(F(I))
        FCL=LOG(F(I))+DFL*(WCL-LOG(W(I)))/DWL
        IF(FCL.LT.FLCUT) FCL=FLCUT
        IF(KC.LT.NTAB) THEN
          DO K=NTAB+1,KC+2,-1
            W0L(K)=W0L(K-1)
            F0L(K)=F0L(K-1)
          ENDDO
          W0L(KC+1)=WCL
          F0L(KC+1)=FCL
        ELSE
          KC=NTAB+1
          W0L(KC)=WCL
          F0L(KC)=FCL
        ENDIF
        NTAB=NTAB+1
        NADDED=NADDED+1
      ENDIF
      IF(NTAB.LT.NR) GO TO 2
C
C  ************  Check of sum rules.
C
      WRITE(6,6000)
 6000 FORMAT(/2X,'********  Check of sum rules:')
C
C  ****  Original data set.
C
      WRITE(6,6010)
 6010 FORMAT(/2X,'****  Original data set:')
C
      DO K=1,NTB
        Y(K)=F(K)
      ENDDO
      ZTOT=SMOMLL(W,Y,W(1),W(NTB),NTB,0,0)
      FSUM=ZTOT/ZT
      EXPOTF=SMOMLL(W,Y,W(1),W(NTB),NTB,0,1)
      EXPOTF=EXP(EXPOTF/ZTOT)
C
      WRITE(6,6020) ZT
 6020 FORMAT(2X,'Electrons/molecule ................... ',
     1 1P,E12.5)
      WRITE(6,6030) FSUM
 6030 FORMAT(2X,'f-sum ................................ ',
     1 1P,E12.5)
      WRITE(6,6040) EXPOTF
 6040 FORMAT(2X,'Mean excitation energy (calc.) ....... ',
     1 1P,E12.5,' eV')
C
C  ****  Modified data set.
C
      DO K=1,NR
        WL(K)=W0L(K)
        WP(K)=EXP(WL(K))
        Y(K)=EXP(F0L(K))
        ETA1(K)=0.0D0
        ETA2(K)=0.0D0
      ENDDO
      ZTOT=SMOMLL(WP,Y,WP(1),WP(NR),NR,0,0)
      WRITE(6,6050)
 6050 FORMAT(/2X,'****  Modified data set:')
      WRITE(6,6060) NADDED
 6060 FORMAT(2X,'Number of inserted grid points ....... ', I5)
      DFL=LOG(ZT/ZTOT)
      DO K=1,NR
        FL(K)=F0L(K)+DFL
        FT(K)=EXP(FL(K))
      ENDDO
C
      ZTOT=SMOMLL(WP,FT,WP(1),WP(NR),NR,0,0)
      FSUM=ZTOT/ZT
      EXPOTF=SMOMLL(WP,FT,WP(1),WP(NR),NR,0,1)
      EXPOTF=EXP(EXPOTF/ZTOT)
      WRITE(6,6070) FSUM
 6070 FORMAT(2X,'f-sum ................................ ',
     1 1P,E12.5)
      WRITE(6,6080) EXPOT
 6080 FORMAT(2X,'Mean excitation energy (calc.) ....... ',
     1 1P,E12.5,' eV')
C
C  ****  Print the OOS table.
C
      OPEN(8,FILE='OOS.dat')
      WRITE(8,7100) MNAME
 7100 FORMAT('# Optical oscillator strength (DHFS model).',/'#',
     1  /,'#  Material filename (.mat) ...  ',A)
      WRITE(8,7000) ZT
 7000 FORMAT('#  Electrons/molecule ......... ',1P,E11.4,/'#')
C
      FSUM=ZTOT/ZT
      WRITE(8,7010) FSUM
 7010 FORMAT('#  f-sum ...................... ',1P,E11.4)
      WRITE(8,7030) EXPOT
 7030 FORMAT('#  Mean excitation energy ..... ',1P,E11.4,' eV')
      WRITE(8,'(''#'')')
      WRITE(8,7040)
 7040 FORMAT('#     W(eV)      OOS(1/eV)     cumulative',
     1  /'# ',41('-'))
C
      SFT(1)=0.0D0
      DO K=2,NR
        SFT(K)=SFT(K-1)+SMOMLL(WP,FT,WP(K-1),WP(K),NR,0,0)
      ENDDO
C
      DO K=1,NR
        IF(ABS(FT(K)).LT.1.0D-75) FT(K)=0.0D0
        WRITE(8,'(1P,10E14.6)') WP(K),FT(K),SFT(K)
      ENDDO
      CLOSE(8)
C
      RETURN
      END


C  *********************************************************************
C                       SUBROUTINE DENSIT
C  *********************************************************************
      SUBROUTINE DENSIT
C
C     This subroutine computes the density effect correction to the
C  stopping power using Fano's integral expression.
C
      USE CONSTANTS
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
C  ****  OOS and dielectric function tables.
      PARAMETER (NR=750)
      COMMON/COOST/ZT,EXPOT,RHO,VMOL,OMEGA2,WOD,ZTOT
      COMMON/CETAS/WP(NR),FT(NR),SFT(NR),WL(NR),FL(NR),ETA2(NR),ETA1(NR)
      COMMON/CDELF/GM(NR),GML(NR),DELTA(NR),DELTAL(NR)
      DIMENSION FUN(NR),FUN2(NR)
C
      FACT=OMEGA2/ZT
      DO I=1,NR
        RL2=WP(I)**2
        DO K=1,NR
          FUN2(K)=FT(K)/(WP(K)**2+RL2)
        ENDDO
        FUN(I)=FACT*SMOMLL(WP,FUN2,WP(1),WP(NR),NR,0,0)
      ENDDO
C
      ELOW=1.0D0
      EUP=1.0D10
      FACT=EXP(LOG(EUP/ELOW)/DBLE(NR-1))
      E=ELOW/FACT
      DO I=1,NR
        E=E*FACT
        GAMMA=1.0D0+E/REV
        GAMMA2=GAMMA*GAMMA
        B21=1.0D0/GAMMA2
        IF(B21.GT.FUN(1)) THEN
          DELTA(I)=1.0D-35
        ELSE IF(B21.LT.FUN(NR)) THEN
          DELTA(I)=LOG(OMEGA2/(B21*EXPOT**2))-1.0D0
        ELSE
          XL=WP(1)
          XU=WP(NR)
 1        CONTINUE
          X=0.5D0*(XL+XU)
          CALL FINDI(X,WP,NR,J)
          DW=WP(J+1)-WP(J)
          IF(DW.GT.1.0D-8) THEN
            FOL=FUN(J)+(FUN(J+1)-FUN(J))*(X-WP(J))/DW
          ELSE
            FOL=(FUN(J)+FUN(J+1))*0.5D0
          ENDIF
          IF(FOL.GT.B21) THEN
            XL=X
          ELSE
            XU=X
          ENDIF
          IF(XU-XL.GT.1.0D-6) GO TO 1
          DO K=1,NR
            FUN2(K)=FT(K)*LOG(1.0D0+(X/WP(K))**2)
          ENDDO
          DELTA(I)=(SMOMLL(WP,FUN2,WP(1),WP(NR),NR,0,0)/ZT)
     1      -(X*X/OMEGA2)*B21
        ENDIF
        GM(I)=GAMMA2
        GML(I)=LOG(GAMMA2)
        DELTAL(I)=LOG(DELTA(I))
      ENDDO
C
      RETURN
      END


C  *********************************************************************
C                       SUBROUTINE ASACSS
C  *********************************************************************
      SUBROUTINE ASACSS(E,PMASS,CS0A,CS1A,CS2A,IZ1,JZ)
C
C     This subroutine delivers integrated cross sections for inelastic
C  collisions of fast charged particles with free neutral atoms of the
C  elements, from hydrogen (JZ=1) to einsteinium (JZ=99). It uses the
C  Bethe asymptotic formulas with parameters calculated from the DHFS
C  self-consistent potential, which are read from file 'atparams.tab'.
C
C  Input arguments:
C     E ........ kinetic energy of the projectile (in eV).
C     PMASS .... projectile mass (in units of the electron mass).
C     IZ1 ...... projectile charge (in units of the elementary charge).
C     JZ ....... atomic number of the target atom.
C
C  Output arguments:
C     CS0A ..... total inelastic cross section (cm^2).
C     CS1A ..... stopping cross section (eV*cm^2).
C     CS2A ..... energy-straggling cross section ((eV*cm)^2).
C
C  >>>>  All quantities in atomic (Hartree) units.
C
      USE CONSTANTS
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      CHARACTER CH5*5
C  ****  Kinematical constants
      COMMON/PKINEM/RMASEV,EC,ET,GAMM1,GAMMA,GAMMA2,BETA2,BETA,
     1  CP2,CP,CONS,ZEF,ICHAR
      PARAMETER (NAM=99)
      COMMON/CPARAA/ASM1(NAM),AIM1(NAM),ADM1(NAM),AS0(NAM),AI0(NAM),
     1  AD0(NAM),AS1(NAM),AI1(NAM),AD1(NAM)
C
      DATA IRDR/0/
      SAVE IRDR
C
      IF(IRDR.EQ.0) THEN
        OPEN(17,FILE='./sdbase/atparams.tab')
        READ(17,'(A)') CH5
        READ(17,'(A)') CH5
        READ(17,'(A)') CH5
        READ(17,'(A)') CH5
        READ(17,'(A)') CH5
        DO I=1,99
          READ(17,*) IZZ,PPSM1,PPIM1,PPDM1,PPS0,PPI0,PPD0,
     1      PPS1,PPI1,PPD1
          IF(I.NE.IZZ) THEN
            WRITE(6,*) '    I =',I
            WRITE(6,*) '  IZZ =',IZZ
            STOP ' The values I and IZZ are not equal.'
          ENDIF
          ASM1(I)=PPSM1; AIM1(I)=PPIM1; ADM1(I)=PPDM1
          AS0(I)=PPS0;    AI0(I)=PPI0;   AD0(I)=PPD0
          AS1(I)=PPS1;    AI1(I)=PPI1;   AD1(I)=PPD1
        ENDDO
        CLOSE(17)
*       DO I=1,99  ! Verification.
*         WRITE(87,1001) I,ASM1(I),AIM1(I),ADM1(I),AS0(I),
*    1      AI0(I),AD0(I),AS1(I),AI1(I),AD1(I)
*       ENDDO
*1001   FORMAT(1X,I3,1X,1P,10E14.6)
        IRDR=1
      ENDIF
C
      IF(ABS(E-EC).GT.1.0D-12*E) THEN
        CALL KINPAR(IZ1,PMASS,E)
        CONS=CONS/ZEF**2  ! Remove electron capture correction.
      ENDIF
C
      GLG=LOG(BETA2*GAMMA2)-BETA2
C
      IF(ABS(PMASS-1.0D0).LT.1.0D-10.AND.IZ1.EQ.-1) THEN
C  *** Electrons.
        F=(2.0D0*GAMMA2-1.0D0)/GAMMA2
     1    +0.125D0*((GAMMA-1.0D0)/GAMMA)**2
     2    -(4.0D0-((GAMMA-1.0D0)/GAMMA)**2)*LOG(2.0D0)-LOG(GAMMA+1.0D0)
        G=GAMM1*(2.50D0
     1   +((1.0D0-2.0D0*GAMMA-2.0D0*GAMMA2)/GAMMA2)*LOG(2.0D0)
     2   -11.0D0*(GAMM1/GAMMA)**2/24.0D0)
      ELSE IF(ABS(PMASS-1.0D0).LT.1.0D-10.AND.IZ1.EQ.1) THEN
C  *** Positrons.
        F=(BETA2/12.0D0)*(1.0D0-(14.0D0/(GAMMA+1.0D0))
     1    -(10.0D0/(GAMMA+1.0D0)**2)-4.0D0/(GAMMA+1)**3)
     2    -LOG(2.0D0)-LOG(GAMMA+1.0D0)
        G=GAMM1*(1.0D0-((GAMMA2-1.0D0)/(30.0D0*GAMMA2))*
     1    (9.0D0+21.0D0/(GAMMA+1.0D0)+23.0D0/(GAMMA+1.0D0)**2
     2    +8.0D0/(GAMMA+1.0D0)**3))
      ELSE IF(PMASS.GT.10.0D0) THEN
C  *** Heavy projectiles.
        R=1.0D0/(1.0D0+1.0D0/PMASS**2+2.0D0*GAMMA/PMASS)
        F=LOG(R)+((GAMMA2-1.0D0)*R/(GAMMA*PMASS))**2
        G=(GAMMA2**2-1.0D0)*R/GAMMA2
      ELSE
        WRITE(6,'(A,1P,E12.5)') 'PMASS=',PMASS
        WRITE(6,'(A,I4)') 'IZ1=',IZ1
        WRITE(6,'(1X,''This kind of particle is not allowed.'')')
        STOP
      ENDIF
C
      Z=DBLE(JZ)
      CS0A=ASM1(JZ)*(GLG+2.0D0*LOG(2.0D0*REV/AIM1(JZ)))+ADM1(JZ)
      CS1A=AS0(JZ)*(GLG+2.0D0*LOG(2.0D0*REV/AI0(JZ)))+AD0(JZ)+Z*(GLG+F)
      CS2A=AS1(JZ)*(GLG+2.0D0*LOG(2.0D0*REV/AI1(JZ)))+AD1(JZ)+Z*REV*G
      CS0A=MAX(CONS*CS0A,1.0D-75)
      CS1A=MAX(CONS*CS1A,1.0D-75)
      CS2A=MAX(CONS*CS2A,1.0D-75)
      RETURN
      END


C **********************************************************************
C                         FUNCTION BETHE
C **********************************************************************
      FUNCTION BETHE(EXPOF,CBF,E)
C
C     Calculates the stopping cross section by using the corrected Bethe
C  formula.
C
C  INPUT:
C     EXPOF .... mean excitation energy (in eV).
C     CBF ...... factor in the Barkas cutoff impact parameter.
C     E ........ kinetic energy of the projectile (in eV).
C
C  OUTPUT (log terms through common block CBETHE):
C     RL0 ...... Bethe logarithm.
C     FCOR ..... correction f(GAMMA)/2.
C     CSHC ..... DHFS shell correction.
C     CDEN ..... density-effect correction, delta_F/2.
C     CLS ...... Lindhard-Sorensen correction.
C     CBAR ..... Barkas correction.
C
      USE CONSTANTS
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
C  ****  Kinematical constants
      COMMON/PKINEM/RMASEV,EC,ET,GAMM1,GAMMA,GAMMA2,BETA2,BETA,
     1  CP2,CP,CONS,ZEF,ICHAR
      COMMON/CBRIDGE/PMASS,WMAX,IZT,IZ1
      COMMON/CBETHE/RL0,FCOR,CSHC,CDEN,CLS,CBAR
C
      PARAMETER (NR=750)
      COMMON/COOST/ZT,EXPOT,RHO,VMOL,OMEGA2,WOD,ZTOT
      COMMON/CDELF/GM(NR),GML(NR),DELTA(NR),DELTAL(NR)
C
      PARAMETER (NT=5000)
      COMMON/CSHCOR/ESHC(NT),SHC(NT),NSHC
C
      IF(ABS(E-EC).GT.1.0D-12*E) THEN
        CALL KINPAR(IZ1,PMASS,E)
      ENDIF
      SFACT=2.0D0*CONS*ZT
C  ----  Bethe logarithm
      RL0=LOG(2.0D0*REV*BETA2/EXPOF)+DLOG(GAMMA2)-BETA2
C  ----  f(GAMMA)/2.
      IF(IZ1.EQ.-1.AND.ABS(PMASS-1.0D0).LT.1.0D-3) THEN  ! Electrons.
        FCOR=(2.0D0*GAMMA2-1.0D0)/GAMMA2
     1    +0.125D0*((GAMMA-1.0D0)/GAMMA)**2
     2    -(4.0D0-((GAMMA-1.0D0)/GAMMA)**2)*LOG(2.0D0)-LOG(GAMMA+1.0D0)
      ELSE IF(IZ1.EQ.1.AND.ABS(PMASS-1.0D0).LT.1.0D-3)
     1 THEN  ! Positrons.
        FCOR=(BETA2/12.0D0)*(1.0D0-(14.0D0/(GAMMA+1.0D0))
     1    -(10.0D0/(GAMMA+1.0D0)**2)-4.0D0/(GAMMA+1)**3)
     2    -LOG(2.0D0)-LOG(GAMMA+1.0D0)
      ELSE  ! Heavier particles.
        R=1.0D0/(1.0D0+(1.0D0/PMASS**2)+2.0D0*GAMMA/PMASS)
        FCOR=LOG(R)+((GAMMA2-1.0D0)*R/(GAMMA*PMASS))**2
      ENDIF
      FCOR=0.5D0*FCOR
C  ----  Complete shell correction, interpolated.
      CSHC=FINTRP(MAX(E,ESHC(1)),ESHC,SHC,NSHC,2)
C  ----  Density-effect correction, interpolated.
      CDEN=0.5D0*FINTRP(GAMMA2,GM,DELTA,NR,3)
C  ----  Lindhard-Sorensen correction.
      IF(PMASS.GT.1.0D1) THEN
        CALL LINSOR(IZ1,PMASS,E,S1LS)
        CLS=S1LS/SFACT
      ELSE
        CLS=0.0D0
      ENDIF
C  ----  Barkas correction.
      IF(PMASS.GT.1.0D1) THEN
        CALL BARKAS(CBF,IZ1,PMASS,E,WMAX,S0BAR,S1BAR,S2BAR)
        CBAR=S1BAR/SFACT
      ELSE
        CBAR=0.0D0
      ENDIF
C
C  ****  Stopping cross section, corrected Bethe formula.
C
      BETHE=SFACT*(RL0+FCOR-CSHC-CDEN+CLS+CBAR)
      RETURN
      END


C  *********************************************************************
C                       FUNCTION BETSTP
C  *********************************************************************
      FUNCTION BETSTP(IZ1,PMASS,E)
C
C    This subroutine computes the collision stopping power for charged
C  particles of charge IZ1 and mass PMASS from the Bethe formula with
C  the density-effect correction and an ad hoc extrapolation to low
C  energies.
C
C  Input parameters:
C     IZ1 ..... projectile charge in units of the elementary charge.
C     PMASS ... mass of the projectile, in units of the electron mass.
C     E ....... kinetic energy of the projectile (eV).
C
C  The output values BETSTP is the stopping power (in eV/cm).
C
      USE CONSTANTS
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER (TREV=REV+REV)
C  ****  Kinematical constants
      COMMON/PKINEM/RMASEV,EC,ET,GAMM1,GAMMA,GAMMA2,BETA2,BETA,
     1  CP2,CP,CONS,ZEF,ICHAR
C
      PARAMETER (NR=750)
      COMMON/COOST/ZT,EXPOT,RHO,VMOL,OMEGA2,WOD,ZTOT
      COMMON/CDELF/GM(NR),GML(NR),DELTA(NR),DELTAL(NR)
C
      IF(ABS(E-EC).GT.1.0D-12*E) THEN
        CALL KINPAR(IZ1,PMASS,E)
      ENDIF
C
C  ****  Density effect correction (interpolated)
C
      IF(GAMMA2.GT.GM(NR)) THEN
        DELTAI=LOG(OMEGA2*GAMMA2/EXPOT**2)-1.0D0
      ELSE IF(GAMMA2.LT.GM(1)) THEN
        DELTAI=0.0D0
      ELSE
        CALL FINDI(GAMMA2,GM,NR,I)
        DGL=GML(I+1)-GML(I)
        IF(DGL.GT.1.0D-9) THEN
          DELTAI=EXP(DELTAL(I)
     1      +(DELTAL(I+1)-DELTAL(I))*(LOG(GAMMA2)-GML(I))/DGL)
        ELSE
          DELTAI=(DELTA(I)+DELTA(I+1))*0.5D0
        ENDIF
      ENDIF
C  ****  Electrons.
      IF(IZ1.EQ.-1.AND.ABS(PMASS-1.0D0).LT.1.0D-3) THEN
        A=SQRT(EXP(1.0D0)/32.0D0)
        FCOR=(2.0D0*GAMMA2-1.0D0)/GAMMA2
     1    +0.125D0*((GAMMA-1.0D0)/GAMMA)**2
     2    -(4.0D0-((GAMMA-1.0D0)/GAMMA)**2)*LOG(2.0D0)-LOG(GAMMA+1.0D0)
C  ****  Positrons.
      ELSE IF(IZ1.EQ.1.AND.ABS(PMASS-1.0D0).LT.1.0D-3) THEN
        A=0.5D0
        FCOR=(BETA2/12.0D0)*(1.0D0-(14.0D0/(GAMMA+1.0D0))
     1    -(10.0D0/(GAMMA+1.0D0)**2)-4.0D0/(GAMMA+1)**3)
     2    -LOG(2.0D0)-LOG(GAMMA+1.0D0)
C  ****  Heavier particles.
      ELSE
        A=PMASS/(PMASS+1.0D0)
        R=1.0D0/(1.0D0+(1.0D0/PMASS**2)+2.0D0*GAMMA/PMASS)
        FCOR=LOG(R)+((GAMMA2-1.0D0)*R/(GAMMA*PMASS))**2
      ENDIF
      FCOR=0.5D0*FCOR
C  ****  Low-energy extrapolation.
      XC=EXP(2.0D0)/A
      BETA2C=XC*EXPOT/TREV
      IF(BETA2.GT.BETA2C) THEN  ! High-energy expression.
        TFLOG=LOG(TREV*BETA2*GAMMA2/EXPOT)-BETA2
      ELSE  ! Low-energy extrapolation.
        TFLOG=4.0D0*BETA2/(BETA2C+BETA2)+LOG(GAMMA2/A)-BETA2
      ENDIF
      BETSTP=2.0D0*CONS*VMOL*ZT*(TFLOG+FCOR-0.5D0*DELTAI)
      RETURN
      END


C  *********************************************************************
C                       SUBROUTINE BARKAS
C  *********************************************************************
      SUBROUTINE BARKAS(CB,IZ1,PMASS,E,WMAX,S0BAR,S1BAR,S2BAR)
C
C     Computes the Barkas (Z_1^3) correction to the integrated cross
C  sections for charged particles. Only distant interactions.
C
C  Input arguments:
C     CB ...... empirical factor in the cutoff impact parameter.
C     IZ1 ..... projectile charge in units of the elementary charge.
C     PMASS ... mass of the projectile, in units of the electron mass.
C     E ....... kinetic energy of the projectile (eV).
C     WMAX .... maximum energy loss (eV).
C
C  Output arguments (atomic or molecular cross sections):
C     S0BAR ... Barkas correction to the total x-section (cm^2).
C     S1BAR ... Barkas corr., stopping x-section (eV cm^2).
C     S2BAR ... Barkas corr., energy straggling x-section ([eV cm]^2).
C
      USE CONSTANTS
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER (ZERO=1.0D-90)
      PARAMETER (GEULER=0.57721566490153286D0)  ! Euler constant.
      COMMON/PKINEM/RMASEV,EC,ET,GAMM1,GAMMA,GAMMA2,BETA2,BETA,
     1  CP2,CP,CONS,ZEF,ICHAR
      PARAMETER (NR=750)
      COMMON/CETAS/WP(NR),FT(NR),SFT(NR),WL(NR),FL(NR),ETA2(NR),ETA1(NR)
      PARAMETER (NRT=NR+50)
      DIMENSION W(NRT),H(NRT)
C
      IF(ABS(E-EC).GT.1.0D-12*E) THEN
        CALL KINPAR(IZ1,PMASS,E)
      ENDIF
C
      S0BAR=0.0D0; S1BAR=0.0D0; S2BAR=0.0D0
      WRIDGE=2.0D0*GAMMA2*BETA2*REV
     1  /(1.0D0+2.0D0*GAMMA/PMASS+1.0D0/PMASS**2)
      WU=MIN(WMAX,WRIDGE)
C  ----  Cutoff impact parameter.
      AC=CB*EXP(-GEULER)*A0B/(BETA*SL)
C
      IF(WU.GT.WP(1)+1.0D-3) THEN
        FBAR=AC/(BETA*GAMMA*SL*A0B*HREV)  ! In 1/eV.
        N2=0
        DO I=1,NR
          IF(WP(I).LT.WRIDGE+100.0D0) THEN
            N2=N2+1
            WW=WP(I)
            XI=FBAR*WW
            W(N2)=WW
            H(N2)=FT(I)*(ARBI1(XI)+ARBI2(XI)/GAMMA2)
          ENDIF
        ENDDO
        IF(N2.GT.2) THEN
          S0P=SMOMLL(W,H,W(1),WU,N2,0,0)
          S1P=SMOMLL(W,H,W(1),WU,N2,1,0)
          S2P=SMOMLL(W,H,W(1),WU,N2,2,0)
          FAC1=2.0D0*(CONS*ZEF*IZ1*BETA/SL)/((GAMMA*BETA2)**2*REV)
          S0BAR=S0BAR+FAC1*S0P
          S1BAR=S1BAR+FAC1*S1P
          S2BAR=S2BAR+FAC1*S2P
        ENDIF
      ENDIF
C
      IF(ABS(S0BAR).LT.ZERO) S0BAR=0.0D0  ! Numerical safety cutoffs.
      IF(ABS(S1BAR).LT.ZERO) S1BAR=0.0D0
      IF(ABS(S2BAR).LT.ZERO) S2BAR=0.0D0
C
      RETURN
      END
C  *********************************************************************
C                       FUNCTION ARBI1
C  *********************************************************************
      FUNCTION ARBI1(X)
C
C     Ashley-Ritchie-Brandt function I1(X) of the Barkas correction.
C  Reproduces the numerical values with an accuracy better than 0.1
C  percent for X up to about 15.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER (PI=3.1415926535897932D0)
C
      IF(X.LT.1.0D-10) THEN
        ARBI1=1.5D0*PI*LOG(0.375D0/X)
      ELSE IF(X.LT.0.25D0) THEN
        P1=1.796676D1
        P2=4.825194D1
        P3=4.661380D1
        P4=2.269807D1
        P5=5.409180D0
        P6=5.116173D-1
        XL=LOG(X)
        ARBI1=1.5D0*PI*LOG(0.375D0/X)-2.0D0*PI*X**2
     1    *(P1+XL*(P2+XL*(P3+XL*(P4+XL*(P5+XL*P6)))))
      ELSE IF(X.LT.2.0D0) THEN
        P1=-3.062544D0
        P2= 1.787672D1
        P3=-4.100572D1
        P4= 4.652797D1
        P5=-2.728692D1
        P6= 8.070905D0
        P7=-9.587701D-1
        XR=SQRT(1.0D0/X)
        ARBI1=(P1+XR*(P2+XR*(P3+XR*(P4+XR*(P5+XR*(P6+XR*P7))))))
     1    *XR**1.25D0
      ELSE IF(X.LT.15.01D0) THEN
        P1=-1.018033D-3
        P2= 1.080448D-1
        P3= 1.568923D0
        P4=-2.779110D0
        P5= 8.079930D0
        P6=-7.244712D0
        XR=1.0D0/X
        ARBI1=(P1+XR*(P2+XR*(P3+XR*(P4+XR*(P5+XR*P6)))))*EXP(-2.0D0*X)
      ELSE IF(X.LT.50.01D0) THEN
        P1=-2.798361D1
        P2= 3.964525D0
        P3=-5.706711D-1
        P4= 2.652851D-2
        P5=-6.442942D-4
        P6= 8.003948D-6
        P7=-4.016094D-8
        ARBI1=EXP(P1+X*(P2+X*(P3+X*(P4+X*(P5+X*(P6+X*P7))))))
      ELSE
        ARBI1=0.0D0
      ENDIF
C
      RETURN
      END
C  *********************************************************************
C                       FUNCTION ARBI2
C  *********************************************************************
      FUNCTION ARBI2(X)
C
C     Ashley-Ritchie-Brandt function I2(X) of the Barkas correction.
C  Reproduces the numerical values with an accuracy better than 0.1
C  percent for X up to about 15.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER (PI=3.1415926535897932D0)
C
      IF(X.LT.1.0D-9) THEN
        ARBI2=2.17759D0
      ELSE IF(X.LT.0.25D0) THEN
        P1=2.177590D0
        P2=5.689823D-1
        P3=1.038828D0
        P4=2.877808D-4
        XL=LOG(X)**2
        ARBI2=P1-2.0D0*PI*X**2*(P2+XL*(P3+XL*P4))
      ELSE IF(X.LT.2.0D0) THEN
        P1= 3.768512D0
        P2=-2.135702D1
        P3= 4.601060D1
        P4=-4.714780D1
        P5= 2.551946D1
        P6=-7.102352D0
        P7= 8.042960D-1
        XR=SQRT(1.0D0/X)
        ARBI2=(P1+XR*(P2+XR*(P3+XR*(P4+XR*(P5+XR*(P6+XR*P7))))))
     1    *XR**2*EXP(-1.5D0*X)
      ELSE IF(X.LT.15.01D0) THEN
        P1= 7.431717D-5
        P2= 6.662051D-2
        P3= 2.142710D0
        P4=-7.407167D0
        P5= 2.327532D1
        P6=-3.497742D1
        P7= 1.950319D1
        XR=1.0D0/X
        ARBI2=(P1+XR*(P2+XR*(P3+XR*(P4+XR*(P5+XR*(P6+XR*P7))))))
     1    *EXP(-2.0D0*X)
      ELSE IF(X.LT.50.01D0) THEN
        P1=-2.808307D1
        P2= 3.977364D0
        P3=-5.714987D-1
        P4= 2.655846D-2
        P5=-6.449171D-4
        P6= 8.010890D-6
        P7=-4.019306D-8
        ARBI2=EXP(P1+X*(P2+X*(P3+X*(P4+X*(P5+X*(P6+X*P7))))))
      ELSE
        ARBI2=0.0D0
      ENDIF
C
      RETURN
      END


C  *********************************************************************
C                       SUBROUTINE LINSOR
C  *********************************************************************
      SUBROUTINE LINSOR(IZ1,PMASS,E,S1LS)
C
C     Computes the Lindhard-Sorensen correction to the stopping cross
C  section for charged particles.
C
C  Input arguments:
C     IZ1 ..... projectile charge in units of the elementary charge.
C     PMASS ... mass of the projectile, in units of the electron mass.
C     E ....... kinetic energy of the projectile (eV).
C
C  Output argument (atomic or molecular cross section):
C     S1LS .... Lindhard-Sorensen stopping x-section (eV cm^2).
C
      USE CONSTANTS
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER (ZERO=1.0D-90)
      COMMON/PKINEM/RMASEV,EC,ET,GAMM1,GAMMA,GAMMA2,BETA2,BETA,
     1  CP2,CP,CONS,ZEF,ICHAR
      PARAMETER (NR=750)
      COMMON/CETAS/WP(NR),FT(NR),SFT(NR),WL(NR),FL(NR),ETA2(NR),ETA1(NR)
C
      IF(ABS(E-EC).GT.1.0D-12*E) THEN
        CALL KINPAR(IZ1,PMASS,E)
      ENDIF
C  ----  Bloch correction.
      SOMP=ZEF*IZ1/(SL*BETA)  ! Sommerfeld parameter.
      RL2=BLOCH(SOMP)
C  ----  Lindhard-Sorensen correction.
      IF(PMASS.GT.5.0D0) THEN
        IF(IZ1.EQ.-1) THEN
          A=-178.34D0
        ELSE IF(IZ1.EQ.-2) THEN
          A=-88.73D0
        ELSE IF(IZ1.EQ.1) THEN
          A=180.20D0
        ELSE IF(IZ1.EQ.2) THEN
          A=90.59D0
        ELSE
          STOP 'Wrong charge of the projectile.'
        ENDIF
        FLS=(1.0D0+A)/(1.0D0+1.92D0*GAMM1**1.41D0)-A
        RL2=FLS*RL2
      ENDIF
C
      S1LS=2.0D0*CONS*RL2*SFT(NR)
      IF(ABS(S1LS).LT.ZERO) S1LS=0.0D0  ! Numerical safety cutoff.
      RETURN
      END
C  *********************************************************************
C                       FUNCTION BLOCH
C  *********************************************************************
      FUNCTION BLOCH(ETA)
C
C     Computes the Bloch correction to the stopping cross section for
C  charged particles.
C
C  Input arguments:
C     ETA ..... Sommerfeld parameter.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER (GEULER=0.57721566490153286D0)  ! Euler constant.
C
      ETA2=ETA*ETA
      IF(ETA2.LT.0.1D0) THEN
        BLOCH=-ETA2*(1.20205683D0-ETA2*(1.03691432D0-ETA2*(1.00756875D0
     1    -ETA2*(0.98268318D0-ETA2*0.78219262D0))))
      ELSE IF(ETA2.GT.10.0D0) THEN
        BLOCH=-GEULER-0.5D0*LOG(ETA2)-1.0D0/(12.0D0*ETA2)
     1    -1.0D0/(120.0D0*ETA2**2)-1.0D0/(252.0D0*ETA2**3)
     2    -1.0D0/(240.0D0*ETA2**4)
      ELSE
        SUM=0.0D0
        DO N=1,5000000
          TERM=ETA2/(N*(N*N+ETA2))
          SUM=SUM+TERM
          IF(TERM.LT.1.0D-12*SUM) GO TO 1
        ENDDO
 1      CONTINUE
        BLOCH=-SUM
      ENDIF
      RETURN
      END


C  *********************************************************************
C                       SUBROUTINE EBRIN
C  *********************************************************************
      SUBROUTINE EBRIN
C
C  This subroutine generates a table of the radiative stopping cross
C  section for electrons. Data are read from the files 'pdebrZZ.p08'.
C
      USE CONSTANTS
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      CHARACTER CH2*2
C  ****  Composition data.
      COMMON/COMPOS/STF(30),AW,IZ(30),NELEM
C  ****  Bremsstrahlung emission.
      PARAMETER (NBE=57, NBW=32)
      COMMON/CBREMS/ET(NBE),EL(NBE),BL(NBE),A(NBE),B(NBE),C(NBE),D(NBE),
     1  ZBR2
      DIMENSION WB0(NBW),PDF(NBE),TXS(NBE)
      DATA WB0/1.0D-12,0.025D0,0.05D0,0.075D0,0.1D0,0.15D0,0.2D0,0.25D0,
     1   0.3D0,0.35D0,0.4D0,0.45D0,0.5D0,0.55D0,0.6D0,0.65D0,0.7D0,
     2   0.75D0,0.8D0,0.85D0,0.9D0,0.925D0,0.95D0,0.97D0,0.99D0,
     3   0.995D0,0.999D0,0.9995D0,0.9999D0,0.99995D0,0.99999D0,1.0D0/
C
C  ****  Building the molecular scaled cross section table.
C
      DO IE=1,NBE
        TXS(IE)=0.0D0
      ENDDO
C
C  ****  'Equivalent' atomic number.
C
      SUMZ2=0.0D0
      SUMS=0.0D0
      DO IEL=1,NELEM
        SUMZ2=SUMZ2+STF(IEL)*IZ(IEL)**2
        SUMS=SUMS+STF(IEL)
      ENDDO
      ZBR2=SUMZ2/SUMS
C
      DO IEL=1,NELEM
        IZZ=IZ(IEL)
        WGHT=STF(IEL)*IZZ*IZZ
        WRITE(CH2,'(I2)') IZZ
        IF(IZZ.LT.10) CH2(1:1)='0'
        OPEN(7,FILE='./sdbase/pdebr'//CH2//'.p08')
        READ(7,*) IZZZ
        IF(IZZZ.NE.IZZ) STOP 'EBRIN. Corrupt file.'
        DO IE=1,NBE
          READ(7,1001) ET(IE),(PDF(IW),IW=1,NBW),TXSP
 1001     FORMAT(E9.2,5E12.5,/9X,5E12.5,/9X,5E12.5,/9X,5E12.5,
     1      /9X,5E12.5,/9X,5E12.5,/9X,2E12.5,36X,E10.3)
          TXS(IE)=TXS(IE)+WGHT*TXSP
        ENDDO
        CLOSE(7)
      ENDDO
C
C  ****  Interpolation in E.
C
      DO IE=1,NBE
        EL(IE)=LOG(ET(IE))
        BL(IE)=LOG(TXS(IE))
      ENDDO
      CALL SPLINE(EL,BL,A,B,C,D,0.0D0,0.0D0,NBE)
C
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE EBRSTP
C  *********************************************************************
      SUBROUTINE EBRSTP(IPROJ,E,RSTP,RSTPU)
C
C    This subroutine computes the radiative stopping power for electrons
C  and positrons using the integrated scaled cross sections PHIRAD from
C  the database of bremsstrahlung emission data compiled by Seltzer and
C  Berger.
C
C  Input parameters:
C     IPROJ ... 1: electron; 2: positron.
C     E ....... kinetic energy of the projectile (eV).
C
C  Output values:
C     RSTP .... stopping power (eV/cm).
C     RSTPU ... mass stopping power (in MeV/mtu, 1 mtu=1 g/cm^2).
C
      USE CONSTANTS
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
*TTTT PARAMETER (ELRAD=HREV*A0B/REV, ER2A=ELRAD**2/SL)
      PARAMETER (NBE=57)
      COMMON/CBREMS/ET(NBE),EL(NBE),BL(NBE),A(NBE),B(NBE),C(NBE),D(NBE),
     1  ZBR2
      COMMON/COOST/ZT,EXPOT,RHO,VMOL,OMEGA2,WOD,ZTOT
C
C
      FACT=(RECL**2/SL)*(E+REV)
      IF(IPROJ.EQ.2) THEN
C  ****  Positron correction factor.
        T=LOG(1.0D0+1.0D6*E/(REV*ZBR2))
        FPOS=1.0D0-EXP(-T*(1.2359D-1-T*(6.1274D-2-T*(3.1516D-2-T
     1    *(7.7446D-3-T*(1.0595D-3-T*(7.0568D-5-T*1.8080D-6)))))))
        FACT=FACT*FPOS
      ENDIF
C
      ELL=LOG(E)
      RSTP=FACT*EXP(SPLVAL(ELL,EL,A,B,C,D,NBE))*VMOL
C  ****  Stopping power in MeV/mtu
      RSTPU=RSTP/(1.0D6*RHO)
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE RMRIN
C  *********************************************************************
      SUBROUTINE RMRIN
C
C  This subroutine generates a table of the radiative stopping cross
C  section for muons. Data are read from the files 'rmuonZZ.tab'.
C
      USE CONSTANTS
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      CHARACTER LASYMB*2,CH2*2,TITLE*50
C
      COMMON/CADATA/ATW(99),EPX(99),RSCR(99),ETA(99),EB(99,30),
     1  ALW(99,30),CP0(99,30),IFI(99,30),IKS(99,30),NSHT(99),LASYMB(99)
C  ****  Composition data.
      COMMON/COMPOS/STF(30),AW,IZ(30),NELEM
      COMMON/COOST/ZT,EXPOT,RHO,VMOL,OMEGA2,WOD,ZTOT
C
      COMMON/CMUON/ETL(16),DMOL(16),AR(16),BR(16),CR(16),DR(16)
C
C  ****  Building the molecular radiative cross section table.
C
      IZZ=IZ(1)
      WRITE(CH2,'(I2)') IZZ
      IF(IZZ.LT.10) CH2(1:1)='0'
      OPEN(7,FILE='./sdbase/rmuon'//CH2//'.tab')
      FACT=ATW(IZZ)/AVOG
      READ(7,'(2X,I2)') IZ1
      IF(IZ1.NE.IZZ) THEN
        WRITE(6,*) 'IZ1=',IZ1
        WRITE(6,*) 'IZZ=',IZZ
        STOP 'IZZ.NE.IZ1.'
      ENDIF
      READ(7,'(A)') TITLE
      DO JE=1,16
        READ(7,*) EE,A1,A2,A3,BTOT
        ETL(JE)=LOG(EE)
        DMOL(JE)=STF(1)*FACT*BTOT
      ENDDO
      IF(NELEM.GT.1) THEN
        DO IEL=2,NELEM
          IZZ=IZ(IEL)
          WRITE(CH2,'(I2)') IZZ
          IF(IZZ.LT.10) CH2(1:1)='0'
          OPEN(7,FILE='./sdbase/rmuon'//CH2//'.tab')
          FACT=ATW(IZZ)/AVOG
          READ(7,'(2X,I2)') IZ1
          IF(IZ1.NE.IZZ) THEN
            WRITE(6,*) 'IZ1=',IZ1
            WRITE(6,*) 'IZZ=',IZZ
            STOP 'IZZ.NE.IZ1.'
          ENDIF
          READ(7,'(A)') TITLE
          DO JE=1,16
            READ(7,*) EE,A1,A2,A3,BTOT
            DMOL(JE)=DMOL(JE)+STF(IEL)*FACT*BTOT
          ENDDO
        ENDDO
      ENDIF
C  ****  Spline interpolation in the log of the total energy.
      DO I=1,16
        DMOL(I)=DMOL(I)*VMOL
      ENDDO
      CALL SPLIN0(ETL,DMOL,AR,BR,CR,DR,0.0D0,0.0D0,16)
C
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE RMRSTP
C  *********************************************************************
      SUBROUTINE RMRSTP(IPROJ,E,RSTP,RSTPU)
C
C    This subroutine computes the radiative stopping power for muons by
C  using the molecular cross sections derived from the tables given by
C  Groom et al. (2001).
C
C  Input parameters:
C     IPROJ ... 3: muon; 4: antimuon.
C     E ....... kinetic energy of the projectile (eV).
C
C  Output values:
C     RSTP .... stopping power (eV/cm).
C     RSTPU ... mass stopping power (in MeV/mtu, 1 mtu=1 g/cm^2).
C
      USE CONSTANTS
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER(PMASS=206.7682830D0)
      COMMON/COOST/ZT,EXPOT,RHO,VMOL,OMEGA2,WOD,ZTOT
      COMMON/PKINEM/RMASEV,EC,ETT,GAMM1,GAMMA,GAMMA2,BETA2,BETA,
     1  CP2,CP,CONS,ZEF,ICHAR
      COMMON/CMUON/ETL(16),DMOL(16),AR(16),BR(16),CR(16),DR(16)
C
      IF(IPROJ.NE.3.AND.IPROJ.NE.4) THEN
        WRITE(6,*) 'IPROJ=',IPROJ
        STOP 'RMRSTP: What do you mean?'
      ENDIF
C
      ET=E+RMASEV
      IF(ET.LT.500.0D6) THEN
        RSTP=0.0D0
      ELSE
        ETCL=LOG(ET)
        RSTP=ET*SPLVAL(ETCL,ETL,AR,BR,CR,DR,16)*1.0D-6
      ENDIF
      RSTPU=RSTP/(1.0D6*RHO)
      RETURN
      END


C  *********************************************************************
C                       SUBROUTINE KINPAR
C  *********************************************************************
      SUBROUTINE KINPAR(IZ1,PMASS,E)
C
C     This subroutine sets the kinematical parameters for a particle of
C  relative mass PMASS (in units of the electron mass) and kinetic
C  energy E (in eV).
C
      USE CONSTANTS
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER(PI=3.1415926535897932D0)
      COMMON/PKINEM/RMASEV,EC,ET,GAMM1,GAMMA,GAMMA2,BETA2,BETA,
     1  CP2,CP,CONS,ZEF,ICHAR
C
      RMASEV=PMASS*REV
      EC=E
      ET=E+RMASEV
      GAMM1=E/RMASEV
      GAMMA=1.0D0+GAMM1
      GAMMA2=GAMMA**2
      CP2=E*(ET+RMASEV)
      BETA2=CP2/ET**2
      BETA=SQRT(BETA2)
      CP=SQRT(CP2)
C
      IF(ICHAR.EQ.1) THEN
        ZEF=(1.0D0-EXP(-BETA*SL/DBLE(IZ1)))
*       WRITE(35,'(A,1P,5E14.6)') 'E,ZEF=',E,ZEF
      ELSE
        ZEF=1.0D0
      ENDIF
C
      CONS=2.0D0*PI*(ZEF*DBLE(IZ1)*HREV*A0B)**2/(REV*BETA2)
      RETURN
      END


C  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
C
C                           Numerical tools
C
C  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

C  *********************************************************************
C                       FUNCTION SUMGA
C  *********************************************************************
      FUNCTION SUMGA(FCT,XL,XU,TOL)
C
C     This function calculates the value SUMGA of the integral of the
C  (external) function FCT over the interval (XL,XU) using the 20-point
C  Gauss-Legendre quadrature method with an adaptive-bisection scheme.
C
C  TOL is the tolerance, i.e. maximum allowed relative error; it should
C  not be less than 1.0D-13. A warning message is written in unit 6 when
C  the required accuracy is not attained. The common block CSUMGA can be
C  used to transfer to the calling program the error ERR, the error flag
C  IERGA, and the number NCALL of calculated function values.
C
C                                    Francesc Salvat. 17 February, 2020.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER (NP=10, NP2=2*NP, NP4=4*NP, NOIT=256, NCALLT=200000)
      DIMENSION X(NP),W(NP),XM(NP),XP(NP)
      DIMENSION S(NOIT),SN(NOIT),XR(NOIT),XRN(NOIT)
C  Output error codes:
C     IERGA = 0, no problem, the calculation has converged.
C           = 1, too many open subintervals.
C           = 2, too many function calls.
      COMMON/CSUMGA/ERR,IERGA,NCALL  ! Error, code, function calls.
      DATA IWR/0/
C
C  ****  Gauss-Legendre 20-point quadrature formula.
C  Abscissas.
      DATA X/7.6526521133497334D-02,2.2778585114164508D-01,
     1       3.7370608871541956D-01,5.1086700195082710D-01,
     2       6.3605368072651503D-01,7.4633190646015079D-01,
     3       8.3911697182221882D-01,9.1223442825132591D-01,
     4       9.6397192727791379D-01,9.9312859918509492D-01/
C  Weights.
      DATA W/1.5275338713072585D-01,1.4917298647260375D-01,
     1       1.4209610931838205D-01,1.3168863844917663D-01,
     2       1.1819453196151842D-01,1.0193011981724044D-01,
     3       8.3276741576704749D-02,6.2672048334109064D-02,
     4       4.0601429800386941D-02,1.7614007139152118D-02/
C
      DO I=1,NP
        XM(I)=1.0D0-X(I)
        XP(I)=1.0D0+X(I)
      ENDDO
C  ****  Global and partial tolerances.
      TOLG=MIN(MAX(TOL,1.0D-13),1.0D-5)  ! Global tolerance.
      SUMGA=0.0D0
      IERGA=0
      ERRP=0.0D0
C  ****  Straight integration from XL to XU.
      H=XU-XL
      HH=0.5D0*H
      X1=XL
      SP=W(1)*(FCT(X1+XM(1)*HH)+FCT(X1+XP(1)*HH))
      DO J=2,NP
        SP=SP+W(J)*(FCT(X1+XM(J)*HH)+FCT(X1+XP(J)*HH))
      ENDDO
      S(1)=SP*HH
      XR(1)=X1
      NCALL=NP2
      NOI=1
      IDONE=1  ! To prevent a compilation warning.
C
C  ****  Adaptive-bisection scheme.
C
 1    CONTINUE
      H=HH  ! Subinterval length.
      HH=0.5D0*H
      SUMR=0.0D0
      NOIP=NOI
      NOI=0
      ERRPA=ERRP
      ERRP=0.0D0
      DO I=1,NOIP
        SI=S(I)  ! Bisect the I-th open interval.
C
        X1=XR(I)
        SP=W(1)*(FCT(X1+XM(1)*HH)+FCT(X1+XP(1)*HH))
        DO J=2,NP
          SP=SP+W(J)*(FCT(X1+XM(J)*HH)+FCT(X1+XP(J)*HH))
        ENDDO
        S1=SP*HH
C
        X2=X1+H
        SP=W(1)*(FCT(X2+XM(1)*HH)+FCT(X2+XP(1)*HH))
        DO J=2,NP
          SP=SP+W(J)*(FCT(X2+XM(J)*HH)+FCT(X2+XP(J)*HH))
        ENDDO
        S2=SP*HH
C
        IDONE=I
        NCALL=NCALL+NP4
        S12=S1+S2  ! Sum of integrals on the two subintervals.
        IF(ABS(S12-SI).LT.MAX(TOLG*ABS(S12),1.0D-35)) THEN
C  ****  The integral over the parent interval has converged.
          SUMGA=SUMGA+S12
        ELSE
          ERRP=ERRP+ABS(S12-SI)
          SUMR=SUMR+S12
          NOI=NOI+2
          IF(NOI.LT.NOIT) THEN
C  ****  Store open intervals.
            SN(NOI-1)=S1
            XRN(NOI-1)=X1
            SN(NOI)=S2
            XRN(NOI)=X2
          ELSE
C  ****  Too many open intervals.
            IERGA=1
            GO TO 2
          ENDIF
        ENDIF
        IF(NCALL.GT.NCALLT) THEN
C  ****  Too many calls to FCT.
          IERGA=2
          GO TO 2
        ENDIF
      ENDDO
C
C  ****  Analysis of partial results and error control.
C
      IF(IERGA.EQ.0) THEN
        IF(ABS(SUMR).LT.MAX(TOLG*ABS(SUMGA+SUMR),1.0D-35).
     1    OR.NOI.EQ.0) THEN
          ERR=TOLG
          SUMGA=SUMGA+SUMR
          RETURN
        ELSE
          DO I=1,NOI
            S(I)=SN(I)
            XR(I)=XRN(I)
          ENDDO
          GO TO 1
        ENDIF
      ENDIF
C
C  ****  Warning (low accuracy) message.
C
 2    CONTINUE
      IF(IDONE.LT.NOIP) THEN
        DO I=IDONE+1,NOIP
          SUMR=SUMR+S(I)
        ENDDO
        NOI=NOI+(NOIP-IDONE)
      ENDIF
      ERR=ERRPA+TOLG*ABS(SUMGA)
      SUMGA=SUMGA+SUMR
      IF(ERR.LT.10.0D0*TOLG*ABS(SUMGA)) THEN
        IF(ABS(SUMGA).GT.1.0D-16) ERR=ERR/ABS(SUMGA)
        IERGA=0
        RETURN
      ENDIF
      IF(IWR.GT.0) WRITE(IWR,11)
 11   FORMAT(/2X,'>>> SUMGA. Gauss adaptive-bisection quadrature.')
      IF(IWR.GT.0) WRITE(IWR,12) XL,XU,TOL
 12   FORMAT(2X,'XL =',1P,E15.8,', XU =',E15.8,', TOL =',E8.1)
      IF(ABS(SUMGA).GT.1.0D-16) THEN
        ERR=ERR/ABS(SUMGA)
        IF(IWR.GT.0) WRITE(IWR,13) SUMGA,ERR
 13     FORMAT(2X,'SUMGA =',1P,E22.15,', relative error =',E8.1)
      ELSE
        IF(IWR.GT.0) WRITE(IWR,14) SUMGA,ERR
 14     FORMAT(2X,'SUMGA =',1P,E22.15,', absolute error =',E8.1)
      ENDIF
      IF(IWR.GT.0) WRITE(IWR,15) NCALL,NOI,HH
 15   FORMAT(2X,'NCALL =',I6,', open subintervals =',I4,', H =',
     1  1P,E10.3)
      IF(IERGA.EQ.1) THEN
        IF(IWR.GT.0) WRITE(IWR,16)
 16     FORMAT(2X,'IERGA = 1, too many open subintervals.')
      ELSE IF(IERGA.EQ.2) THEN
        IF(IWR.GT.0) WRITE(IWR,17)
 17     FORMAT(2X,'IERGA = 2, too many function calls.')
      ELSE IF(IERGA.EQ.3) THEN
        IF(IWR.GT.0) WRITE(IWR,18)
 18     FORMAT(2X,'IERGA = 3, subintervals are too narrow.')
      ENDIF
      IF(IWR.GT.0) WRITE(IWR,19)
 19   FORMAT(2X,'WARNING: the required accuracy has not been ',
     1  'attained.'/)
      RETURN
      END


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
CCCCCCCCCCCCC       Tables and simple interpolations       CCCCCCCCCCCCC
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C
C     SUBROUTINE TABLEF(FCT,XL,XU,X,Y,TOL,ERR,NPM,NFIX,NU,NP,MODE)
C       FUNCTION FINTRP(XC,X,Y,N,MODE)
C     SUBROUTINE TCLEAN(X,Y,ERCUT,ERRM,NP,NPS,MODE)
C     SUBROUTINE MERGE2(X1,Y1,X2,Y2,X,Y,N1,N2,NPM,N,MODE)
C     SUBROUTINE SORT2(X,Y,N)
C       FUNCTION SMOMLL(X,Y,XL,XU,NP,MOM,ILOG)
C
C  *********************************************************************
C                       SUBROUTINE TABLEF
C  *********************************************************************
      SUBROUTINE TABLEF(FCT,XL,XU,X,Y,TOL,ERR,NPM,NFIX,NU,NP,MODE)
C
C     This subroutine builds a table of the external function FCT(X)
C  (provided by the user) in the interval (XL,XU). The grid consists of
C  a first subgrid with NU equally spaced points and NP-NU additional
C  points that concentrate in regions where the function has the largest
C  'curvature'. Optionally a number NFIX of X-values can be fixed; they
C  are to be entered in the first NFIX positions of the input array X().
C  TOL is the tolerance; the subroutine stops when the table is such
C  that the largest relative error of the interpolation is less than
C  TOL.
C
C  Input arguments:
C    FCT ...... name of the external function.
C    XL,XU .... end points of the considered interval.
C    TOL ...... tolerance, desired relative error of the interpolation.
C    NPM ...... physical dimension of arrays X and Y.
C    NFIX ..... number of fixed points. Their abscissas must be entered
C               as the first NFIX elements of the array X. A duplicated
C               value is regarded as a discontinuity.
C    NU ....... number of points in the initial uniform subgrid.
C    NP ....... desired number of points in the table. Must be larger
C               than NFIX+NU.
C    MODE ..... interpolation mode:
C                    <1 or >7 => linear [Y is linear in X],
C                    1 => lin-log linear [LOG(Y) is linear in X],
C                    2 => log-lin linear [Y is linear in LOG(X)],
C                    3 => log-log linear [LOG(Y) is linear in LOG(X)],
C                    4 => 4-point Lagrange,
C                    5 => lin-log 4-point Lagrange,
C                    6 => log-lin 4-point Lagrange,
C                    7 => log-log 4-point Lagrange.
C
C  Output arguments:
C    X(1:NP),Y(1:NP) ... generated arrays of abscisas and function
C               values.
C    ERR ...... estimate of the interpolation error; relative error if
C               log(FCT) is interpolated, absolute error otherwise.
C    NP ....... number of points in the generated grid. It may be less
C               than the input value, because the subroutine stops
C               adding grid points as soon as the required tolerance is
C               reached.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER (EPS=1.0D-10, RES=1.0D-3, ZERO=1.0D-60)
      DIMENSION X(NPM),Y(NPM),IND(100)
C
      TTOL=MAX(1.0D-8,TOL)
      NFFIX=MAX(NFIX,0)
      IF(NP.LE.NFFIX+ABS(NU)) THEN
        WRITE(6,*) '  NP =',NP
        WRITE(6,*) 'NFIX =',NFFIX
        WRITE(6,*) '  NU =',ABS(NU)
        WRITE(6,*) ' NPR =',MAX(NFFIX+ABS(NU),NP)
        STOP 'TABLEF: NPM must be larger than NPR.'
      ENDIF
C
      IF(XU.LT.XL*1.00001D0) THEN
        WRITE(6,*) 'XL =',XL
        WRITE(6,*) 'XU =',XU
        STOP 'TABLEF: XU must be larger than XL.'
      ENDIF
C
      IF((MODE.EQ.2).OR.(MODE.EQ.3).OR.(MODE.EQ.6).OR.(MODE.EQ.7)) THEN
        IF(XL.LT.1.0D-90) THEN
          WRITE(6,*) 'XL =',XL
          STOP 'TABLEF: A log scale extending to zero?'
        ENDIF
        LOGX=1
      ELSE
        LOGX=0
      ENDIF
      IF((MODE.EQ.1).OR.(MODE.EQ.3).OR.(MODE.EQ.5).OR.(MODE.EQ.7)) THEN
        LOGY=1
      ELSE
        LOGY=0
      ENDIF
C
C  ****  Fixed grid points.
C
      N=0
      IF(NFFIX.GT.0) THEN
        DO I=1,NFFIX
          Y(I)=X(I)
        ENDDO
        DO I=1,NFFIX
          IF(Y(I).GT.XL.AND.Y(I).LT.XU) THEN
            N=N+1
            X(N)=Y(I)
          ENDIF
        ENDDO
      ENDIF
      X(N+1)=XL
      X(N+2)=XU
      N=N+2
C
C  ****  Uniform subgrid.
C
      NNU=MAX(ABS(NU),NP/4)
      IF(LOGX.EQ.1) THEN
        FACT=(XU/XL)**(1.0D0/DBLE(NNU+1))
        DO I=1,NNU
          XT=XL*FACT**I
          TT=2.0D0*EPS*(ABS(XT)+1.0D0)
          IADD=1
          DO J=1,N
            IF(ABS(XT-X(J)).LT.TT) IADD=0
          ENDDO
          IF(IADD.EQ.1) THEN
            N=N+1
            X(N)=XT
          ENDIF
        ENDDO
      ELSE
        DX=(XU-XL)/DBLE(NNU+1)
        DO I=1,NNU
          XT=XL+DX*I
          TT=2.0D0*EPS*(ABS(XT)+1.0D0)
          IADD=1
          DO J=1,N
            IF(ABS(XT-X(J)).LT.TT) IADD=0
          ENDDO
          IF(IADD.EQ.1) THEN
            N=N+1
            X(N)=XT
          ENDIF
        ENDDO
      ENDIF
C
C  ****  Sort the grid points in non-decreasing order
C
      DO I=1,N-1
        DO J=I+1,N
          IF(X(I).GT.X(J)) THEN
            SAVE=X(I)
            X(I)=X(J)
            X(J)=SAVE
          ENDIF
        ENDDO
      ENDDO
C  ****  ... and clean the grid.
 1    CONTINUE
      IF(X(2)-X(1).LT.EPS*ABS(X(1))) THEN
        N=N-1
        DO J=2,N
          X(J)=X(J+1)
        ENDDO
        GO TO 1
      ENDIF
C
 2    CONTINUE
      IF(X(N)-X(N-1).LT.EPS*ABS(X(N))) THEN
        N=N-1
        X(N)=X(N+1)
        GO TO 2
      ENDIF
C  ****  Isolate the discontinuities.
      I0=2
 3    CONTINUE
      DO I=I0,N-2
        IF(X(I+1)-X(I).LT.EPS*ABS(X(I))) THEN
          X(I+1)=X(I)
          IF(I.LT.N-2) THEN
            IF(X(I+2)-X(I).LT.EPS*ABS(X(I))) THEN
              N=N-1
              DO J=I+2,N
                X(J)=X(J+1)
              ENDDO
            ENDIF
            I0=I+1
            GO TO 3
          ENDIF
        ENDIF
      ENDDO
C  ****  Remove clustered points.
      I0=2
 4    CONTINUE
      DO I=I0,N-1
        IF((X(I)-X(I-1).GT.EPS*ABS(X(I))).AND.
     1     (X(I+1)-X(I).GT.EPS*ABS(X(I)))) THEN
          TST=MIN(X(I)-X(I-1),X(I+1)-X(I))
          IF(TST.LT.RES*ABS(X(I))) THEN
            N=N-1
            DO J=I,N
              X(J)=X(J+1)
            ENDDO
            IF(I.LT.N-1) THEN
              I0=I
              GO TO 4
            ENDIF
          ENDIF
        ENDIF
      ENDDO
C  ****  Locate the discontinuities.
 5    CONTINUE
      NDISC=1
      IND(1)=0
      DO I=2,N-2
        IF(X(I+1)-X(I).LT.EPS*ABS(X(I))) THEN
          NDISC=NDISC+1
          IND(NDISC)=I
        ENDIF
      ENDDO
      NDISC=NDISC+1
      IND(NDISC)=N
C  ****  Insert a few grid points between each pair of discontinuities.
      DO ID=1,NDISC-1
        IL=IND(ID)+1
        IH=IND(ID+1)
        NIS=IH-IL+1
        IF(NIS.LT.6) THEN
          IF(NIS.EQ.2) THEN
            XX=X(IL)
            DX=(X(IH)-XX)/5.0D0
            X(N+1)=XX+DX
            X(N+2)=XX+2.0D0*DX
            X(N+3)=XX+3.0D0*DX
            X(N+4)=XX+4.0D0*DX
            N=N+4
          ELSE
            DO J=IL,IH-1
              N=N+1
              X(N)=X(J)+0.25D0*(X(J+1)-X(J))
              N=N+1
              X(N)=X(J)+0.50D0*(X(J+1)-X(J))
              N=N+1
              X(N)=X(J)+0.75D0*(X(J+1)-X(J))
            ENDDO
          ENDIF
C
          DO I=1,N-1
            DO J=I+1,N
              IF(X(I).GT.X(J)) THEN
                SAVE=X(I)
                X(I)=X(J)
                X(J)=SAVE
              ENDIF
            ENDDO
          ENDDO
          GO TO 5
        ENDIF
      ENDDO
C
C  ****  Function values at the grid points.
C
      DO I=1,N
        IF(I.GT.1.AND.I.LT.N) THEN
          IF(X(I)-X(I-1).LT.RES) THEN
            XX=X(I)+ABS(X(I))*EPS
          ELSE IF(X(I+1)-X(I).LT.RES) THEN
            XX=X(I)-ABS(X(I))*EPS
          ELSE
            XX=X(I)
          ENDIF
        ELSE
          XX=X(I)
        ENDIF
        Y(I)=FCT(XX)
      ENDDO
C
      DO I=1,N
        IF(LOGY.EQ.1) THEN
          IF(Y(I).LT.0.0D0) THEN
            WRITE(6,*) '     X =',X(I)
            WRITE(6,*) 'FCT(X) =',YY
            STOP 'TABLEF: FCT must be non-negative.'
          ENDIF
          Y(I)=LOG(MAX(Y(I),ZERO))
        ENDIF
        IF(LOGX.EQ.1) X(I)=LOG(X(I))
      ENDDO
      TCUT=0.01D0*(X(N)-X(1))/DBLE(NP)
C
C  ****  Search for the largest interpolation error.
C
 6    CONTINUE
      IMAX=2
      ERR=-1.0D0
      DO K=1,NDISC-1
        IL=IND(K)+1
        IH=IND(K+1)
        DO 7 I=IL+1,IH-1
          XC=X(I)
          DX=MIN(XC-X(I-1),X(I+1)-XC)
          IF(DX.LT.TCUT) GO TO 7
          IF((MODE.GT.3).AND.(MODE.LT.8)) THEN
            IF(I.EQ.IL+1) THEN
              X0=X(IL);   Y0=Y(IL)
              X1=X(IL+2); Y1=Y(IL+2)
              X2=X(IL+3); Y2=Y(IL+3)
              X3=X(IL+4); Y3=Y(IL+4)
            ELSE IF(I.EQ.IH-1) THEN
              X0=X(IH-4); Y0=Y(IH-4)
              X1=X(IH-3); Y1=Y(IH-3)
              X2=X(IH-2); Y2=Y(IH-2)
              X3=X(IH)  ; Y3=Y(IH)
            ELSE
              X0=X(I-2); Y0=Y(I-2)
              X1=X(I-1); Y1=Y(I-1)
              X2=X(I+1); Y2=Y(I+1)
              X3=X(I+2); Y3=Y(I+2)
            ENDIF
            YP1=(Y1-Y0)/(X1-X0)
            YP2=(Y2-Y1)/(X2-X1)
            A2=(((Y3-Y2)/(X3-X2))-YP2)/(X3-X1)
            A3=(A2-((YP2-YP1)/(X2-X0)))/(X3-X0)
            A2=A2-A3*(X3+X2+X1)
            A1=YP1-A2*(X0+X1)-A3*(X1*(X1+X0)+X0*X0)
            A0=Y0-X0*(A1+X0*(A2+X0*A3))
            F=A0+XC*(A1+XC*(A2+XC*A3))
          ELSE
            X0=X(I-1); Y0=Y(I-1)
            X1=X(I+1); Y1=Y(I+1)
            F=Y0+(XC-X0)*(Y1-Y0)/(X1-X0)
          ENDIF
          ERRP=ABS(F-Y(I))
          IF(ERRP.GT.ERR) THEN
            ERR=ERRP
            IMAX=I
          ENDIF
 7      CONTINUE
      ENDDO
      IF(ERR.LT.TTOL) GO TO 8
C
C  ****  A new grid point is added to the table.
C
      IF(X(IMAX)-X(IMAX-1).LT.X(IMAX+1)-X(IMAX)) IMAX=IMAX+1
C
      N=N+1
      DO I=N,IMAX+1,-1
        X(I)=X(I-1)
        Y(I)=Y(I-1)
      ENDDO
      X(IMAX)=(X(IMAX-1)+X(IMAX+1))*0.5D0
      IF(NDISC.GT.0) THEN
        DO KK=1,NDISC
          IF(IND(KK).GE.IMAX) IND(KK)=IND(KK)+1
        ENDDO
      ENDIF
C
      IF(LOGX.EQ.1) THEN
        YY=FCT(EXP(X(IMAX)))
      ELSE
        YY=FCT(X(IMAX))
      ENDIF
      IF(LOGY.EQ.1) THEN
        IF(YY.LT.0.0D0) THEN
          WRITE(6,*) '     X =',X(IMAX)
          WRITE(6,*) 'FCT(X) =',YY
          STOP 'TABLEF: FCT must be non-negative.'
        ENDIF
        Y(IMAX)=LOG(MAX(YY,ZERO))
      ELSE
        Y(IMAX)=YY
      ENDIF
      IF(N.LT.MIN(NP,NPM)) GO TO 6
C
 8    CONTINUE
      NP=N
      IF(LOGX.EQ.1) THEN
        DO I=1,NP
          X(I)=EXP(X(I))
        ENDDO
      ENDIF
      IF(LOGY.EQ.1) THEN
        ERR=EXP(ERR)-1.0D0
        DO I=1,NP
          Y(I)=EXP(Y(I))
        ENDDO
      ENDIF
      RETURN
      END
C  *********************************************************************
C                       FUNCTION FINTRP
C  *********************************************************************
      FUNCTION FINTRP(XC,X,Y,N,MODE)
C
C     Interpolation (and extrapolation) in a table X(1:N),Y(1:N) with N
C  data points. The abscissas X(I) must be in non-decreasing order; a
C  duplicated abscissa is considered as a discontinuity.
C
C  Input arguments:
C     XC ......... value of the variable.
C     X(1:N) ..... grid points (in non-decreasing order!).
C     Y(1:N) ..... corresponding function values.
C     N .......... number of grid points.
C     MODE ....... interpolation mode:
C                    <1 or >7 => linear [Y is linear in X],
C                    1 => lin-log linear [LOG(Y) is linear in X],
C                    2 => log-lin linear [Y is linear in LOG(X)],
C                    3 => log-log linear [LOG(Y) is linear in LOG(X)],
C                    4 => 4-point Lagrange,
C                    5 => lin-log 4-point Lagrange,
C                    6 => log-lin 4-point Lagrange,
C                    7 => log-log 4-point Lagrange.
C
C  Output:
C     FINTRP .... value of the function at XC.
C
C  The logarithm of a zero or negative abscissa (X<1.0D-99) will abort
C  the program. In log interpolations of the function, negative function
C  values are replaced with 1.0D-99
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER (EPS=1.0D-10)
      DIMENSION X(N),Y(N)
C  **** Protection against incorrect array bounds
      IF((MODE.GT.3).AND.(MODE.LT.8)) THEN
        IF(N.LT.4) THEN
          WRITE(6,*) 'MODE =',MODE,',  N =',N
          STOP 'FINTRP: Not enough data points'
        ENDIF
      ELSE
        IF(N.LT.2) THEN
          WRITE(6,*) 'MODE =',MODE,',  N =',N
          STOP 'FINTRP: Not enough data points'
        ENDIF
      ENDIF
C
C  ****  Bisection to locate the covering interval.
C
      IF(XC.GT.X(N)) THEN
        I=N-1
      ELSE IF((XC.LT.X(1)).OR.(N.EQ.2)) THEN
        I=1
      ELSE
        I=1
        I1=N
 1      IT=(I+I1)/2
        IF(XC.GT.X(IT)) THEN
          I=IT
        ELSE
          I1=IT
        ENDIF
        IF(I1-I.GT.1) GO TO 1
      ENDIF
C  ****  Protection.
      IF(X(I+1)-X(I).LT.EPS*(1.0D0+MAX(ABS(X(I)),ABS(X(I+1))))) THEN
        FINTRP=0.5D0*(Y(I+1)+Y(I))
        RETURN
      ENDIF
C
      IF((MODE.EQ.2).OR.(MODE.EQ.3).OR.(MODE.EQ.6).OR.(MODE.EQ.7)) THEN
        IF(X(I).LT.1.0D-99) THEN
          WRITE(6,*) 'X(I) =',X(I)
          STOP 'FINTRP: A log scale extending to zero?'
        ENDIF
        LOGX=1
        XXC=LOG(XC)
      ELSE
        LOGX=0
        XXC=XC
      ENDIF
      IF((MODE.EQ.1).OR.(MODE.EQ.3).OR.(MODE.EQ.5).OR.(MODE.EQ.7)) THEN
        LOGY=1
      ELSE
        LOGY=0
      ENDIF
      IF((MODE.LT.4).OR.(MODE.GT.7)) GO TO 3
C
C  ****  Determining the four-point bracketing interval, if possible.
C
      IL=I
      IU=I+1
 2    CONTINUE
      NT=IU-IL+1
      IF(IL.GT.1) THEN
        IF(X(IL)-X(IL-1).GT.EPS*(1.0D0+MAX(ABS(X(IL)),ABS(X(IL-1)))))
     1    THEN
          IL=IL-1
        ENDIF
      ENDIF
      IF(IU.LT.N) THEN
        IF(X(IU+1)-X(IU).GT.EPS*(1.0D0+MAX(ABS(X(IU)),ABS(X(IU+1)))))
     1    THEN
          IU=IU+1
        ENDIF
      ENDIF
      NTN=IU-IL+1
      IF((NTN.LT.4).AND.(NTN.NE.NT)) GO TO 2
C
C  ****  Four-point Lagrange interpolation.
C
      IF(NTN.EQ.4) THEN
        IF(LOGX.EQ.1) THEN
          IF(X(IL).LT.1.0D-99) THEN
            WRITE(6,*) 'X(IL) =',X(IL)
            STOP 'FINTRP: A log scale extending to zero?'
          ENDIF
          X0=LOG(X(IL))
          X1=LOG(X(IL+1))
          X2=LOG(X(IL+2))
          X3=LOG(X(IL+3))
        ELSE
          X0=X(IL)
          X1=X(IL+1)
          X2=X(IL+2)
          X3=X(IL+3)
        ENDIF
        IF(LOGY.EQ.1) THEN
          Y0=LOG(MAX(Y(IL),1.0D-99))
          Y1=LOG(MAX(Y(IL+1),1.0D-99))
          Y2=LOG(MAX(Y(IL+2),1.0D-99))
          Y3=LOG(MAX(Y(IL+3),1.0D-99))
        ELSE
          Y0=Y(IL)
          Y1=Y(IL+1)
          Y2=Y(IL+2)
          Y3=Y(IL+3)
        ENDIF
C  ----  Direct formula (4 divisions, 20 products, 24 additions).
*       FINTRP=((XC-X1)*(XC-X2)*(XC-X3)*F0)/((X0-X1)*(X0-X2)*(X0-X3))
*    1        +((XC-X0)*(XC-X2)*(XC-X3)*F1)/((X1-X0)*(X1-X2)*(X1-X3))
*    2        +((XC-X0)*(XC-X1)*(XC-X3)*F2)/((X2-X0)*(X2-X1)*(X2-X3))
*    3        +((XC-X0)*(XC-X1)*(XC-X2)*F3)/((X3-X0)*(X3-X1)*(X3-X2))
C  ----  Coefficients of the interpolating polynomial (6 divisions,
C        11 products, 26 additions). Allows further calculations.
        YP1=(Y1-Y0)/(X1-X0)
        YP2=(Y2-Y1)/(X2-X1)
        A2=(((Y3-Y2)/(X3-X2))-YP2)/(X3-X1)
        A3=(A2-((YP2-YP1)/(X2-X0)))/(X3-X0)
        A2=A2-A3*(X3+X2+X1)
        A1=YP1-A2*(X0+X1)-A3*(X1*(X1+X0)+X0*X0)
        A0=Y0-X0*(A1+X0*(A2+X0*A3))
        FINTRP=A0+XXC*(A1+XXC*(A2+XXC*A3))
        IF(LOGY.EQ.1) FINTRP=EXP(FINTRP)
        RETURN
      ENDIF
C
C  ****  Linear interpolation.
C
 3    CONTINUE
      IF(LOGX.EQ.1) THEN
        IF(X(I).LT.1.0D-99) THEN
          WRITE(6,*) 'X(I) =',X(I)
          STOP 'FINTRP: A log scale extending to zero?'
        ENDIF
        X0=LOG(X(I))
        X1=LOG(X(I+1))
      ELSE
        X0=X(I)
        X1=X(I+1)
      ENDIF
      IF(LOGY.EQ.1) THEN
        Y0=LOG(MAX(Y(I),1.0D-99))
        Y1=LOG(MAX(Y(I+1),1.0D-99))
      ELSE
        Y0=Y(I)
        Y1=Y(I+1)
      ENDIF
      FINTRP=Y0+(XXC-X0)*(Y1-Y0)/(X1-X0)
      IF(LOGY.EQ.1) FINTRP=EXP(FINTRP)
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE TCLEAN
C  *********************************************************************
      SUBROUTINE TCLEAN(X,Y,ERCUT,ERRM,NP,NPS,MODE)
C
C     Cleans a table of function values X(I),Y(I). The error at each
C  grid point is estimated by temporarily removing the point and
C  considering the interpolated value from the rest of the table. Points
C  with errors less than the tolerance ERCUT are permanently removed.
C
C  Input arguments:
C    X(1:NP) ... grid points (in increasing order!).
C    Y(1:NP) ... corresponding function values.
C    ERCUT ..... desired accuracy [tolerable error; relative error if
C                log(Y) is interpolated, absolute error otherwise].
C    NP ........ dimension of arrays X and Y.
C    NPS ....... desired size of the cleaned table, points with the
C                smaller relative (log) or absolute (lin) errors are
C                removed from the table.
C    MODE ...... interpolation mode:
C                    <1 or >7 => linear [Y is linear in X],
C                    1 => lin-log linear [LOG(Y) is linear in X],
C                    2 => log-lin linear [Y is linear in LOG(X)],
C                    3 => log-log linear [LOG(Y) is linear in LOG(X)],
C                    4 => 4-point Lagrange,
C                    5 => lin-log 4-point Lagrange,
C                    6 => log-lin 4-point Lagrange,
C                    7 => log-log 4-point Lagrange.
C
C  Output arguments:
C    X(1:NPS),Y(1:NPS) ..... abscisas and function values.
C    NPS ....... number of points in the cleaned table.
C    ERRM ...... largest interpolation error of the cleaned table;
C                relative error if log(Y) is interpolated, absolute
C                error otherwise.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER (EPS=1.0D-10)
      PARAMETER (NPM=10000)
      DIMENSION X(NP),Y(NP)
      DIMENSION XM(NPM),YM(NPM),ERR(NPM)
C
      IF(NP.LE.NPS) THEN
        NPS=NP
        RETURN
      ENDIF
C
      IF(NP.LT.5) THEN
        WRITE(6,*) '  NP =',NP
        STOP 'TCLEAN: Too few points to clean a table.'
      ENDIF
      IF(NP.GT.NPM) THEN
        WRITE(6,*) '  NP =',NP
        WRITE(6,*) ' NPM =',NPM
        STOP 'TCLEAN: The parameter NPM must be larger than NP.'
      ENDIF
      ERC=ERCUT
C
      N=NP
      IF((MODE.EQ.1).OR.(MODE.EQ.3).OR.(MODE.EQ.5).OR.(MODE.EQ.7)) THEN
        LREL=1  ! Relative error.
      ELSE
        LREL=0  ! Absolute error.
      ENDIF
      IF(MODE.GT.3.AND.MODE.LT.8) THEN
        IADD=2
      ELSE
        IADD=1
      ENDIF
C
C  ****  Evaluation of errors at the grid points.
C
 1    CONTINUE
      N1=N-1
      ERRM=0.0D0
      DO I=1,N1
        XM(I)=X(I+1)
        YM(I)=Y(I+1)
      ENDDO
      LDISC=-2
      DO I=1,N1
        IF(I.EQ.1) THEN
          ERR(I)=1.0D35
        ELSE
C  ****  Keep the discontinuities (duplicated abscissas).
          IF(X(I+1)-X(I).LT.EPS*ABS(X(I))) THEN
            LDISC=I
            ERR(I)=1.0D35
            ERR(I-1)=1.0D35
            IF(IADD.EQ.2) THEN
              IF(I-2.GT.0) ERR(I-2)=1.0D35
              IF(I-3.GT.0) ERR(I-3)=1.0D35
            ENDIF
          ELSE
            YMI=FINTRP(X(I),XM,YM,N1,MODE)
            IF(LREL.EQ.1) THEN
              ERR(I)=ABS(YMI/Y(I)-1.0D0)
            ELSE
              ERR(I)=ABS(YMI-Y(I))
            ENDIF
          ENDIF
        ENDIF
        IF(IADD.EQ.1.AND.I.LT.LDISC+3) ERR(I)=1.0D35
        IF(IADD.EQ.2.AND.I.LT.LDISC+5) ERR(I)=1.0D35
        XM(I)=X(I)
        YM(I)=Y(I)
        IF(ERR(I).LT.1.0D34) ERRM=MAX(ERR(I),ERRM)
      ENDDO
      XM(N)=X(N)
      YM(N)=Y(N)
C
C  ****  Remove points with interpolation error less than ERC.
C
      M=1
      X(1)=XM(1)
      Y(1)=YM(1)
      ILAST=0
      DO I=2,N1
        IF(I.LT.N-2.AND.I.GT.ILAST+IADD) THEN
          IF(ERR(I).GT.ERC) THEN
            M=M+1
            X(M)=XM(I)
            Y(M)=YM(I)
          ELSE
            ILAST=I
          ENDIF
        ELSE
          M=M+1
          X(M)=XM(I)
          Y(M)=YM(I)
        ENDIF
      ENDDO
      M=M+1
      X(M)=XM(N)
      Y(M)=YM(N)
      IF(M.LT.N) THEN
        N=M
        IF(N.GT.NPS) GO TO 1
        IF(4*N.GT.NPS.AND.ERRM.LT.ERC) GO TO 1
      ELSE
C  ****  Increase ERC step by step to keep reducing the size of the
C        table. In this case, the desired accuracy is not reached.
        ERC=1.05D0*ERC
        N=M
        GO TO 1
      ENDIF
C  ****  Returns when the table has been cleaned.
      NPS=N
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE MERGE2
C  *********************************************************************
      SUBROUTINE MERGE2(X1,Y1,X2,Y2,X,Y,N1,N2,NPM,N,MODE)
C
C     This subroutine merges the tables (X1,Y1), (X2,Y2) of two func-
C  tions to produce a table (X,Y) of the sum of these functions, with
C  abscissas in increasing order. N1, N2 and N are the numbers of grid
C  points in the input and merged tables. A discontinuity of a function
C  is described by giving twice the abscissa; the merged table maintains
C  the discontinuities of the original tables.
C
C    NPM ...... physical dimension of arrays X and Y.
C               It is assumed that NPM > MAX(N1,N2,N1+N2).
C    MODE ..... interpolation mode:
C                    <1 or >7 => linear [Y is linear in X],
C                    1 => lin-log linear [LOG(Y) is linear in X],
C                    2 => log-lin linear [Y is linear in LOG(X)],
C                    3 => log-log linear [LOG(Y) is linear in LOG(X)],
C                    4 => 4-point Lagrange,
C                    5 => lin-log 4-point Lagrange,
C                    6 => log-lin 4-point Lagrange,
C                    7 => log-log 4-point Lagrange.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER (EPS=1.0D-4)
      DIMENSION X1(N1),Y1(N1),X2(N2),Y2(N2),X(NPM),Y(NPM)
C
      CALL SORT2(X1,Y1,N1)
      CALL SORT2(X2,Y2,N2)
C
      DO I1=1,N1
        X(I1)=X1(I1)
      ENDDO
      N=N1
      DO I2=1,N2
        XC=X2(I2)
        IF(XC.GT.X1(N1)) THEN
          I1=N1-1
        ELSE IF(XC.LT.X1(1)) THEN
          I1=1
        ELSE
          I1=1
          IS=N1
 1        IT=(I1+IS)/2
          IF(XC.GT.X1(IT)) THEN
            I1=IT
          ELSE
            IS=IT
          ENDIF
          IF(IS-I1.GT.1) GO TO 1
        ENDIF
C
        TST1=ABS(XC-X1(I1))
        TST2=ABS(XC-X1(I1+1))
        TST12=MIN(TST1,TST2)
        IF(I2.GT.1) THEN
          TST3=ABS(XC-X2(I2-1))
        ELSE
          TST3=1.0D6
        ENDIF
        IF(I2.LT.N2) THEN
          TST4=ABS(XC-X2(I2+1))
        ELSE
          TST4=1.0D6
        ENDIF
        TST34=MIN(TST3,TST4)
        TST=EPS*XC
        IF(TST34.LT.TST) THEN
          N=N+1
          IF(N.GT.NPM) THEN
            WRITE(6,*) 'NPM =',NPM
            STOP 'MERGE2: Increase the dimension NPM.'
          ENDIF
          X(N)=XC
        ELSE IF(TST12.GT.TST) THEN
          N=N+1
          IF(N.GT.NPM) THEN
            WRITE(6,*) 'NPM =',NPM
            STOP 'MERGE2: Increase the dimension NPM.'
          ENDIF
          X(N)=XC
        ENDIF
      ENDDO
C
C  ****  Sort and clean the merged grid.
C
 2    CONTINUE
      DO I=1,N-1
        IMIN=I
        XMIN=X(I)
        DO J=I+1,N
          IF(X(J).LT.XMIN) THEN
            IMIN=J
            XMIN=X(J)
          ENDIF
        ENDDO
        SAVE=X(I)
        X(I)=X(IMIN)
        X(IMIN)=SAVE
      ENDDO
C
C  ****  Remove redundant triple points.
C
      DO I=1,N-2
        IF(X(I).GT.X(I+2)-EPS*ABS(X(I+2))) THEN
          X(I+1)=X(N)
          N=N-1
          GO TO 2
        ENDIF
      ENDDO
C  ****  Remove duplicated first and last points.
      IF(X(1).GT.X(2)-EPS*ABS(X(2))) THEN
        DO I=2,N-1
          X(I)=X(I+1)
        ENDDO
        N=N-1
      ENDIF
      IF(X(N-1).GT.X(N)-EPS*ABS(X(N))) THEN
        N=N-1
      ENDIF
C  ****  Ensure duplicated abscissas are equal.
      DO I=2,N-2
        IF(X(I).GT.X(I+1)-EPS*ABS(X(I+1))) X(I+1)=X(I)
      ENDDO
C
      DO I=1,N
        IF(I.GT.1.AND.I.LT.N) THEN
          IF(ABS(X(I)-X(I-1)).LT.EPS*ABS(X(I))) THEN
            XC=X(I)+EPS*ABS(X(I))
          ELSE IF(ABS(X(I+1)-X(I)).LT.EPS*ABS(X(I))) THEN
            XC=X(I)-EPS*ABS(X(I))
          ELSE
            XC=X(I)
          ENDIF
        ELSE
          XC=X(I)
        ENDIF
        YI1=FINTRP(XC,X1,Y1,N1,MODE)
        YI2=FINTRP(XC,X2,Y2,N2,MODE)
        Y(I)=YI1+YI2
      ENDDO
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE SORT2
C  *********************************************************************
      SUBROUTINE SORT2(X,Y,N)
C
C  This subroutine sorts a table (X,Y) of a function with N data points.
C  A discontinuity of the function is described by giving twice the abs-
C  cissa, the ordering of function values at discontinuities is not
C  altered.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER (EPS=1.0D-10, NDIM=5000)
      DIMENSION X(N),Y(N),IORDER(NDIM)
C
      IF(N.GT.NDIM) THEN
        WRITE(6,*) 'NDIM =',N
        WRITE(6,*) 'N =',N
        STOP 'SORT2: Increase the value of the parameter NDIM.'
      ENDIF
C
      IF(N.EQ.1) RETURN
      DO I=1,N
        IORDER(I)=I
      ENDDO
C
      DO 1 I=1,N-1
        IMIN=I
        XMIN=X(I)
        DO J=I+1,N
          IF(X(J).LT.XMIN) THEN
            IMIN=J
            XMIN=X(J)
          ENDIF
        ENDDO
        SAVE=X(I)
        X(I)=X(IMIN)
        X(IMIN)=SAVE
        SAVE=Y(I)
        Y(I)=Y(IMIN)
        Y(IMIN)=SAVE
        ISAVE=IORDER(I)
        IORDER(I)=IORDER(IMIN)
        IORDER(IMIN)=ISAVE
        IF(I.EQ.1) GO TO 1
        IF(IORDER(I).LT.IORDER(I-1).AND.ABS(X(I)-X(I-1)).LT.EPS)
     1    THEN
          SAVE=X(I-1)
          X(I-1)=X(I)
          X(I)=SAVE
          SAVE=Y(I-1)
          Y(I-1)=Y(I)
          Y(I)=SAVE
          ISAVE=IORDER(I-1)
          IORDER(I-1)=IORDER(I)
          IORDER(I)=ISAVE
        ENDIF
 1    CONTINUE
      RETURN
      END
C  *********************************************************************
C                       FUNCTION SMOMLL
C  *********************************************************************
      FUNCTION SMOMLL(X,Y,XL,XU,NP,MOM,ILOG)
C
C     Calculates integrals of a tabulated function, Y(X), over the
C  interval (XL,XU) by using linear log-log interpolation of the input
C  table. The values of both the variable X and the function Y are
C  assumed to be non-negative.
C
C  Input arguments:
C     X(1:NP) ..... array of variable values (in increasing order).
C     Y(1:NP) ..... corresponding function values.
C     NP .......... number of points in the table.
C     XL, XU ...... limits of the integration interval.
C     MOM ......... moment order.
C     ILOG ........ optional logarithm:
C                   ILOG=1,  SMOMLL = INTEGRAL X**MOM*LOG(X)*Y(X) dX
C                   else     SMOMLL = INTEGRAL X**MOM*Y(X) dX.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER (EPS=1.0D-12, ONEM=1.0D0-EPS, ZERO=1.0D-90)
      DIMENSION X(NP),Y(NP)
C
      IF(NP.LT.2) STOP 'SMOMLL: NP is too small.'
      IF(X(1).LT.0.0D0.OR.Y(1).LT.0.0D0) THEN
        I=1
        WRITE(6,'(A,I5,1P,2E15.7)') 'I,X(I),Y(I) =',I,X(I),Y(I)
        STOP 'SMOMLL: Negative values in the table.'
      ENDIF
      DO I=2,NP
        IF(X(I).LT.0.0D0.OR.Y(I).LT.0.0D0) THEN
          WRITE(6,'(A,I5,1P,2E15.7)') 'I,X(I),Y(I) =',I,X(I),Y(I)
          STOP 'SMOMLL: Negative values in the table.'
        ENDIF
        IF(X(I).LT.X(I-1)*ONEM) THEN
          J=I-1
          WRITE(6,'(A,I5,1P,2E15.7)') 'I,X(I),Y(I) =',J,X(J),Y(J)
          WRITE(6,'(A,I5,1P,2E15.7)') 'I,X(I),Y(I) =',I,X(I),Y(I)
          STOP 'SMOMLL: X values are in decreasing order.'
        ENDIF
      ENDDO
C
      XLOW=XL
      IF(XLOW.LT.ZERO) XLOW=ZERO
      XUP=XU
C
      IF(XLOW.GT.XUP) THEN
        WRITE(6,*) 'SMOMLL (warning): XLOW is greater than XUP.'
        WRITE(6,'(A,1P,E15.7,A,E15.7)') ' XLOW =',XLOW,', XUP =',XUP
        SMOMLL=0.0D0
        RETURN
      ENDIF
C
      IF(XLOW.GT.X(NP)) THEN
        I=NP-1
      ELSE IF(XLOW.LT.X(1)) THEN
        I=1
      ELSE
        I=1
        I1=NP
 1      IT=(I+I1)/2
        IF(XLOW.GT.X(IT)) THEN
          I=IT
        ELSE
          I1=IT
        ENDIF
        IF(I1-I.GT.1) GO TO 1
      ENDIF
      IL=I
C
      IF(XUP.GT.X(NP)) THEN
        I=NP-1
      ELSE IF(XUP.LT.X(1)) THEN
        I=1
      ELSE
        I=1
        I1=NP
 2      IT=(I+I1)/2
        IF(XUP.GT.X(IT)) THEN
          I=IT
        ELSE
          I1=IT
        ENDIF
        IF(I1-I.GT.1) GO TO 2
      ENDIF
      IU=I
C
      SMOMLL=0.0D0
      IF(ILOG.EQ.1) GO TO 3
C
C  ****  SMOMLL = INTEGRAL (X**N)*Y(X) dX, MOM.GT.-100.
C
      DO I=IL,IU
        XA=MAX(XLOW,X(I))
        XB=MIN(XUP,X(I+1))
        IF(I.EQ.1) THEN
          XA=XLOW
        ELSE IF(I.EQ.NP-1) THEN
          XB=XUP
        ENDIF
        X1L=LOG(MAX(X(I),ZERO))
        X2L=LOG(MAX(X(I+1),ZERO))
        Y1L=LOG(MAX(Y(I),ZERO))
        Y2L=LOG(MAX(Y(I+1),ZERO))
        DEN=X2L-X1L
        IF(ABS(DEN).GT.EPS) THEN  ! Interpolated values.
          YA=EXP(Y1L+(Y2L-Y1L)*(LOG(XA)-X1L)/DEN)*XA**MOM
          YB=EXP(Y1L+(Y2L-Y1L)*(LOG(XB)-X1L)/DEN)*XB**MOM
        ELSE
          YAV=EXP(0.5D0*(Y1L+Y2L))
          YA=YAV*XA**MOM
          YB=YAV*XB**MOM
        ENDIF
C
        DXL=LOG(XB)-LOG(XA)
        DYL=LOG(YB)-LOG(YA)
        IF(ABS(DXL).GT.EPS*ABS(DYL)) THEN
          AP1=1.0D0+(DYL/DXL)
          IF(ABS(AP1).GT.EPS) THEN
            DSUM=(YB*XB-YA*XA)/AP1
          ELSE
            DSUM=YA*XA*DXL
          ENDIF
        ELSE
          DSUM=0.5D0*(YA+YB)*(XB-XA)
        ENDIF
        SMOMLL=SMOMLL+DSUM
      ENDDO
      RETURN
C
C  ****  SMOMLL = INTEGRAL LOG(X)*Y(X) dX, MOM.LT.-100.
C
 3    CONTINUE
      DO I=IL,IU
        XA=MAX(XLOW,X(I))
        XB=MIN(XUP,X(I+1))
        IF(I.EQ.1) THEN
          XA=XLOW
        ELSE IF(I.EQ.NP-1) THEN
          XB=XUP
        ENDIF
        X1L=LOG(MAX(X(I),ZERO))
        X2L=LOG(X(I+1))
        Y1L=LOG(MAX(Y(I),ZERO))
        Y2L=LOG(MAX(Y(I+1),ZERO))
        DEN=X2L-X1L
        IF(ABS(DEN).GT.ZERO) THEN
          YA=EXP(Y1L+(Y2L-Y1L)*(LOG(XA)-X1L)/DEN)*XA**MOM
          YB=EXP(Y1L+(Y2L-Y1L)*(LOG(XB)-X1L)/DEN)*XB**MOM
        ELSE
          YAV=EXP(0.5D0*(Y1L+Y2L))
          YA=YAV*XA**MOM
          YB=YAV*XB**MOM
        ENDIF
        DXL=LOG(XB)-LOG(XA)
        DYL=LOG(YB)-LOG(YA)
        IF(ABS(DXL).GT.EPS*ABS(DYL)) THEN
          AP1=1.0D0+(DYL/DXL)
          IF(ABS(AP1).GT.EPS) THEN
            APREC=1.0D0/AP1
            DSUM=(YB*XB*(LOG(XB)-APREC)-YA*XA*(LOG(XA)-APREC))*APREC
          ELSE
            DSUM=YA*XA*0.5D0*(LOG(XB)**2-LOG(XA)**2)
          ENDIF
        ELSE
          DSUM=0.5D0*(YA*LOG(XA)+YB*LOG(XB))*(XB-XA)
        ENDIF
        SMOMLL=SMOMLL+DSUM
      ENDDO
      RETURN
      END


C  *********************************************************************
C                       SUBROUTINE SPLINE
C  *********************************************************************
      SUBROUTINE SPLINE(X,Y,A,B,C,D,S1,SN,N)
C
C     This subroutine determines the coefficients of a piecewise cubic
C  spline that interpolates the input table (X,Y) of function values.
C  Duplicated abscissas are considered as discontinuities; a separate
C  spline is used for each interval between consecutive discontinuities,
C  with 'natural' end-point shape (null second derivative) at the
C  discontinuities.
C
C  Input:
C     X(1:N) ..... grid points (must be in non-decreasing order).
C     Y(1:N) ..... corresponding function values.
C     S1,SN ...... second derivatives at X(1) and X(N).
C     N .......... number of grid points.
C
C  Output:
C     A,B,C,D(1:N) ... spline coefficients.
C
C  Other subprograms used: subroutine SPLIN0.
C
C  The interpolating cubic polynomial in the I-th interval, from X(I) to
C  X(I+1), is
C               P(x) = A(I)+x*(B(I)+x*(C(I)+x*D(I)))
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER (EPS=1.0D-10)
      DIMENSION X(N),Y(N),A(N),B(N),C(N),D(N)
C
      PARAMETER (NM=25000)  ! Auxiliary storage.
      DIMENSION XP(NM),YP(NM),AP(NM),BP(NM),CP(NM),DP(NM)
C
      SS1=S1
      SSN=SN
C
      IO=0
      I=0
      NP=0
 1    I=I+1
      NP=NP+1
      XP(NP)=X(I)
      YP(NP)=Y(I)
      IF(I.EQ.N) GO TO 2
C
      IF(ABS(X(I+1)-X(I)).GT.EPS*MAX(ABS(X(I)),ABS(X(I+1)))) THEN
        GO TO 1
      ELSE
        X(I)=X(I+1)
      ENDIF
 2    CONTINUE
C
      IF(NP.LT.2) THEN
        WRITE(6,10)
 10     FORMAT(1X,'*** Error in SPLINE: More than 2 coinciding ',
     1    'abscissas.',/5X,'Interpolation is not possible.')
        OPEN(33,FILE='SPLINE-error.dat')
          WRITE(33,*) '# Error at I =',I
          DO J=1,N
            WRITE(33,'(I5,1P,2E18.10)') J,X(J),Y(J)
          ENDDO
        CLOSE(33)
        STOP 'SPLINE: More than 2 coinciding abscissas.'
      ELSE IF(NP.EQ.2) THEN  ! Linear interpolation.
        AP(1)=(XP(2)*YP(1)-XP(1)*YP(2))/(XP(2)-XP(1))
        BP(1)=(YP(2)-YP(1))/(XP(2)-XP(1))
        CP(1)=0.0D0
        DP(1)=0.0D0
        AP(2)=AP(1)
        BP(2)=BP(1)
        CP(2)=CP(1)
        DP(2)=DP(1)
      ELSE IF(NP.EQ.3) THEN  ! Quadratic interpolation.
        BB=((XP(1)-XP(2))**2*(YP(3)-YP(2))
     1    -(XP(3)-XP(2))**2*(YP(1)-YP(2)))
     2    /((XP(3)-XP(1))*(XP(3)-XP(2))*(XP(2)-XP(1)))
        CC=((XP(3)-XP(2))*(YP(1)-YP(2))
     1    -(XP(1)-XP(2))*(YP(3)-YP(2)))
     2    /((XP(3)-XP(1))*(XP(3)-XP(2))*(XP(2)-XP(1)))
        AP(1)=YP(2)-BB*XP(2)+CC*XP(2)**2
        BP(1)=BB-2.0D0*CC*XP(2)
        CP(1)=CC
        DP(1)=0.0D0
        AP(2)=AP(1)
        BP(2)=BP(1)
        CP(2)=CP(1)
        DP(2)=DP(1)
        AP(3)=AP(1)
        BP(3)=BP(1)
        CP(3)=CP(1)
        DP(3)=DP(1)
      ELSE
        IF(NP.GT.NM) THEN
          WRITE(6,11)
 11       FORMAT(1X,'*** Error in SPLINE: too many grid points ',
     1      'between two',/5X,'discontinuities of Y(X). ',
     2      /5X,'Details in file ''SPLIN0-error.dat''.')
          WRITE(6,*) '     NP =',NP
          STOP 'SPLINE: NP is larger than NPM.'
        ENDIF
        IF(IO.EQ.1) THEN
          SP1=SS1
        ELSE
          SP1=0.0D0
        ENDIF
        IF(I.EQ.N) THEN
          SPN=SSN
        ELSE
          SPN=0.0D0
        ENDIF
        CALL SPLIN0(XP,YP,AP,BP,CP,DP,SP1,SPN,NP)
      ENDIF
C
      DO J=1,NP
        IO=IO+1
        A(IO)=AP(J)
        B(IO)=BP(J)
        C(IO)=CP(J)
        D(IO)=DP(J)
      ENDDO
      IF(I.LT.N) THEN
        NP=0
        GO TO 1
      ENDIF
C
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE SPLIN0
C  *********************************************************************
      SUBROUTINE SPLIN0(X,Y,A,B,C,D,S1,SN,N)
C
C     Initialization of cubic spline interpolation of tabulated data.
C  It is assumed that the function and its first two derivatives are
C  continuous.
C
C  Input:
C     X(1:N) ..... grid points (the X values must be in strictly
C                      increasing order).
C     Y(1:N) ..... corresponding function values.
C     S1,SN ...... second derivatives at X(1) and X(N). The natural
C                  spline corresponds to taking S1=SN=0.
C     N .......... number of grid points.
C  Output:
C     A,B,C,D(1:N) ... spline coefficients.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER (EPS=1.0D-10)
      DIMENSION X(N),Y(N),A(N),B(N),C(N),D(N)
C
      IF(N.LT.4) THEN
        WRITE(6,10) N
 10     FORMAT(1X,'*** Error in SPLIN0: interpolation cannot be ',
     1    'performed with',I4,' points.',
     2    /5X,'Details in file ''SPLIN0-error.dat''.')
        OPEN(33,FILE='SPLIN0-error.dat')
          DO J=1,N
            WRITE(33,'(I5,1P,2E18.10)') J,X(J),Y(J)
          ENDDO
        CLOSE(33)
        STOP 'SPLIN0: N is less than 4.'
      ENDIF
      N1=N-1
      N2=N-2
C  ****  Auxiliary arrays H(=A) and DELTA(=D).
      DO I=1,N1
        A(I)=X(I+1)-X(I)  ! h_i
        IF(A(I).LT.EPS*MAX(ABS(X(I)),ABS(X(I+1)))) THEN
          WRITE(6,11)
 11       FORMAT(1X,'*** Error in SPLIN0: X values not in',
     1      'increasing order.',
     2      /5X,'Details in file ''SPLIN0-error.dat''.')
          OPEN(33,FILE='SPLIN0-error.dat')
            WRITE(33,'(A,I5)') 'Order error at I =',I
            DO J=1,N
              WRITE(33,'(I5,1P,2E18.10)') J,X(J),Y(J)
            ENDDO
          CLOSE(33)
          STOP 'SPLIN0: X values not in increasing order.'
        ENDIF
        D(I)=(Y(I+1)-Y(I))/A(I)  ! delta_i
      ENDDO
C  ****  Symmetric coefficient matrix (augmented).
      DO I=1,N2
        B(I)=2.0D0*(A(I)+A(I+1))  ! C_i
      ENDDO
      DO K=N1,2,-1
        D(K)=6.0D0*(D(K)-D(K-1))  ! D_i
      ENDDO
      D(2)=D(2)-A(1)*S1
      D(N1)=D(N1)-A(N1)*SN
C  ****  Gauss solution of the tridiagonal system.
      DO I=2,N2
        R=A(I)/B(I-1)
        B(I)=B(I)-R*A(I)
        D(I+1)=D(I+1)-R*D(I)
      ENDDO
C  ****  The SIGMA coefficients are stored in array D.
      D(N)=SN
      D(N1)=D(N1)/B(N2)
      DO K=N2,2,-1
        D(K)=(D(K)-A(K)*D(K+1))/B(K-1)
      ENDDO
C  ****  Spline coefficients.
      SI1=S1
      DO I=1,N1
        SI=SI1
        SI1=D(I+1)
        H=A(I)
        HI=1.0D0/H
        A(I)=(HI/6.0D0)*(SI*X(I+1)**3-SI1*X(I)**3)
     1      +HI*(Y(I)*X(I+1)-Y(I+1)*X(I))
     2      +(H/6.0D0)*(SI1*X(I)-SI*X(I+1))
        B(I)=(HI/2.0D0)*(SI1*X(I)**2-SI*X(I+1)**2)
     1      +HI*(Y(I+1)-Y(I))+(H/6.0D0)*(SI-SI1)
        C(I)=(HI/2.0D0)*(SI*X(I+1)-SI1*X(I))
        D(I)=(HI/6.0D0)*(SI1-SI)
      ENDDO
C  ****  Quadratic extrapolation for X.GT.X(N). Natural spline if SN=0.
      A(N)=A(N1)+D(N1)*X(N)**3
      B(N)=B(N1)-3.0D0*D(N1)*X(N)**2
      IF(ABS(SN).LT.1.0D-16) THEN
        C(N)=0.0D0
      ELSE
        C(N)=C(N1)+3.0D0*D(N1)*X(N)
      ENDIF
      D(N)=0.0D0
C
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE FINDI
C  *********************************************************************
      SUBROUTINE FINDI(XC,X,N,I)
C
C     This subroutine finds the interval (X(I),X(I+1)) that contains the
C  value XC by using the binary search algorithm.
C
C  Input:
C     XC ............. point to be located.
C     X(1:N) ......... grid points.
C     N  ............. number of grid points.
C  Output:
C     I .............. interval index.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      DIMENSION X(N)
C
      IF(XC.GT.X(N)) THEN
        I=N
      ELSE IF(XC.LT.X(1)) THEN
        I=1
      ELSE
        I=1
        I1=N
 1      IT=(I+I1)/2
        IF(XC.GT.X(IT)) THEN
          I=IT
        ELSE
          I1=IT
        ENDIF
        IF(I1-I.GT.1) GO TO 1
      ENDIF
      RETURN
      END
C  *********************************************************************
C                       FUNCTION SPLVAL
C  *********************************************************************
      FUNCTION SPLVAL(XC,X,A,B,C,D,N)
C
C     This function gives the value of a cubic spline at the point XC;
C  quadratic extrapolation is used for points outside the interval
C  (X(1),X(N)).
C
C  Input:
C     XC ............. spline argument.
C     N .............. number of grid points.
C     X(1:N) ......... grid points.
C     A,B,C,D(1:N) ... spline coefficients.
C
C  Output:
C     SPLVAL ......... value of the spline function at XC.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      DIMENSION X(N),A(N),B(N),C(N),D(N)
C
      IF(XC.LT.X(1)) THEN
C  ****  Quadratic extrapolation for X.LT.X(1). Natural spline if S1=0.
        A0=A(1)+D(1)*X(1)**3
        B0=B(1)-3.0D0*D(1)*X(1)**2
        C0=C(1)+3.0D0*D(1)*X(1)
        SPLVAL=A0+XC*(B0+XC*C0)
      ELSE IF(XC.GT.X(N)) THEN
        SPLVAL=A(N)+XC*(B(N)+XC*C(N))
      ELSE
        I=1
        I1=N
 1      IT=(I+I1)/2
        IF(XC.GT.X(IT)) THEN
          I=IT
        ELSE
          I1=IT
        ENDIF
        IF(I1-I.GT.1) GO TO 1
        SPLVAL=A(I)+XC*(B(I)+XC*(C(I)+XC*D(I)))
      ENDIF
      RETURN
      END
