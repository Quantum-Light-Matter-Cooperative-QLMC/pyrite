C  NOTE: The present subroutine package uses I/O units 33, 98 and 99.
C        Do not use these unit numbers in your main program.
C
C  *********************************************************************
C                       SUBROUTINE ELSEPA
C  *********************************************************************
      SUBROUTINE ELSEPA(IELEC,EV,IZ,NELEC,MNUCL,MELEC,MUFIN,RMUF,VMOL,
     1  MEXCH,MCPOL,VPOLA,VPOLB,MABS,VABSA,VABSD,IHEF,IW)
C
C
C                       F. Salvat, D. Bote, A. Jablonski and C.J. Powell
C                              September 25, 2008
C
C  Updated in September 2020 by F. Salvat, to run with the subroutine
C  package RADIAL.
C
C  Ref.: F. Salvat and J. M. Fernandez-Varea,
C        'RADIAL: a Fortran subroutine package for the solution of
C        radial Schrodinger and Dirac wave equations',
C        Comput. Phys. Commun. 240 (2019) 165-177.
C
C     This subroutine computes scattering amplitudes, differential cross
C  sections and total (integrated) cross sections for ELastic Scattering
C  of Electrons and Positrons by neutral Atoms and positive ions.
C
C     The interaction is described through a static (central) field,
C  which consists of the electrostatic potential and, for projectile
C  electrons, an approximate local exchange potential. For slow
C  projectiles, a correlation-polarization potential and an absorptive
C  imaginary potential can optionally be included. The differential
C  cross section is evaluated by means of relativistic (Dirac) partial-
C  wave analysis, or from approximate high-energy factorizations.
C
C  Input arguments:
C    IELEC ..... electron-positron flag;
C                =-1 for electrons,
C                =+1 for positrons.
C    EV ........ projectile's kinetic energy (in eV).
C    IZ ........ atomic number of the target atom or ion.
C    NELEC ..... number of bound atomic electrons.
C    MNUCL ..... nuclear charge density model.
C                  1 --> point nucleus (P),
C                  2 --> uniform distribution (U),
C                  3 --> Fermi distribution (F),
C                  4 --> Helm's uniform-uniform distribution (Uu).
C    MELEC ..... electron density model.
C                  1 --> TFM analytical density,
C                  2 --> TFD analytical density,
C                  3 --> DHFS analytical density,
C                  4 --> DF numerical density, read from 'Z_zzz.DEN',
C                  5 --> density read from file 'density.usr'.
C    MUFIN ..... Aggregation effects...
C                  0 --> free atom,
C                  1 --> muffin-tin model.
C      RMUF .... Muffin-tin radius (in cm).
C      VMOL .... Number of atoms per unit volume (in 1/cm**3).
C                    Reciprocal of the Wigner-Seitz cell volume.
C    MEXCH ..... exchange correction for electrons.
C                  0 --> no exchange correction,
C                  1 --> Furness-McCarthy (FM),
C                  2 --> Thomas-Fermi (TF),
C                  3 --> Riley-Truhlar (RT).
C    MCPOL ..... correlation-polarization correction.
C                  0 --> no correlation-polarization correction,
C                  1 --> Buckingham potential (B),
C                  2 --> Local density approximation (LDA).
C      VPOLA ... atomic polarizability (in cm**3).
C      VPOLB ... cutoff radius parameter b_pol
C                    (used only when MCPOL>0).
C    MABS ...... absorption correction (imaginary potential).
C                  0 --> no absorption correction,
C                  1 --> LDA-I (electron-hole excitations only).
C                  2 --> LDA-II (full Lindhard dielectric function).
C      VABSA ... strength of the absorption potential.
C      VABSD ... energy gap, DELTA (eV).
C                    (used only when MABS is different from 0).
C    IHEF ...... =0: phase shifts are computed for the electrostatic
C                    field of the whole atom (nucleus+electron cloud)
C                    with optional exchange, polarization and absorption
C                    corrections.
C                =1: the differential cross section is obtained from a
C                    high-energy factorization. The phase shifts are
C                    evaluated for the bare nucleus. The screening of
C                    the nuclear charge by the atomic electrons is
C                    accounted for by means of a pre-evaluated high-
C                    energy correction factor, which is read from file
C                    'Z_zzz.DFS'.
C                =2: when the energy is larger than 100 MeV, the DCS is
C                    obtained as the product of the Mott DCS for a point
C                    nucleus, the Helm Uu nuclear form factor (with an
C                    empirical Coulomb correction) and the electron
C                    screening factor.
C    IW ........ output unit (to be defined in the main program).
C
C  The electrostatic potential and electron density of the target atom
C  or ion are calculated by subroutine EFIELD and delivered through the
C  the named common block /CFIELD/.
C
C  Output (through the common block /DCSTAB/):
C     ECS ........ total cross section (cm**2).
C     TCS1 ....... 1st transport cross section (cm**2).
C     TCS2 ....... 2nd transport cross section (cm**2).
C     TH(I) ...... scattering angles (in deg)
C     XT(I) ...... values of (1-COS(TH(I)))/2.
C     DCST(I) .... differential cross section per unit solid
C                  angle at TH(I) (in cm**2/sr).
C     SPOL(I) .... Sherman spin-polarization function at TH(I).
C     ERROR(I) ... relative uncertainty of the computed DCS
C                  values. Estimated from the convergence of the
C                  series.
C     NTAB ....... number of angles in the table.
C
      USE CONSTANTS
C
      IMPLICIT DOUBLE PRECISION (A-B,D-H,O-Z), COMPLEX*16 (C),
     1   INTEGER*4 (I-N)
C
C  ****  The parameter IWR defines the amount of information printed on
C  output files:
C  IWR>0 => the scattering potential is printed on file 'scfield.dat'.
C  IWR>1 => the scattering amplitudes are printed on file 'scatamp.dat'.
      PARAMETER (IWR=2)
C
      PARAMETER (A0B2=A0B*A0B)
      PARAMETER (TREV=REV+REV)
      PARAMETER (PI=3.1415926535897932D0,FOURPI=4.0D0*PI)
C
      CHARACTER*120 SCFILE,FILE1,CS120,NULL
      CHARACTER*1 LIT10(10),LIT1,LIT2,LIT3
      DATA LIT10/'0','1','2','3','4','5','6','7','8','9'/
C
      PARAMETER (NGT=650)
      COMMON/DCSTAB/ECS,TCS1,TCS2,TH(NGT),XT(NGT),DCST(NGT),SPOL(NGT),
     1              ERROR(NGT),NTAB
      COMMON/CTOTCS/TOTCS,ABCS
      DIMENSION Q2T(NGT),FQ(NGT)
C  ****  Link with the RADIAL package.
      COMMON/RADWF/RRR(NDIM),P(NDIM),Q(NDIM),NRT,ILAST,IER
C
      COMMON/CFIELD/R(NDIM),RVN(NDIM),DEN(NDIM),RVST(NDIM),NPOT
      COMMON/FIELD/RAD(NDIM),RV(NDIM),NP
      COMMON/FIELDI/RADI(NDIM),RVI(NDIM),RW(NDIM),IAB,NPI
      DIMENSION RVEX(NDIM),RVPOL(NDIM)
C
      PARAMETER (NPC=1500,NDM=25000)
      DIMENSION SA(NDM),SB(NDM),SC(NDM),SD(NDM)
      COMMON/CRMORU/CFM(NPC),CGM(NPC),DPC(NPC),DMC(NPC),
     1              CFX,CGX,RUTHC,WATSC,RK2,ERRFC,ERRGC,NPC1
      DIMENSION DENA(NPC),DENB(NPC),DENC(NPC),DEND(NPC)
C
C  ****  Path to the ELSEPA database
      CHARACTER*100 PATHE
      PATHE='./database/'
C
C  ----  Mott DCS and spin polarization (point unscreened nucleus)
      IF(NELEC.EQ.0.AND.MNUCL.EQ.1) THEN
        CALL MOTTSC(IELEC,IZ,EV,IW)
        RETURN
      ENDIF
C  ----  High-energy Mott-Born approximation for neutral atoms.
      IF(EV.GT.100.0D6.AND.IHEF.EQ.2.AND.IZ.EQ.NELEC) THEN
        CALL HEBORN(IELEC,IZ,MNUCL,EV,IW)
        RETURN
      ENDIF
C
      WRITE(IW,1000)
 1000 FORMAT(1X,'#',/1X,'# Subroutine ELSEPA. Elastic scattering of ',
     1  'electrons and positrons',/1X,'#',20X,
     2  'by neutral atoms and positive ions')
      IF(IELEC.EQ.-1) THEN
        WRITE(IW,1100)
 1100   FORMAT(1X,'#',/1X,'# Projectile: electron')
      ELSE
        WRITE(IW,1200)
 1200   FORMAT(1X,'#',/1X,'# Projectile: positron')
      ENDIF
      E=EV/HREV
      WRITE(IW,1300) EV,E
 1300 FORMAT(1X,'# Kinetic energy =',1P,E12.5,' eV =',
     1       E12.5,' a.u.')
      IF(MUFIN.EQ.1.AND.NELEC.NE.IZ) THEN
        WRITE(IW,*) '   IZ =',IZ
        WRITE(IW,*) 'NELEC =',NELEC
        WRITE(IW,'(''  STOP. For muffin-tin atoms, NELEC '',
     1    ''must equal IZ.'')')
        WRITE(6,*) '   IZ =',IZ
        WRITE(6,*) 'NELEC =',NELEC
        STOP 'ELSEPA: For muffin-tin atoms, NELEC must equal IZ.'
      ENDIF
C
C  ****  You may wish to comment off the next condition to run the
C  program for kinetic energies less that 5 eV. However, the results
C  for these energies may be highly inaccurate.
C
      IF(EV.LT.4.9990D0) THEN
        WRITE(IW,'(''  STOP. The kinetic energy is too small.'')')
        STOP 'ELSEPA: The kinetic energy is too small.'
      ENDIF
      IF(EV.LT.100.0D0) THEN
        WRITE(IW,'(1X,''#'',/1X,''#  ***  WARNING: Energy is '',
     1   ''too low.'',/1X,''#'',16X,''The reliability of the '',
     2   ''results is questionable.'')')
      ENDIF
      ERE0=0.0D0
      EIM=0.0D0
      VMOL1=-1.0D-23  ! Serves only to prevent compiler warnings.
C
C  ****  Electrostatic field.
C
      IF(MNUCL.LT.1.OR.MNUCL.GT.4) THEN
        WRITE(IW,*) 'ELSEPA: incorrect MNUCL value.'
        STOP 'ELSEPA: incorrect MNUCL value.'
      ENDIF
      IF(MELEC.LT.1.OR.MELEC.GT.5) THEN
        WRITE(IW,*) 'ELSEPA: incorrect MELEC value.'
        STOP 'ELSEPA: incorrect MELEC value.'
      ENDIF
      CALL EFIELD(IZ,NELEC,MNUCL,MELEC,IW,0)
C
      IHEF0=0
      IAB=0
      IF(NELEC.EQ.0) THEN  ! Bare nuclei treated separately.
        IHEF0=2
        GO TO 100
      ENDIF
      IF(NELEC.EQ.IZ.AND.MABS.EQ.0.AND.EV.GT.20.1D3*IZ) THEN
        IF(IHEF.GT.0) THEN  ! Neutral atoms, high-energy factorization.
          IHEF0=1
          GO TO 100
        ENDIF
      ENDIF
C  ****  Caution: Partially stripped atoms are difficult to compute...
      ZINF=IELEC*DBLE(IZ-NELEC)
      DO I=1,NPOT
        RV(I)=DBLE(IELEC)*RVST(I)
      ENDDO
      RV(NPOT)=ZINF
C
C  ************  Muffin-tin model for scattering in solids.
C
      NMT=0
      IF(MUFIN.EQ.1) THEN
        WRITE(IW,1600) RMUF
 1600   FORMAT(1X,'#',/1X,'# Muffin-tin model:    Rmt =',1P,E12.5,' cm')
        IF(ABS(ZINF).GT.1.0D-6) THEN
          WRITE(IW,*) 'ELSEPA: muffin-tin model does not work for ions.'
          STOP 'ELSEPA: muffin-tin model does not work for ions.'
        ENDIF
        CALL SPLINE(R,RV,SA,SB,SC,SD,0.0D0,0.0D0,NPOT)
        CALL SPLINE(R,DEN,DENA,DENB,DENC,DEND,0.0D0,0.0D0,NPOT)
        RMT=RMUF/A0B
        IF(RMT.LT.R(NPOT)) THEN
          CALL FINDI(RMT,R,NPOT,J)
          IF(J.LT.5) STOP 'ELSEPA: The muffin-tin radius is too small.'
          DENRMT=DENA(J)+RMT*(DENB(J)+RMT*(DENC(J)+RMT*DEND(J)))
        ELSE
          RMT=R(NPOT)
          DENRMT=DEN(NPOT)
        ENDIF
C
        IF(VMOL.GT.0.0D0) THEN
          VMOL1=VMOL*A0B**3
          WRITE(IW,1601) VMOL
        ELSE
          VMOL1=1.0D0/(2.0D0*RMT)**3
          WRITE(IW,1601) VMOL1/A0B**3
        ENDIF
 1601   FORMAT(1X,'#   Atomic density:   Vmol =',1P,E12.5,' 1/cm**3')
        DO I=1,NPOT
          IF(R(I).GT.RMT) THEN
            IF(RAD(I-1).LT.RMT*0.9999999D0) THEN
              NP=I
              RAD(NP)=RMT
            ELSE
              NP=I-1
              RAD(NP)=RMT
            ENDIF
            RC1=RMT
            CALL FINDI(RC1,R,NPOT,J)
            V1=SA(J)+RC1*(SB(J)+RC1*(SC(J)+RC1*SD(J)))
            DEN1=DENA(J)+RC1*(DENB(J)+RC1*(DENC(J)+RC1*DEND(J)))
            RV(NP)=2.0D0*V1
            DEN(NP)=2.0D0*DEN1
            RVST(NP)=RV(NP)/DBLE(IELEC)
            GO TO 1
          ELSE
            RAD(I)=R(I)
C
            RC1=R(I)
            FD1=FOURPI*RC1**2
            V1=RV(I)
            DEN1=DEN(I)
C
            RC2=2.0D0*RMT-R(I)
            FD2=FOURPI*RC2**2
            CALL FINDI(RC2,R,NPOT,J)
            V2=SA(J)+RC2*(SB(J)+RC2*(SC(J)+RC2*SD(J)))
            DEN2=DENA(J)+RC2*(DENB(J)+RC2*(DENC(J)+RC2*DEND(J)))
C
            IF(I.GT.1) THEN
              RV(I)=V1+RC1*(V2/RC2)
              DEN(I)=DEN1+FD1*(DEN2/FD2)
            ELSE
              RV(I)=V1
              DEN(I)=DEN1
            ENDIF
          ENDIF
          RVST(I)=RV(I)/DBLE(IELEC)
        ENDDO
        NP=NPOT
 1      CONTINUE
C
C  ****  Ensure proper normalization of the muffin-tin electron density.
C
        SUM=SMOMLL(RAD,DEN,RAD(1),RAD(NP),NP,0,0)
        RHOU=(NELEC-SUM)/(FOURPI*RMT**3/3.0D0)
        DO I=1,NP
          DEN(I)=DEN(I)+RHOU*FOURPI*RAD(I)**2
        ENDDO
        SUM=SMOMLL(RAD,DEN,RAD(1),RAD(NP),NP,0,0)
        WRITE(6,*) 'Electron density normalization =',SUM
C
        NMT=NP
        ERE0=-RV(NMT)/RAD(NMT)
        E=E-RV(NMT)/RAD(NMT)
        DO I=2,NMT
          RV(I)=RV(I)-RV(NMT)*RAD(I)/RAD(NMT)
          RVST(I)=RVST(I)-RVST(NMT)*RAD(I)/RAD(NMT)
        ENDDO
        NPP=NP+10
        IF(NP.LT.NPP) THEN
          NP=NP+1
          DO I=NP,NPP
            IF(I.EQ.NP) THEN
              RAD(I)=RAD(I-1)
            ELSE
              RAD(I)=RAD(I-1)+0.01D0*RMT
            ENDIF
            RV(I)=0.0D0
            RVST(I)=0.0D0
            DEN(I)=0.0D0
          ENDDO
          NP=NPP
        ENDIF
      ELSE
        NP=NPOT
        DO I=1,NPOT
          RAD(I)=R(I)
        ENDDO
      ENDIF
C
C  ************  Exchange correction for electrons.
C
      IF(IELEC.EQ.-1.AND.MEXCH.NE.0) THEN
        IF(MEXCH.EQ.1) THEN
C  ****  Furness-McCarthy exchange potential.
          WRITE(IW,1500)
 1500     FORMAT(1X,'#',/1X,'# Furness-McCarthy exchange',
     1      ' potential')
          DO I=2,NP
            AUX=RAD(I)*E*(1.0D0+EV/TREV)+RVST(I)
            AUX2=AUX*AUX
            IF(DEN(I).GT.1.0D-5*AUX2) THEN
              RVEX(I)=0.5D0*(AUX-SQRT(AUX2+DEN(I)))
            ELSE
              T=DEN(I)/AUX2
              RVEX(I)=-0.5D0*AUX*T*(0.5D0-T*(0.125D0-T*0.065D0))
            ENDIF
            RV(I)=RV(I)+RVEX(I)
          ENDDO
        ELSE IF(MEXCH.EQ.2) THEN
C  ****  Thomas-Fermi exchange potential.
          WRITE(IW,1400)
 1400     FORMAT(1X,'#',/1X,'# Thomas-Fermi exchange potential')
          DO I=1,NP
            RHO=DEN(MAX(2,I))/(FOURPI*RAD(MAX(2,I))**2)
            SKF=(3.0D0*PI*PI*RHO)**3.333333333333333D-1
            EF=0.5D0*SKF*SKF
            SKL=SQRT(2.0D0*(E*(1.0D0+EV/TREV)+EF))
            X=SKF/SKL
            IF(X.LT.0.001D0) THEN
              FX=(2.0D0/3.0D0)*X**3
            ELSE
              FX=X-0.5D0*(1.0D0-X*X)*LOG(ABS((1.0D0+X)/(1.0D0-X)))
            ENDIF
            RVEX(I)=-(SKL/PI)*FX*RAD(I)
            RV(I)=RV(I)+RVEX(I)
          ENDDO
        ELSE IF(MEXCH.EQ.3) THEN
C  ****  Riley-Truhlar exchange potential.
          WRITE(IW,1501)
 1501     FORMAT(1X,'#',/1X,'# Riley-Truhlar exchange potential')
          DO I=1,NP
            AUX=4.0D0*(RAD(I)*E*(1.0D0+EV/TREV)+RVST(I))
            IF(AUX.GT.1.0D-16*DEN(I)) THEN
              RVEX(I)=-DEN(I)/AUX
              RV(I)=RV(I)+RVEX(I)
            ENDIF
          ENDDO
        ELSE
          WRITE(IW,*) 'ELSEPA: incorrect MEXCH value.'
          STOP 'ELSEPA: incorrect MEXCH value.'
        ENDIF
        IF(NMT.GT.1) THEN
          ERE0=ERE0-RV(NMT)/RAD(NMT)
          DO I=2,NMT
            RV(I)=RV(I)-RAD(I)*(RV(NMT)/RAD(NMT))
            RVEX(I)=RVEX(I)-RAD(I)*(RVEX(NMT)/RAD(NMT))
          ENDDO
        ENDIF
      ELSE
        IF(IELEC.EQ.-1) WRITE(IW,1511)
 1511   FORMAT(1X,'#',/1X,'# No exchange potential')
        DO I=1,NP
          RVEX(I)=0.0D0
        ENDDO
      ENDIF
C
C  ********  Absorption potential.
C
      IF(MABS.EQ.1.AND.VABSA.GT.1.0D-12.AND.EV.LT.1.001D6) THEN
C
C  ****  LDA-I model
C
        IAB=1
        WRITE(IW,1502) VABSD,VABSA
 1502   FORMAT(1X,'#',/1X,'# LDA-I absorption potential (only electr',
     1    'on-hole excitations):',/1X,'#',
     2    27X,'Delta =',1P,E12.5,' eV',/1X,'#',28X,'Aabs =',E12.5)
        DELTA=VABSD/HREV
        AABS=VABSA
C
        RW(1)=0.0D0
        RVPOL(1)=0.0D0
        DO I=2,NP
          RHO=DEN(I)/(FOURPI*RAD(I)**2)
C  ****  Local kinetic energy.
          EKIN=E-RV(I)/RAD(I)
          IF(RHO.GT.1.0D-16.AND.EKIN.GT.DELTA) THEN
            VEL=SQRT(2.0D0*EKIN)
C  ****  Relativistic correction.
            EKEV=EKIN*HREV
            FREL=SQRT(2.0D0*(EKEV+REV)**2/(REV*(EKEV+2.0D0*REV)))
C  ****  Only electron-hole excitations.
            CALL XSFEG(RHO,DELTA,IELEC,EKIN,0,XSEC,2)
            RW(I)=-0.5D0*VEL*RHO*XSEC*RAD(I)*AABS*FREL
          ELSE
            RW(I)=0.0D0
          ENDIF
          RVPOL(I)=0.0D0
          WRITE(6,1503) I,RAD(I),RW(I)
 1503     FORMAT(1X,'i=',I4,',   r=',1P,E12.5,',   r*Wabs= ',E12.5)
        ENDDO
      ELSE IF(MABS.EQ.2.AND.VABSA.GT.1.0D-12.AND.EV.LT.1.001D6) THEN
C
C  ****  LDA-II model.
C
        IAB=1
        WRITE(IW,1504) VABSD,VABSA
 1504   FORMAT(1X,'#',/1X,'# LDA-II absorption potential (plasmon and',
     1    ' e-h excitations):',/1X,'#',
     2    27X,'Delta =',1P,E12.5,' eV',/1X,'#',28X,'Aabs =',E12.5)
        DELTA=VABSD/HREV
        AABS=VABSA
C
        RW(1)=0.0D0
        RVPOL(1)=0.0D0
        DO I=2,NP
          RHO=DEN(I)/(FOURPI*RAD(I)**2)
          IF(RHO.GT.1.0D-16) THEN
            EFERMI=0.5D0*(3.0D0*PI**2*RHO)**0.666666666666666D0
C  ****  Local kinetic energy.
            IF(IELEC.EQ.-1) THEN
              EKIN=E+EFERMI
            ELSE
              EKIN=E-EFERMI
            ENDIF
            IF(EKIN.GT.DELTA) THEN
              VEL=SQRT(2.0D0*EKIN)
C  ****  Relativistic correction.
              EKEV=EKIN*HREV
              FREL=SQRT(2.0D0*(EKEV+REV)**2/(REV*(EKEV+2.0D0*REV)))
C  ****  Plasmon and electron-hole excitations.
              CALL XSFEG(RHO,DELTA,IELEC,EKIN,0,XSEC,1)
              RW(I)=-0.5D0*VEL*RHO*XSEC*RAD(I)*AABS*FREL
            ELSE
              RW(I)=0.0D0
            ENDIF
          ELSE
            RW(I)=0.0D0
          ENDIF
          RVPOL(I)=0.0D0
          WRITE(6,1503) I,RAD(I),RW(I)
        ENDDO
      ELSE
        WRITE(IW,1514)
 1514   FORMAT(1X,'#',/1X,'# No absorption potential')
        DO I=1,NP
          RW(I)=0.0D0
          RVPOL(I)=0.0D0
        ENDDO
      ENDIF
C
C  ****  We add a 'constant' tail to the potential to extend the grid
C  up to a point where irregular Coulomb functions can be be calculated.
C
      IF((MCPOL.NE.1.AND.MCPOL.NE.2).OR.EV.GT.1.0D4) THEN
        WRITE(IW,1710)
 1710   FORMAT(1X,'#',/1X,'# No correlation-polarization potential')
        IF(NP.LT.NDIM-10) THEN
          IF(RAD(NP)-RAD(NP-1).LT.1.0D-16) THEN
            I=NP
            RAD(I)=RAD(I-1)
          ELSE
            I=NP+1
            RAD(I)=RAD(I-1)
          ENDIF
          RV(I)=ZINF
          RVST(I)=ZINF/DBLE(IELEC)
          DEN(I)=0.0D0
          RVEX(I)=0.0D0
          RVPOL(I)=0.0D0
          RW(I)=0.0D0
          IST=I+1
          NADD=1
          DO I=IST,NDIM
            NADD=NADD+1
            RAD(I)=2.0D0*RAD(I-1)
            RV(I)=ZINF
            RVST(I)=ZINF/DBLE(IELEC)
            DEN(I)=0.0D0
            RVEX(I)=0.0D0
            RVPOL(I)=0.0D0
            RW(I)=0.0D0
            NP=I
            IF(RAD(I).GT.1.0D4.AND.NADD.GT.4) GO TO 2
          ENDDO
        ELSE
          STOP 'ELSEPA: Not enough memory space 1.'
        ENDIF
 2      CONTINUE
      ELSE IF(MCPOL.EQ.1) THEN
C
C  ************  Atomic polarizability correction.
C
C  ****  Buckingham empirical potential.
        WRITE(IW,1700) VPOLA,VPOLB
 1700   FORMAT(1X,'#',/1X,'# Correlation-polarization potential (Buc',
     1    'kingham):',/1X,'#',27X,'Alpha =',1P,E12.5,' cm**3',
     2     /1X,'#',28X,'Bpol =',E12.5)
        IF(VPOLB.LT.0.01D0) THEN
          WRITE(IW,*) 'ELSEPA: VPOLB cannot be less than 0.01.'
          STOP 'ELSEPA: VPOLB cannot be less than 0.01.'
        ENDIF
        ALPHA=VPOLA/A0B**3
        D2=SQRT(0.5D0*ALPHA*VPOLB**2/DBLE(IZ)**3.333333333333333D-1)
        NPOL=NP
        DO I=1,NPOL
          VPOL=-0.5D0*ALPHA/(RAD(I)**2+D2)**2
          RVPOL(I)=VPOL*RAD(I)
          RV(I)=RV(I)+RVPOL(I)
        ENDDO
        IF(NPOL.LT.NDIM-10) THEN
          DO I=NPOL+1,NDIM-10
            RAD(I)=1.25D0*RAD(I-1)
            VPOL=-0.5D0*ALPHA/(RAD(I)**2+D2)**2
            RVPOL(I)=VPOL*RAD(I)
            RVST(I)=ZINF
            RV(I)=ZINF+RVPOL(I)
            DEN(I)=0.0D0
            RVEX(I)=0.0D0
            RW(I)=0.0D0
            NP=I
            IF(ABS(VPOL).LT.1.0D-8*MAX(E,1.0D1*ABS(ZINF)/RAD(I))
     1        .AND.RAD(I).GT.50.0D0) GO TO 3
          ENDDO
        ENDIF
        STOP 'ELSEPA: Not enough memory space 2.'
 3      CONTINUE
        IF(NP.LT.NDIM-10) THEN
          I=NP+1
          RAD(I)=RAD(I-1)
          RVST(I)=ZINF
          RV(I)=ZINF
          DEN(I)=0.0D0
          RVEX(I)=0.0D0
          RW(I)=0.0D0
          RVPOL(I)=0.0D0
          NDIN=NP+10
          DO I=NP+2,NDIN
            RAD(I)=2.0D0*RAD(I-1)
            RVST(I)=ZINF
            RV(I)=ZINF
            DEN(I)=0.0D0
            RVEX(I)=0.0D0
            RW(I)=0.0D0
            RVPOL(I)=0.0D0
            NP=I
            IF(RAD(I).GT.1.0D4) GO TO 33
          ENDDO
        ELSE
          STOP 'ELSEPA: Not enough memory space 3.'
        ENDIF
 33     CONTINUE
C
      ELSE IF(MCPOL.EQ.2) THEN
C  ****  LDA correlation-polarization potential.
        IF(MUFIN.NE.1) THEN
          WRITE(IW,1701) VPOLA,VPOLB
 1701     FORMAT(1X,'#',/1X,'# Correlation-polarization potential (LDA',
     1      '):',/1X,'#',27X,'Alpha =',1P,E12.5,' cm**3',
     2      /1X,'#',28X,'Bpol =',E12.5)
          IF(VPOLB.LT.0.01D0) THEN
            WRITE(IW,*) 'ELSEPA: VPOLB cannot be less than 0.01.'
            STOP 'ELSEPA: VPOLB cannot be less than 0.01.'
          ENDIF
        ELSE
          WRITE(IW,1711)
 1711     FORMAT(1X,'#',/1X,'# Correlation potential (muffin-tin',
     1      ' model): LDA or Lindhard')
        ENDIF
        NPOL=NP
        IMODE=0
        IF(MUFIN.NE.1) THEN
          ALPHA=VPOLA/A0B**3
          D2=SQRT(0.5D0*ALPHA*VPOLB**2/DBLE(IZ)**3.333333333333333D-1)
          DO I=NPOL,1,-1
            RIP=RAD(MAX(2,I))
            RHO=DEN(MAX(2,I))/(FOURPI*RIP**2)
            VCO=VCPOL(IELEC,RHO)
C
            VPAS=-0.5D0*ALPHA/(RIP**2+D2)**2
            IF(IMODE.EQ.0) THEN
              VPOL=VPAS
              IF(VCO.LT.VPAS) IMODE=1
            ELSE
              VPOL=MAX(VCO,VPAS)
            ENDIF
            RVPOL(I)=VPOL*RAD(I)
            RV(I)=RV(I)+RVPOL(I)
          ENDDO
          IF(IMODE.EQ.0) THEN
            WRITE(IW,1702)
            WRITE(6,1702)
 1702       FORMAT(1X,'#',/1X,'# ERROR: The correlation and pol',
     1        'arization potentials do not cross.')
            STOP 'ELSEPA: V_corr and V_pol do not cross.'
          ENDIF
          VCOUT=-1.0D16
        ELSE
          VCOUT=0.0D0  ! Serves only to prevent compiler warnings.
          DO I=1,NPOL
            RIP=RAD(MAX(2,I))
            RHO=DEN(MAX(2,I))/(FOURPI*RIP**2)
            IF(I.LE.NMT) THEN
              VCO=VCPOL(IELEC,RHO)
C  ****  Lindhard high-E correlation potential.
              EKIN=E-DBLE(IELEC)*RVST(I)/RIP
              IF(EKIN.GT.1.0D-12) THEN
                VCOL=-SQRT((PI/2.0D0)**3*RHO/EKIN)
              ELSE
                VCOL=-1.0D35
              ENDIF
              VCO=MAX(VCO,VCOL)
              IF(I.EQ.NMT) VCOUT=VCO
            ELSE
              VCO=VCOUT
            ENDIF
            RVPOL(I)=VCO*RAD(I)
            RV(I)=RV(I)+RVPOL(I)
          ENDDO
          ERE0=ERE0-VCOUT
          DO I=2,NPOL
            RV(I)=RV(I)-RAD(I)*VCOUT
            RVPOL(I)=RVPOL(I)-RAD(I)*VCOUT
          ENDDO
          GO TO 34
        ENDIF
C
        IF(NPOL.LT.NDIM-10) THEN
          DO I=NPOL+1,NDIM-10
            IF(IMODE.EQ.0) THEN
              RAD(I)=RAD(I-1)+0.05D0
            ELSE
              RAD(I)=1.25D0*RAD(I-1)
            ENDIF
            VPOL=-0.5D0*ALPHA/(RAD(I)**2+D2)**2
            IF(IMODE.EQ.0) THEN
              IF(VPOL.GT.VCOUT) IMODE=1
              VPOL=MAX(VCOUT,VPOL)
            ENDIF
            RVPOL(I)=VPOL*RAD(I)
            RVST(I)=ZINF
            RV(I)=ZINF+RVPOL(I)
            DEN(I)=0.0D0
            RVEX(I)=0.0D0
            RW(I)=0.0D0
            NP=I
            IF(ABS(VPOL).LT.1.0D-8*MAX(E,1.0D1*ABS(ZINF)/RAD(I))
     1        .AND.RAD(I).GT.50.0D0) GO TO 34
          ENDDO
        ENDIF
        STOP 'ELSEPA: Not enough memory space 4.'
 34     CONTINUE
        IF(NP.LT.NDIM-10) THEN
          I=NP+1
          RAD(I)=RAD(I-1)
          RVST(I)=ZINF
          RV(I)=ZINF
          DEN(I)=0.0D0
          RVEX(I)=0.0D0
          RW(I)=0.0D0
          RVPOL(I)=0.0D0
          NDIN=NP+10
          DO I=NP+2,NDIN
            RAD(I)=2.0D0*RAD(I-1)
            RVST(I)=ZINF
            RV(I)=ZINF
            DEN(I)=0.0D0
            RVEX(I)=0.0D0
            RW(I)=0.0D0
            RVPOL(I)=0.0D0
            NP=I
            IF(RAD(I).GT.1.0D4) GO TO 35
          ENDDO
        ELSE
          STOP 'ELSEPA: Not enough memory space 5.'
        ENDIF
 35     CONTINUE
      ENDIF
C
C  ****  Muffin-tin absorption tail.
C
      IF(NMT.GT.0.AND.MABS.NE.0) THEN
        WAR=RW(NMT)/RAD(NMT)
        EIM=-WAR
        DO I=1,NP
          IF(RAD(I).LT.RAD(NMT)) THEN
            RW(I)=RW(I)-WAR*RAD(I)
          ELSE
            RW(I)=0.0D0
          ENDIF
        ENDDO
      ENDIF
C
C  ****  At high energies, we compute the DCS for scattering by the bare
C  nucleus and multiply it by a pre-evaluated screening factor.
C
 100  CONTINUE
      IF(IHEF0.EQ.1) THEN
        WRITE(IW,1800)
 1800   FORMAT(1X,'#',/1X,'# WARNING: High-energy factorization',
     1    ' with free-atom DF screening.',/1X,'#',
     2    10X,'Absorption, polarization and exchange corrections are',
     3    /1X,'#',10X,'switched off.',/1X,'#',10X,
     4    'Phase shifts are calculated for the bare nucleus.'/1X,'#',
     5    10X,'Scattering amplitudes are not evaluated.')
C  ****  Read screening function from data files.
        JT=IZ
        J1=JT-10*(JT/10)
        JT=(JT-J1)/10
        J2=JT-10*(JT/10)
        JT=(JT-J2)/10
        J3=JT-10*(JT/10)
        LIT1=LIT10(J1+1)
        LIT2=LIT10(J2+1)
        LIT3=LIT10(J3+1)
        FILE1=PATHE//'z_'//LIT3//LIT2//LIT1//'.dfs'
        NC=120
        CS120=' '
        I=0
        DO J=1,NC
          IF(FILE1(J:J).NE.' ') THEN
            I=I+1
            CS120(I:I)=FILE1(J:J)
          ENDIF
        ENDDO
        SCFILE=CS120
        WRITE(6,'(A)') SCFILE
C
        OPEN(99,FILE=SCFILE,STATUS='OLD',ERR=4)
        READ(99,'(1X,A1)') NULL
        READ(99,'(1X,A1)') NULL
        READ(99,'(1X,A1)') NULL
        NQS=0
        DO I=1,NGT
          READ(99,*,END=4) Q2T(I),FQ(I)
          NQS=I
        ENDDO
 4      CONTINUE
        CLOSE(UNIT=99)
        IF(NQS.EQ.0) STOP 'ELSEPA: I/O error. SCFILE does not exist.'
      ELSE IF(IHEF0.EQ.2) THEN
        WRITE(IW,1801)
 1801   FORMAT(1X,'#',/1X,'# WARNING: Scattering by the bare ',
     1    'nucleus.')
      ENDIF
C
      IF(IHEF0.GT.0) THEN
C  ----  The Coulomb tail of the potential is removed.
        NP=NPOT
        ZTAIL=RVN(NP)
        DO I=NPOT,1,-1
          IF(ABS(RVN(I)-ZTAIL).GT.1.0D-10) THEN
            NP=I+4
            GO TO 41
          ENDIF
        ENDDO
 41     CONTINUE
        DO I=1,NP
          RAD(I)=R(I)
          RV(I)=DBLE(IELEC)*RVN(I)
          RVST(I)=RVN(I)
          RVEX(I)=0.0D0
          RVPOL(I)=0.0D0
          RW(I)=0.0D0
        ENDDO
        RV(NP)=DBLE(IELEC)*DBLE(IZ)
C  ----  A Coulomb tail is appended to the potential table...
        IF(NP.LT.NDIM-4) THEN
          I=NP+1
          RAD(I)=RAD(I-1)
          RVST(I)=RVST(I-1)
          RV(I)=RV(I-1)
          DO I=NP+2,NDIM
            RAD(I)=2.0D0*RAD(I-1)
            RV(I)=RV(I-1)
            RVST(I)=RVST(I-1)
            RVEX(I)=0.0D0
            RVPOL(I)=0.0D0
            RW(I)=0.0D0
            NP=I
            IF(RAD(I).GT.1.0D4) GO TO 5
          ENDDO
 5        CONTINUE
        ELSE
          STOP 'ELSEPA: Not enough memory space 6'
        ENDIF
      ENDIF
C
      EEV=EV+ERE0*HREV
C
      OPEN(99,FILE='scfield.dat')
      WRITE(99,3001)
 3001 FORMAT(1X,'#  Scattering field.',/1X,'#  All quantities in',
     1  ' atomic units (a.u.), unless otherwise indicated.')
      IF(IELEC.EQ.-1) THEN
        WRITE(99,3002) IZ,NELEC
 3002   FORMAT(1X,'#  Z =',I4,', NELEC =',I4,
     1    ',   projectile: electron')
      ELSE
        WRITE(99,3003) IZ,NELEC
 3003   FORMAT(1X,'#  Z =',I4,', NELEC =',I4,
     1    ',   projectile: positron')
      ENDIF
      WRITE(99,3004) EV/HREV,EV
 3004 FORMAT(1X,'#  Kinetic energy =',1P,E12.5,' a.u. =',
     1    E12.5,' eV',/1X,'#')
      IF(NMT.GT.0) THEN
        WRITE(99,3005) NMT,RAD(NMT)
 3005   FORMAT(1X,'#  Muffin-tin radius = RAD(',I3,') =',
     1     1P,E12.5,' a.u.')
        WRITE(99,3105) VMOL
 3105   FORMAT(1X,'#  Atomic density    =',1P,E12.5,' 1/cm**3')
        WRITE(99,3006) ERE0,ERE0*HREV
 3006   FORMAT(1X,'#  Zero-energy shift = ',1P,E12.5,
     1     ' a.u. = ',E12.5,' eV')
        WRITE(99,3106) EIM,EIM*HREV
 3106   FORMAT(1X,'#  Background absorption potential = ',1P,E12.5,
     1     ' a.u. = ',E12.5,' eV')
        WRITE(99,3007) EEV/HREV,EEV
 3007   FORMAT(1X,'#  Effective kinetic energy =',1P,E12.5,
     1     ' a.u. =',E12.5,' eV',/1X,'#')
      ENDIF
      WRITE(99,3008)
 3008 FORMAT(1X,'#',3X,'i',7X,'r',11X,'r*V',9X,'r*Vst',8X,'r*Vex',7X,
     1      'r*Vpol',7X,'r*Wabs',8X,'rho_e',/1X,'#',96('-'))
      DO I=1,NP
        IF(IHEF0.GT.0) THEN
          RHO=0.0D0
        ELSE
          RHO=DEN(MAX(2,I))/(FOURPI*RAD(MAX(2,I))**2)
        ENDIF
        WRITE(99,'(2X,I4,1P,7E13.5)') I,RAD(I),RV(I),
     1    IELEC*RVST(I),RVEX(I),RVPOL(I),RW(I),RHO
      ENDDO
      CLOSE(99)
C
C  ************  Partial-wave analysis.
C
      NRT=NP
      NPI=NP
      DO I=1,NRT
        RRR(I)=RAD(I)
        RADI(I)=RAD(I)
        RVI(I)=RV(I)
      ENDDO
C
      NDELTA=NDM
      IF(IAB.EQ.0) THEN
        IF(IHEF0.GT.0) THEN
          CALL DPWA0(EEV,NDELTA,2)
        ELSE
          IF(EEV.LT.1.001D3) THEN
            ISCH=1
          ELSE
            ISCH=2
          ENDIF
          IF(MUFIN.EQ.1) ISCH=1
          CALL DPWA0(EEV,NDELTA,ISCH)
        ENDIF
        TOTCS=ECS
        ABCS=0.0D0
      ELSE
        IF(EEV.LT.1.001D3) THEN
          ISCH=1
        ELSE
          ISCH=2
        ENDIF
        IF(MUFIN.EQ.1) ISCH=1
        CALL DPWAI0(EEV,TOTCS,ABCS,NDELTA,ISCH)
      ENDIF
C
C  ************  DCS table.
C
      IF(IHEF0.EQ.1) THEN
        NTABT=NTAB
        DO I=1,NTAB
C  ****  Screening correction and check for numerical artifacts.
          Q2=4.0D0*RK2*XT(I)
          IF(Q2.LT.Q2T(NQS)) THEN
            CALL FINDI(Q2,Q2T,NQS,J)
            F=FQ(J)+(FQ(J+1)-FQ(J))*(Q2-Q2T(J))/(Q2T(J+1)-Q2T(J))
          ELSE
            F=1.0D0
          ENDIF
          IF(TH(I).GT.1.0D0.AND.ERROR(I).LT.1.0D-2) THEN
            DCST(I)=DCST(I)*(Q2*F/(1.0D0+Q2))**2
          ELSE IF(TH(I).LT.10.0001D0) THEN
            THRAD=TH(I)*PI/180.0D0
            RMR=DPWAC(THRAD)
            DCST(I)=RUTHC*F*F*RMR/(1.0D0+Q2)**2
            ERROR(I)=1.0D-5
          ELSE
            IF(ERROR(I).GT.1.0D-2) THEN
              NTABT=I-1
              GO TO 6
            ENDIF
          ENDIF
          SPOL(I)=0.0D0
        ENDDO
 6      CONTINUE
        IF(NTABT.LT.NTAB) THEN
          DO I=NTABT,NTAB
            DCST(I)=1.0D-45
            ERROR(I)=1.0D0
          ENDDO
        ENDIF
      ENDIF
C
C  ****  Small-angle DCS for ions.
C
      XTL=0.0D0
      IMATCH=1
      IF(IZ.NE.NELEC) THEN
        DO I=1,NTAB-2
          IF(MAX(ERROR(I),ERROR(I+1),ERROR(I+2)).LT.5.0D-4) THEN
            IMATCH=I
            GO TO 7
          ENDIF
        ENDDO
 7      CONTINUE
        THL=MAX(0.5D0,TH(IMATCH))
        WRITE(IW,2012) THL
 2012   FORMAT(1X,'#',/1X,'# WARNING: DCSs are calculated and inte',
     1    'grated only for angles',/1X,'#',10X,'THETA .gt.',1P,E10.3,
     2    ' deg')
        XTL=SIN(THL*PI/360.0D0)**2
        DO I=1,IMATCH-1
          DCST(I)=0.0D0
          SPOL(I)=0.0D0
          ERROR(I)=1.0D0
        ENDDO
      ENDIF
C
C  ****  Integrated cross sections.
C
      ECS0=FOURPI*SMOMLL(XT,DCST,XTL,XT(NTAB),NTAB,0,0)
      ECS1=FOURPI*SMOMLL(XT,DCST,XTL,XT(NTAB),NTAB,1,0)
      ECS2=FOURPI*SMOMLL(XT,DCST,XTL,XT(NTAB),NTAB,2,0)
      ECS=ECS0
      TCS1=2.0D0*ECS1
      TCS2=6.0D0*(ECS1-ECS2)
      WRITE(IW,2212)
 2212 FORMAT(1X,'#',/1X,'# Integrated cross sections:')
      WRITE(IW,2013) ECS,ECS/A0B2
 2013 FORMAT(1X,'# Total elastic cross section =',1P,
     1     E12.5,' cm**2 =',E12.5,' a0**2')
      WRITE(IW,2014) TCS1,TCS1/A0B2
 2014 FORMAT(1X,'# 1st transport cross section =',1P,
     1     E12.5,' cm**2 =',E12.5,' a0**2')
      WRITE(IW,2015) TCS2,TCS2/A0B2
 2015 FORMAT(1X,'# 2nd transport cross section =',1P,
     1     E12.5,' cm**2 =',E12.5,' a0**2',/1X,'#')
      IF(IAB.EQ.1.AND.IZ.EQ.NELEC) THEN
        IF(NMT.GT.0) THEN
C  ****  Muffin-tin, atoms in solids. Contribution of the constant
C  imaginary potential to the absorption cross section.
          BETA=SQRT(EV*(EV+2.0D0*REV)/(EV+REV)**2)
          ABCSO=A0B2*2.0D0*EIM/(BETA*SL*VMOL1)
          ABCS=ABCS+ABCSO
          TOTCS=TOTCS+ABCSO
        ENDIF
C
        WRITE(IW,2016) ABCS,ABCS/A0B2
 2016   FORMAT(1X,'#    Absorption cross section =',1P,
     1     E12.5,' cm**2 =',E12.5,' a0**2')
        WRITE(IW,2017) TOTCS,TOTCS/A0B2
 2017   FORMAT(1X,'#   Grand total cross section =',1P,
     1     E12.5,' cm**2 =',E12.5,' a0**2',/1X,'#')
      ENDIF
C
      ECUT=MAX(20.0D3*IZ,2.0D6)
      WRITE(IW,'(1X,''#'')')
      WRITE(IW,'(1X,''# Differential cross section:'',23X,
     1  ''MU=(1-COS(THETA))/2'')')
      WRITE(IW,'(1X,''#'',/1X,''#  THETA'',8X,''MU'',10X,''DCS'',
     1  10X,''DCS'',8X,''Sherman'',6X,''error''/1X,''#  (deg)'',
     2  17X,''(cm**2/sr)'',3X,''(a0**2/sr)'',4X,''function'',
     3  /1X,''#'',70(''-''))')
C
      DO I=1,NTAB
        WRITE(IW,2018) TH(I),XT(I),DCST(I),DCST(I)/A0B2,SPOL(I),ERROR(I)
      ENDDO
 2018 FORMAT(1X,1P,E10.3,E13.5,3E13.5,E9.1)
C
C  ****  Scattering amplitudes.
C
      OPEN(99,FILE='scatamp.dat')
      WRITE(99,'(1X,''#  Scattering amplitudes (in cm)'',
     1  6X,''MU=(1-cos(TH))/2'')')
      IF(IELEC.EQ.-1) THEN
        WRITE(99,3010) IZ
 3010   FORMAT(1X,'#  Z =',I4,',   projectile: electron')
      ELSE
        WRITE(99,3011) IZ
 3011   FORMAT(1X,'#  Z =',I4,',   projectile: positron')
      ENDIF
      WRITE(99,3012) EV
 3012 FORMAT(1X,'#  Kinetic energy =',1P,E12.5,' eV',/1X,'#')
      WRITE(99,'(1X,''# TH (deg)'',6X,''MU'',10X,''Re(F)'',8X,
     1  ''Im(F)'',8X,''Re(G)'',8X,''Im(G)'',/1X,''#'',
     2  74(''-''))')
      IF(IHEF0.NE.0) RETURN
      DO I=1,NTAB
        THRAD=TH(I)*PI/180.0D0
        IF(EV.LT.ECUT) THEN
          CALL DPWA(THRAD,CF,CG,DCS,SPL,ERRF,ERRG)
        ELSE
          CF=0.0D0
          CG=0.0D0
        ENDIF
        WRITE(99,3013) TH(I),XT(I),CF,CG
      ENDDO
 3013 FORMAT(1X,1P,E10.3,E13.5,4E13.5)
      CLOSE(99)
C
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE EFIELD
C  *********************************************************************
      SUBROUTINE EFIELD(IZ,NELEC,MNUCL,MELEC,IW,IWR)
C
C     Electrostatic potential of atoms and ions.
C
C  Input parameters:
C     IZ ....... atomic number (INTEGER).
C     NELEC .... number of electrons (INTEGER, .GE.0 and .LE.IZ).
C     MNUCL .... nuclear density model (INTEGER).
C                 1 --> point nucleus,
C                 2 --> uniform distribution,
C                 3 --> Fermi distribution,
C                 4 --> Helm's uniform-uniform distribution.
C     MELEC .... electron density model (INTEGER).
C                 1 --> TFM analytical density,
C                 2 --> TFD analytical density,
C                 3 --> DHFS analytical density,
C                 4 --> DF density from pre-evaluated files,
C                 5 --> density read from file 'density.usr'.
C     IW ....... output unit (to be defined in the main program).
C     IWR ...... if >0, the potential is written in a file named
C                'esfield.dat'.
C
C  Output (through common block /CFIELD/):
C     R(I) ..... radial grid points. R(1)=0.0D0.
C     RVN(I) ... nuclear potential times R.
C     DEN(I) ... radial electron density, i.e. the electron density
C                multiplied by 4*PI*R**2.
C     RVST(I) ... atomic electrostatic potential (nuclear+electronic)
C                times R.
C     NPOT ..... number of grid points where the potential function is
C                tabulated. For I.GT.NPOT, RVST(I) is set equal to
C                RVST(NPOT).
C
      USE CONSTANTS
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      CHARACTER*1 LIT10(10),LIT1,LIT2,LIT3
      CHARACTER*120 ELFILE,FILE1,CS120,NULL
      CHARACTER*2 LSYMBL(103)
C
      PARAMETER (F2BOHR=1.0D-13/A0B)
      PARAMETER (A0B2=A0B*A0B)
      PARAMETER (PI=3.1415926535897932D0, FOURPI=4.0D0*PI)
C
      DIMENSION DIFR(NDIM),DENN(NDIM),RVE(NDIM),ELAW(103)
      COMMON/CFIELD/R(NDIM),RVN(NDIM),DEN(NDIM),RVST(NDIM),NPOT
      PARAMETER (NPPG=NDIM+1,NPTG=NDIM+NPPG)
      DIMENSION AUX(NPTG),A(NPTG),B(NPTG),C(NPTG),D(NPTG)
C
      DATA LIT10/'0','1','2','3','4','5','6','7','8','9'/
C
      DATA LSYMBL       /' H','He','Li','Be',' B',' C',' N',' O',
     1    ' F','Ne','Na','Mg','Al','Si',' P',' S','Cl','Ar',' K',
     2    'Ca','Sc','Ti',' V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
     3    'Ga','Ge','As','Se','Br','Kr','Rb','Sr',' Y','Zr','Nb',
     4    'Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te',
     5    ' I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu',
     6    'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta',' W',
     7    'Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At',
     8    'Rn','Fr','Ra','Ac','Th','Pa',' U','Np','Pu','Am','Cm',
     9    'Bk','Cf','Es','Fm','Md','No','Lr'/
      DATA ELAW     /1.007900D0,4.002600D0,6.941000D0,9.012200D0,
     1    1.081100D1,1.201070D1,1.400670D1,1.599940D1,1.899840D1,
     2    2.017970D1,2.298980D1,2.430500D1,2.698150D1,2.808550D1,
     3    3.097380D1,3.206600D1,3.545270D1,3.994800D1,3.909830D1,
     4    4.007800D1,4.495590D1,4.786700D1,5.094150D1,5.199610D1,
     5    5.493800D1,5.584500D1,5.893320D1,5.869340D1,6.354600D1,
     6    6.539000D1,6.972300D1,7.261000D1,7.492160D1,7.896000D1,
     7    7.990400D1,8.380000D1,8.546780D1,8.762000D1,8.890590D1,
     8    9.122400D1,9.290640D1,9.594000D1,9.890630D1,1.010700D2,
     9    1.029055D2,1.064200D2,1.078682D2,1.124110D2,1.148180D2,
     1    1.187100D2,1.217600D2,1.276000D2,1.269045D2,1.312900D2,
     1    1.329054D2,1.373270D2,1.389055D2,1.401160D2,1.409076D2,
     2    1.442400D2,1.449127D2,1.503600D2,1.519640D2,1.572500D2,
     3    1.589253D2,1.625000D2,1.649303D2,1.672600D2,1.689342D2,
     4    1.730400D2,1.749670D2,1.784900D2,1.809479D2,1.838400D2,
     5    1.862070D2,1.902300D2,1.922170D2,1.950780D2,1.969666D2,
     6    2.005900D2,2.043833D2,2.072000D2,2.089804D2,2.089824D2,
     7    2.099871D2,2.220176D2,2.230197D2,2.260254D2,2.270277D2,
     8    2.320381D2,2.310359D2,2.380289D2,2.370482D2,2.440642D2,
     9    2.430614D2,2.470000D2,2.470000D2,2.510000D2,2.520000D2,
     1    2.570000D2,2.580000D2,2.590000D2,2.620000D2/
C
C  ****  Path to the ELSEPA database
      CHARACTER*100 PATHE
      PATHE='./database/'
C
      IF(IZ.LE.0) STOP 'EFIELD: Negative atomic number.'
      IF(IZ.GT.103) STOP 'EFIELD: Atomic number larger than 103.'
      IF(NELEC.GT.IZ) STOP 'EFIELD: Negative ion.'
      IF(NELEC.LT.0) STOP 'EFIELD: Negative number of electrons.'
C
      Z=DBLE(IZ)
      AW=ELAW(IZ)
      IF(IW.GT.0) WRITE(IW,1001) LSYMBL(IZ),IZ,AW
 1001 FORMAT(1X,'#',/1X,'# Element: ',A2,',  Z = ',I3,
     1  ',  atomic weight =',1P,E12.5,' g/mol')
      NDIN=1000
C
C  ************  Nuclear electrostatic potential (times R).
C
      IF(MNUCL.EQ.1) THEN
        IF(IW.GT.0) WRITE(IW,1002)
 1002   FORMAT(1X,'#',/1X,'# Nuclear model: point charge')
C
        RTN=100.0D0
        RT2=2.0D-8
        DRN=RTN/DBLE(NDIN/3)
        CALL SGRID(R,DIFR,RTN,RT2,DRN,NDIN,NDIM,IER)
C
        DO I=1,NDIN
          RVN(I)=Z
          DENN(I)=0.0D0
        ENDDO
        GO TO 10
      ELSE IF (MNUCL.EQ.2) THEN
C  ****  The factor F2BOHR=1.889726D-5 transforms from fm to Bohr.
        R1=1.07D0*F2BOHR*AW**0.3333333333333333D0
        R2=2.0D0*F2BOHR
        R1=R1*SQRT((1.0D0+2.5D0*(R2/R1)**2)
     1             /(1.0D0+0.75D0*(R2/R1)**2))
        R0=R1/10.0D0
C
        RTN=100.0D0
        RT2=MAX(0.1D0*R0,1.01D-8)
        DRN=RTN/DBLE(NDIN/3)
        CALL SGRID(R,DIFR,RTN,RT2,DRN,NDIN,NDIM,IER)
C
        IF(IW.GT.0) WRITE(IW,1004) R1*A0B
 1004   FORMAT(1X,'#',/1X,'# Nuclear model: uniform spherical d',
     1    'istribution',/1X,'#',16X,'Nuclear radius =',1P,E12.5,
     2    ' cm')
        DO I=1,NDIN
          X=R(I)
          IF(X.LT.R1) THEN
            RVN(I)=Z*(1.5D0-0.5D0*(X/R1)**2)*(X/R1)
            DENN(I)=Z/(FOURPI*R1**3/3.0D0)
          ELSE
            RVN(I)=Z
            DENN(I)=0.0D0
          ENDIF
        ENDDO
        GO TO 10
      ELSE IF (MNUCL.EQ.3) THEN
        R1=1.07D0*F2BOHR*AW**0.3333333333333333D0
        R2=0.546D0*F2BOHR
        R0=R1/10.0D0
C
        RTN=100.0D0
        RT2=MAX(0.1D0*R0,1.01D-8)
        DRN=RTN/DBLE(NDIN/3)
        CALL SGRID(R,DIFR,RTN,RT2,DRN,NDIN,NDIM,IER)
C
        IF(IW.GT.0) WRITE(IW,1005) R1*A0B,R2*A0B
 1005   FORMAT(1X,'#',/1X,'# Nuclear model: Fermi distribution',
     1    /1X,'#',16X,'Average radius =',1P,E12.5, ' cm',
     2    /1X,'#',16X,'Skin thickness =',E12.5,' cm')
C  ****  The array DENN contains the nuclear charge density.
C        (unnormalized).
        DO I=1,NDIN
          X=R(I)
          XX=EXP((R1-X)/R2)
          DENN(I)=XX/(1.0D0+XX)
          RVN(I)=DENN(I)*X**2*DIFR(I)
        ENDDO
      ELSE
        RNUC=1.070D0*F2BOHR*AW**0.3333333333333333D0
        R1=0.96219D0*RNUC+0.435D0*F2BOHR
        R2=2.0D0*F2BOHR
        IF(IW.GT.0) WRITE(IW,1105) R1*A0B,R2*A0B
 1105   FORMAT(1X,'#',/1X,'# Nuclear model: Helm''s Uu distribu',
     1    'tion',/1X,'#',16X,'  Inner radius =',1P,E12.5, ' cm',
     2    /1X,'#',16X,'Skin thickness =',E12.5,' cm')
        IF(R2.GT.R1) THEN
          STORED=R1
          R1=R2
          R2=STORED
        ENDIF
        R0=R1/10.0D0
C
        RTN=100.0D0
        RT2=MAX(0.1D0*R0,1.01D-8)
        DRN=RTN/DBLE(NDIN/3)
        CALL SGRID(R,DIFR,RTN,RT2,DRN,NDIN,NDIM,IER)
C
        DO I=1,NDIN
          RR=R(I)
          IF(RR.LT.R1-R2) THEN
            V=1.0D0
          ELSE IF(RR.GT.R1+R2) THEN
            V=0.0D0
          ELSE
            T=RR*RR+R1*R1-R2*R2
            V1=(T+4.0D0*RR*R1)*(T-2.0D0*RR*R1)**2
            T=RR*RR+R2*R2-R1*R1
            V2=(T+4.0D0*RR*R2)*(T-2.0D0*RR*R2)**2
            V=(V1+V2)/(32.0D0*(R2*RR)**3)
          ENDIF
          DENN(I)=V
          RVN(I)=DENN(I)*RR**2*DIFR(I)
        ENDDO
      ENDIF
C
      CALL SLAG6(1.0D0,RVN,RVN,NDIN)
      NDIN1=NDIN+1
      DO I=1,NDIN
        K=NDIN1-I
        AUX(I)=DENN(K)*R(K)*DIFR(K)
      ENDDO
      CALL SLAG6(1.0D0,AUX,AUX,NDIN)
      FNORM=Z/RVN(NDIN)
      DO I=1,NDIN
        RVN(I)=FNORM*(RVN(I)+AUX(NDIN1-I)*R(I))
        DENN(I)=FNORM*DENN(I)/FOURPI
        IF(DENN(I).LT.1.0D-35) DENN(I)=0.0D0
      ENDDO
 10   CONTINUE
C
C  ************  Electronic electrostatic potential (times R).
C
      IF(IW.GT.0) WRITE(IW,1006) NELEC
 1006 FORMAT(1X,'#',/1X,'# Number of electrons =',I3)
      IF(NELEC.EQ.0) THEN
        DO I=1,NDIN
          DEN(I)=0.0D0
          RVE(I)=0.0D0
        ENDDO
        GO TO 2
      ENDIF
C
      IF(MELEC.LT.4.OR.MELEC.GT.5) THEN
C  ****  Analytical electron density models.
        IF(MELEC.EQ.1) THEN
          CALL TFM(IZ,A1,A2,A3,AL1,AL2,AL3)
          IF(IW.GT.0) WRITE(IW,1007)
 1007     FORMAT(1X,'#',/1X,
     1      '# Electron density: analytical TFM model')
        ELSE IF(MELEC.EQ.2) THEN
          CALL TFD(IZ,A1,A2,A3,AL1,AL2,AL3)
          IF(IW.GT.0) WRITE(IW,1008)
 1008     FORMAT(1X,'#',/1X,
     1      '# Electron density: analytical TFD model')
        ELSE
          CALL DHFS(IZ,A1,A2,A3,AL1,AL2,AL3)
          IF(IW.GT.0) WRITE(IW,1009)
 1009     FORMAT(1X,'#',/1X,
     1      '# Electron density: analytical DHFS model')
        ENDIF
        IF(IW.GT.0) WRITE(IW,1010) A1,AL1,A2,AL2,A3,AL3
 1010   FORMAT(1X,'#',19X,'A1 = ',1P,D12.5,' ,   ALPHA1 =',D12.5,
     1      /1X,'#',19X,'A2 = ',D12.5,' ,   ALPHA2 =',D12.5,
     2      /1X,'#',19X,'A3 = ',D12.5,' ,   ALPHA3 =',D12.5)
        XN=DBLE(NELEC)
        DO I=1,NDIN
          DEN(I)=(A1*AL1*AL1*EXP(-AL1*R(I))
     1           +A2*AL2*AL2*EXP(-AL2*R(I))
     2           +A3*AL3*AL3*EXP(-AL3*R(I)))*XN
        ENDDO
      ELSE
C  ****  Electron density read from a file.
        NE=0
        IF(MELEC.EQ.4) THEN
          JT=IZ
          J1=JT-10*(JT/10)
          JT=(JT-J1)/10
          J2=JT-10*(JT/10)
          JT=(JT-J2)/10
          J3=JT-10*(JT/10)
          LIT1=LIT10(J1+1)
          LIT2=LIT10(J2+1)
          LIT3=LIT10(J3+1)
          FILE1=PATHE//'z_'//LIT3//LIT2//LIT1//'.den'
          NC=120
          CS120=' '
          I=0
          DO J=1,NC
            IF(FILE1(J:J).NE.' ') THEN
              I=I+1
              CS120(I:I)=FILE1(J:J)
            ENDIF
          ENDDO
          ELFILE=CS120
          WRITE(6,'(A)') ELFILE
        ELSE
          ELFILE='density.usr'
        ENDIF
C
        IF(IW.GT.0) WRITE(IW,1011) ELFILE
 1011   FORMAT(1X,'#',/1X,'# Electron density: Read from file ',A120)
        OPEN(99,FILE=ELFILE,STATUS='OLD',ERR=1)
        READ(99,'(A12)') NULL
        READ(99,'(A12)') NULL
        READ(99,'(A12)') NULL
        DO I=1,NDIN
          READ(99,*,END=1) AUX(I),RVE(I)
          NE=I
          RVE(I)=LOG(RVE(I))
        ENDDO
        STOP 'EFIELD: File is too large.'
 1      CONTINUE
        IF(NE.EQ.0) STOP 'EFIELD: I/O error in EFIELD.'
        CLOSE(99)
C
        IF(IW.GT.0) WRITE(IW,1012) NE
 1012   FORMAT(1X,'#',19X,'Number of data points = ',I4)
        IF(NE.LT.4) STOP 'EFIELD: SPLINE needs more than 4 points.'
C  ****  ... and interpolated (lin-log cubic spline).
        CALL SPLINE(AUX,RVE,A,B,C,D,0.0D0,0.0D0,NE)
        B(NE)=(RVE(NE)-RVE(NE-1))/(AUX(NE)-AUX(NE-1))
        A(NE)=RVE(NE-1)-B(NE)*AUX(NE-1)
        C(NE)=0.0D0
        D(NE)=0.0D0
        DO I=1,NDIN
          X=R(I)
          IF(X.GT.AUX(NE)) THEN
            DEN(I)=0.0D0
          ELSE
            CALL FINDI(X,AUX,NE,J)
            DEN(I)=EXP(A(J)+X*(B(J)+X*(C(J)+X*D(J))))*X*FOURPI
          ENDIF
        ENDDO
      ENDIF
C  ****  Calculation of the electrostatic potential.
      DO I=1,NDIN
        RVE(I)=DEN(I)*R(I)*DIFR(I)
      ENDDO
      CALL SLAG6(1.0D0,RVE,RVE,NDIN)
      NDIN1=NDIN+1
      DO I=1,NDIN
        K=NDIN1-I
        AUX(I)=DEN(K)*DIFR(K)
      ENDDO
      CALL SLAG6(1.0D0,AUX,AUX,NDIN)
      IF(IW.GT.0) WRITE(IW,1013) RVE(NDIN)
 1013 FORMAT(1X,'#',19X,'Volume integral =',1P,E12.5)
      FNORM=DBLE(NELEC)/RVE(NDIN)
      DO I=1,NDIN
        RVE(I)=FNORM*(RVE(I)+AUX(NDIN1-I)*R(I))
        DEN(I)=FNORM*DEN(I)*R(I)
      ENDDO
C
 2    CONTINUE
      ZINF=DBLE(IZ-NELEC)
      DO I=1,NDIN
        RVST(I)=RVN(I)-RVE(I)
      ENDDO
      NPOT=NDIN
      DO I=NDIN,6,-1
        IF((ABS(RVST(I)-ZINF).LT.5.0D-12).
     1       AND.(ABS(DEN(I)).LT.5.0D-12)) THEN
          RVST(I)=ZINF
          NPOT=I
          IF(R(I).LT.1.0D0) GO TO 3
        ELSE
          GO TO 3
        ENDIF
      ENDDO
 3    CONTINUE
C
      IF(IWR.GT.0) THEN
        OPEN(99,FILE='esfield.dat')
        WRITE(99,'(1X,''#  Electrostatic field (a.u.)'')')
        WRITE(99,2001) LSYMBL(IZ),IZ,AW
 2001   FORMAT(1X,'#  Element: ',A2,',  Z = ',I3,
     1      ',  atomic weight =',1P,E14.7,' g/mol')
        WRITE(99,2002) NELEC
 2002 FORMAT(1X,'#  Number of electrons =',I3)
        WRITE(99,2003) MNUCL,MELEC
 2003 FORMAT(1X,'#  MNUCL =',I3,',   MELEC =',I3,/1X,'#')
        WRITE(99,2004)
 2004 FORMAT(1X,'#',3X,'I',7X,'R(I)',11X,'RHON(I)',9X,'RVN(I)',10X,
     1         'DEN(I)',10X,'RVST(I)',/1X,'#',84('-'))
        DO I=1,NPOT
         WRITE(99,'(2X,I4,1P,5E16.8)')
     1     I,R(I),DENN(I),RVN(I),DEN(I),RVST(I)
        ENDDO
        CLOSE(99)
      ENDIF
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE TFM
C  *********************************************************************
      SUBROUTINE TFM(IZ,A1,A2,A3,AL1,AL2,AL3)
C
C     Parameters in Moliere's analytical approximation (three Yukawa
C  terms) to the Thomas-Fermi atomic screening function.
C     Ref.: G. Moliere, Z. Naturforsch. 2a (1947) 133.
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      Z=DBLE(IZ)
      RTF=0.88534D0/Z**0.33333333333D0
      AL1=6.0D0/RTF
      AL2=1.2D0/RTF
      AL3=0.3D0/RTF
      A1=0.10D0
      A2=0.55D0
      A3=0.35D0
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE TFD
C  *********************************************************************
      SUBROUTINE TFD(IZ,A1,A2,A3,AL1,AL2,AL3)
C
C     Parameters in the analytical approximation (three Yukawa terms)
C  for the Thomas-Fermi-Dirac atomic screening function.
C     Ref.: R.A. Bonham and T.G. Strand, J. Chem. Phys. 39 (1963) 2200.
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      DIMENSION AA1(5),AA2(5),AA3(5),AAL1(5),AAL2(5),AAL3(5)
      DATA AA1/1.26671D-2,-2.61047D-2,2.14184D-2,-2.35686D-3,
     12.10672D-5/
      DATA AA2/5.80612D-2,2.93077D-2,8.57135D-2,-2.23342D-2,
     11.64675D-3/
      DATA AA3/9.27968D-1,-1.64643D-3,-1.07685D-1,2.47998D-2,
     1-1.67822D-3/
      DATA AAL1/1.64564D2,-1.52192D2,6.23879D1,-1.15005D1,
     18.08424D-1/
      DATA AAL2/1.13060D1,-6.31902D0,2.26025D0,-3.70738D-1,
     12.61151D-2/
      DATA AAL3/1.48219D0,-5.57601D-2,1.64387D-2,-4.39703D-3,
     19.97225D-4/
C
      IF(IZ.LE.0) THEN
        WRITE(6,100)
 100    FORMAT(5X,'*** TFD: Negative atomic number. STOP.')
        STOP 'TFD: Negative atomic number.'
      ENDIF
C
      X=LOG(DBLE(IZ))
      A1=AA1(1)+X*(AA1(2)+X*(AA1(3)+X*(AA1(4)+X*AA1(5))))
      A2=AA2(1)+X*(AA2(2)+X*(AA2(3)+X*(AA2(4)+X*AA2(5))))
      A3=AA3(1)+X*(AA3(2)+X*(AA3(3)+X*(AA3(4)+X*AA3(5))))
      AL1=AAL1(1)+X*(AAL1(2)+X*(AAL1(3)+X*(AAL1(4)+X*AAL1(5))))
      AL2=AAL2(1)+X*(AAL2(2)+X*(AAL2(3)+X*(AAL2(4)+X*AAL2(5))))
      AL3=AAL3(1)+X*(AAL3(2)+X*(AAL3(3)+X*(AAL3(4)+X*AAL3(5))))
      A3=1.0D0-A1-A2
C
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE DHFS
C  *********************************************************************
      SUBROUTINE DHFS(IZ,A1,A2,A3,AL1,AL2,AL3)
C
C     DHFS analytical screening function parameters for free neutral
C  atoms. The input argument is the atomic number.
C
C     Ref.: F. Salvat et al., Phys. Rev. A36 (1987) 467-474.
C     Elements from Z=93 to 103 added in march 1992.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      DIMENSION B1(103),B2(103),BL1(103),BL2(103),BL3(103)
      DATA B1/-7.05665D-6,-2.25920D-1,6.04537D-1,3.27766D-1,
     1   2.32684D-1,1.53676D-1,9.95750D-2,6.25130D-2,3.68040D-2,
     2   1.88410D-2,7.44440D-1,6.42349D-1,6.00152D-1,5.15971D-1,
     3   4.38675D-1,5.45871D-1,7.24889D-1,2.19124D+0,4.85607D-2,
     4   5.80017D-1,5.54340D-1,1.11950D-2,3.18350D-2,1.07503D-1,
     5   4.97556D-2,5.11841D-2,5.00039D-2,4.73509D-2,7.70967D-2,
     6   4.00041D-2,1.08344D-1,6.09767D-2,2.11561D-2,4.83575D-1,
     7   4.50364D-1,4.19036D-1,1.73438D-1,3.35694D-2,6.88939D-2,
     8   1.17552D-1,2.55689D-1,2.69313D-1,2.20138D-1,2.75057D-1,
     9   2.71053D-1,2.78363D-1,2.56210D-1,2.27100D-1,2.49215D-1,
     A   2.15313D-1,1.80560D-1,1.30772D-1,5.88293D-2,4.45145D-1,
     B   2.70796D-1,1.72814D-1,1.94726D-1,1.91338D-1,1.86776D-1,
     C   1.66461D-1,1.62350D-1,1.58016D-1,1.53759D-1,1.58729D-1,
     D   1.45327D-1,1.41260D-1,1.37360D-1,1.33614D-1,1.29853D-1,
     E   1.26659D-1,1.28806D-1,1.30256D-1,1.38420D-1,1.50030D-1,
     F   1.60803D-1,1.72164D-1,1.83411D-1,2.23043D-1,2.28909D-1,
     G   2.09753D-1,2.70821D-1,2.37958D-1,2.28771D-1,1.94059D-1,
     H   1.49995D-1,9.55262D-2,3.19155D-1,2.40406D-1,2.26579D-1,
     I   2.17619D-1,2.41294D-1,2.44758D-1,2.46231D-1,2.55572D-1,
     J   2.53567D-1,2.43832D-1,2.41898D-1,2.44050D-1,2.40237D-1,
     K   2.34997D-1,2.32114D-1,2.27937D-1,2.29571D-1/
      DATA B2/-1.84386D+2,1.22592D+0,3.95463D-1,6.72234D-1,
     1   7.67316D-1,8.46324D-1,9.00425D-1,9.37487D-1,9.63196D-1,
     2   9.81159D-1,2.55560D-1,3.57651D-1,3.99848D-1,4.84029D-1,
     3  5.61325D-1,-5.33329D-1,-7.54809D-1,-2.2852D0,7.75935D-1,
     4   4.19983D-1,4.45660D-1,6.83176D-1,6.75303D-1,7.16172D-1,
     5   6.86632D-1,6.99533D-1,7.14201D-1,7.29404D-1,7.95083D-1,
     6   7.59034D-1,7.48941D-1,7.15671D-1,6.70932D-1,5.16425D-1,
     7   5.49636D-1,5.80964D-1,7.25336D-1,7.81581D-1,7.20203D-1,
     8   6.58088D-1,5.82051D-1,5.75262D-1,5.61797D-1,5.94338D-1,
     9   6.11921D-1,6.06653D-1,6.50520D-1,6.15496D-1,6.43990D-1,
     A   6.11497D-1,5.76688D-1,5.50366D-1,5.48174D-1,5.54855D-1,
     B   6.52415D-1,6.84485D-1,6.38429D-1,6.46684D-1,6.55810D-1,
     C   7.05677D-1,7.13311D-1,7.20978D-1,7.28385D-1,7.02414D-1,
     D   7.42619D-1,7.49352D-1,7.55797D-1,7.61947D-1,7.68005D-1,
     E   7.73365D-1,7.52781D-1,7.32428D-1,7.09596D-1,6.87141D-1,
     F   6.65932D-1,6.46849D-1,6.30598D-1,6.17575D-1,6.11402D-1,
     G   6.00426D-1,6.42829D-1,6.30789D-1,6.21959D-1,6.10455D-1,
     H   6.03147D-1,6.05994D-1,6.23324D-1,6.56665D-1,6.42246D-1,
     I   6.24013D-1,6.30394D-1,6.29816D-1,6.31596D-1,6.49005D-1,
     J   6.53604D-1,6.43738D-1,6.48850D-1,6.70318D-1,6.76319D-1,
     K   6.65571D-1,6.88406D-1,6.94394D-1,6.82014D-1/
      DATA BL1/ 4.92969D+0,5.52725D+0,2.81741D+0,4.54302D+0,
     1   5.99006D+0,8.04043D+0,1.08122D+1,1.48233D+1,2.14001D+1,
     2   3.49994D+1,4.12050D+0,4.72663D+0,5.14051D+0,5.84918D+0,
     3   6.67070D+0,6.37029D+0,6.21183D+0,5.54701D+0,3.02597D+1,
     4   6.32184D+0,6.63280D+0,9.97569D+1,4.25330D+1,1.89587D+1,
     5   3.18642D+1,3.18251D+1,3.29153D+1,3.47580D+1,2.53264D+1,
     6   4.03429D+1,2.01922D+1,2.91996D+1,6.24873D+1,8.78242D+0,
     7   9.33480D+0,9.91420D+0,1.71659D+1,5.52077D+1,3.13659D+1,
     8   2.20537D+1,1.42403D+1,1.40442D+1,1.59176D+1,1.43137D+1,
     9   1.46537D+1,1.46455D+1,1.55878D+1,1.69141D+1,1.61552D+1,
     A   1.77931D+1,1.98751D+1,2.41540D+1,3.99955D+1,1.18053D+1,
     B   1.65915D+1,2.23966D+1,2.07637D+1,2.12350D+1,2.18033D+1,
     C   2.39492D+1,2.45984D+1,2.52966D+1,2.60169D+1,2.54973D+1,
     D   2.75466D+1,2.83460D+1,2.91604D+1,2.99904D+1,3.08345D+1,
     E   3.16806D+1,3.13526D+1,3.12166D+1,3.00767D+1,2.86302D+1,
     F   2.75684D+1,2.65861D+1,2.57339D+1,2.29939D+1,2.28644D+1,
     G   2.44080D+1,2.09409D+1,2.29872D+1,2.37917D+1,2.66951D+1,
     H   3.18397D+1,4.34890D+1,2.00150D+1,2.45012D+1,2.56843D+1,
     I   2.65542D+1,2.51930D+1,2.52522D+1,2.54271D+1,2.51526D+1,
     J   2.55959D+1,2.65567D+1,2.70360D+1,2.72673D+1,2.79152D+1,
     K   2.86446D+1,2.93353D+1,3.01040D+1,3.02650D+1/
      DATA BL2/ 2.00272D+0,2.39924D+0,6.62463D-1,9.85154D-1,
     1   1.21347D+0,1.49129D+0,1.76868D+0,2.04035D+0,2.30601D+0,
     2   2.56621D+0,8.71798D-1,1.00247D+0,1.01529D+0,1.17314D+0,
     3   1.34102D+0,2.55169D+0,3.38827D+0,4.56873D+0,3.12426D+0,
     4   1.00935D+0,1.10227D+0,4.12865D+0,3.94043D+0,3.06375D+0,
     5   3.78110D+0,3.77161D+0,3.79085D+0,3.82989D+0,3.39276D+0,
     6   3.94645D+0,3.47325D+0,4.12525D+0,4.95015D+0,1.69671D+0,
     7   1.79002D+0,1.88354D+0,3.11025D+0,4.28418D+0,4.24121D+0,
     8   4.03254D+0,2.97020D+0,2.86107D+0,3.36719D+0,2.73701D+0,
     9   2.71828D+0,2.61549D+0,2.74124D+0,3.08408D+0,2.88189D+0,
     A   3.29372D+0,3.80921D+0,4.61191D+0,5.91318D+0,1.79673D+0,
     B   2.69645D+0,3.45951D+0,3.46574D+0,3.48193D+0,3.50982D+0,
     C   3.51987D+0,3.55603D+0,3.59628D+0,3.63834D+0,3.73639D+0,
     D   3.72882D+0,3.77625D+0,3.82444D+0,3.87344D+0,3.92327D+0,
     E   3.97271D+0,4.09040D+0,4.20492D+0,4.24918D+0,4.24261D+0,
     F   4.23412D+0,4.19992D+0,4.14615D+0,3.73461D+0,3.69138D+0,
     G   3.96429D+0,3.24563D+0,3.62172D+0,3.77959D+0,4.25824D+0,
     H   4.92848D+0,5.85205D+0,2.90906D+0,3.55241D+0,3.79223D+0,
     I   4.00437D+0,3.67795D+0,3.63966D+0,3.61328D+0,3.43021D+0,
     J   3.43474D+0,3.59089D+0,3.59411D+0,3.48061D+0,3.50331D+0,
     K   3.61870D+0,3.55697D+0,3.58685D+0,3.64085D+0/
      DATA BL3/ 1.99732D+0,1.00000D+0,1.00000D+0,1.00000D+0,
     1   1.00000D+0,1.00000D+0,1.00000D+0,1.00000D+0,1.00000D+0,
     2   1.00000D+0,1.00000D+0,1.00000D+0,1.00000D+0,1.00000D+0,
     3   1.00000D+0,1.67534D+0,1.85964D+0,2.04455D+0,7.32637D-1,
     4   1.00000D+0,1.00000D+0,1.00896D+0,1.05333D+0,1.00137D+0,
     5   1.12787D+0,1.16064D+0,1.19152D+0,1.22089D+0,1.14261D+0,
     6   1.27594D+0,1.00643D+0,1.18447D+0,1.35819D+0,1.00000D+0,
     7   1.00000D+0,1.00000D+0,7.17673D-1,8.57842D-1,9.47152D-1,
     8   1.01806D+0,1.01699D+0,1.05906D+0,1.15477D+0,1.10923D+0,
     9   1.12336D+0,1.43183D+0,1.14079D+0,1.26189D+0,9.94156D-1,
     A   1.14781D+0,1.28288D+0,1.41954D+0,1.54707D+0,1.00000D+0,
     B   6.81361D-1,8.07311D-1,8.91057D-1,9.01112D-1,9.10636D-1,
     C   8.48620D-1,8.56929D-1,8.65025D-1,8.73083D-1,9.54998D-1,
     D   8.88981D-1,8.96917D-1,9.04803D-1,9.12768D-1,9.20306D-1,
     E   9.28838D-1,1.00717D+0,1.09456D+0,1.16966D+0,1.23403D+0,
     F   1.29699D+0,1.35350D+0,1.40374D+0,1.44284D+0,1.48856D+0,
     G   1.53432D+0,1.11214D+0,1.23735D+0,1.25338D+0,1.35772D+0,
     H   1.46828D+0,1.57359D+0,7.20714D-1,8.37599D-1,9.33468D-1,
     I   1.02385D+0,9.69895D-1,9.82474D-1,9.92527D-1,9.32751D-1,
     J   9.41671D-1,1.01827D+0,1.02554D+0,9.66447D-1,9.74347D-1,
     K   1.04137D+0,9.90568D-1,9.98878D-1,1.04473D+0/
C
      IIZ=IABS(IZ)
      IF(IIZ.GT.103) IIZ=103
      IF(IIZ.EQ.0) IIZ=1
      A1=B1(IIZ)
      A2=B2(IIZ)
      A3=1.0D0-(A1+A2)
      IF(ABS(A3).LT.1.0D-15) A3=0.0D0
      AL1=BL1(IIZ)
      AL2=BL2(IIZ)
      AL3=BL3(IIZ)
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE MOTTCS
C  *********************************************************************
      SUBROUTINE MOTTSC(IELEC,IZ,EV,IW)
C
C     Mott cross section for elastic scattering of high-energy electrons
C  and positrons by unscreened point nuclei.
C
C  Input parameters:
C    IELEC ..... electron-positron flag;
C                =-1 for electrons,
C                =+1 for positrons.
C    IZ ........ atomic number of the target atom.
C    EV ........ projectile's kinetic energy (in eV).
C    IW ........ output unit (to be defined in the main program).
C
C  The Mott DCS and spin polarization function are printed on unit IW.
C
      IMPLICIT DOUBLE PRECISION (A-B,D-H,O-Z), COMPLEX*16 (C),
     1   INTEGER*4 (I-N)
C
      PARAMETER (SL=137.035999074D0)  ! Speed of light (1/alpha)
      PARAMETER (A0B=5.2917721092D-9)  ! Bohr radius (cm)
      PARAMETER (HREV=27.21138505D0)  ! Hartree energy (eV)
      PARAMETER (A0B2=A0B*A0B)
      PARAMETER (F2BOHR=1.0D-13/A0B)
      PARAMETER (PI=3.1415926535897932D0,FOURPI=4.0D0*PI)
C
      PARAMETER (NGT=650)
      COMMON/DCSTAB/ECS,TCS1,TCS2,TH(NGT),XT(NGT),DCST(NGT),SPOL(NGT),
     1              ERROR(NGT),NTAB
C
      PARAMETER (NPC=1500)
      COMMON/CRMORU/CFM(NPC),CGM(NPC),DPC(NPC),DMC(NPC),
     1              CF,CG,RUTHC,WATSC,RK2,ERRF,ERRG,NPC1
C
      WRITE(IW,1000)
 1000 FORMAT(1X,'#',/1X,'# Subroutine MOTTSC. Elastic scattering of ',
     1  'electrons and positrons',/1X,'#',20X,
     2  'by unscreened Coulomb fields')
      IF(IELEC.EQ.-1) THEN
        WRITE(IW,1100)
 1100   FORMAT(1X,'#',/1X,'# Projectile: electron')
      ELSE
        WRITE(IW,1200)
 1200   FORMAT(1X,'#',/1X,'# Projectile: positron')
      ENDIF
      E=EV/HREV
      WRITE(IW,1300) EV,E
 1300 FORMAT(1X,'# Kinetic energy =',1P,E12.5,' eV =',
     1       E12.5,' a.u.')
C
      IF(IZ.LE.0) STOP 'MOTTCS: Negative atomic number.'
      WRITE(IW,1001) IZ
 1001 FORMAT(1X,'#',/1X,'# Z = ',I3)
C
      Z=DBLE(IZ*IELEC)
      CALL DPWAC0(Z,EV)
C
      TH(1)=0.0D0
      TH(2)=1.0D-4
      I=2
 10   CONTINUE
      I=I+1
      IF(TH(I-1).LT.0.9999D-3) THEN
        TH(I)=TH(I-1)+2.5D-5
      ELSE IF(TH(I-1).LT.0.9999D-2) THEN
        TH(I)=TH(I-1)+2.5D-4
      ELSE IF(TH(I-1).LT.0.9999D-1) THEN
        TH(I)=TH(I-1)+2.5D-3
      ELSE IF(TH(I-1).LT.0.9999D+0) THEN
        TH(I)=TH(I-1)+2.5D-2
      ELSE IF(TH(I-1).LT.0.9999D+1) THEN
        TH(I)=TH(I-1)+1.0D-1
      ELSE IF(TH(I-1).LT.2.4999D+1) THEN
        TH(I)=TH(I-1)+2.5D-1
      ELSE
        TH(I)=TH(I-1)+5.0D-1
      ENDIF
      IF(TH(I).LT.180.0D0) GO TO 10
      NTAB=I
C
      DO I=1,NTAB
        THR=TH(I)*PI/180.0D0
        XT(I)=(1.0D0-COS(THR))/2.0D0
        IF(TH(I).GT.1.0D-5) THEN
          Q2=4.0D0*RK2*XT(I)
          RMR=DPWAC(THR)
          DCST(I)=RUTHC*RMR/Q2**2
C  ****  Spin polarization (Sherman) function.
          CF=CF*A0B
          CG=CG*A0B
          ACF=CDABS(CF)**2
          ACG=CDABS(CG)**2
          DCS=ACF+ACG
          IF(DCS.GT.1.0D-45) THEN
            ERR=2.0D0*(ACF*ERRF+ACG*ERRG)/DCS
          ELSE
            ERR=1.0D0
          ENDIF
          ERROR(I)=ERR
        ELSE
          DCST(I)=1.0D-45
          ERROR(I)=1.0D0
        ENDIF
      ENDDO
C
      WRITE(IW,'(1X,''#'')')
      WRITE(IW,'(1X,''# Differential cross section'',6X,
     1  ''MU=(1-COS(THETA))/2'')')
      WRITE(IW,'(1X,''#'',/1X,''#  THETA'',8X,''MU'',10X,''DCS'',
     1  10X,''DCS'',7X,''McKinl-Fesh'',4X,''error''/1X,''#  (deg)'',
     2  17X,''(cm**2/sr)'',3X,''(a0**2/sr)'',3X,''(a0**2/sr)'',
     3  /1X,''#'',71(''-''))')
C
      BETA2=E*(E+2.0D0*SL**2)/(E+SL**2)**2
      BETA=SQRT(BETA2)
      GAMMA=1.0D0+E/SL**2
      CONS=(0.5D0*Z/(BETA2*GAMMA*SL**2))**2
C
      DO I=1,NTAB
        STH2=SIN(0.5D0*MAX(TH(I),1.0D-5)*PI/180.0D0)
        XSMOT=(CONS/STH2**4)*(1.0D0-BETA2*STH2**2
     1    -PI*Z*(BETA/SL)*STH2*(1.0D0-STH2))
        WRITE(IW,2018) TH(I),XT(I),DCST(I),DCST(I)/A0B2,XSMOT,ERROR(I)
      ENDDO
 2018 FORMAT(1X,1P,E10.3,E13.5,3E13.5,2X,E8.1)
C
      ECS=1.0D35
      TCS1=1.0D35
      TCS2=1.0D35
C
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE HEBORN
C  *********************************************************************
      SUBROUTINE HEBORN(IELEC,IZ,MNUCL,EV,IW)
C
C     Mott-Born cross section for elastic scattering of high-energy
C  electrons and positrons by neutral atoms.
C
C    The DCS is obtained as the product of the Mott DCS for a point
C  nucleus, the Helm uniform-uniform nuclear form factor (with an
C  empirical Coulomb correction) and the high-energy DF screening
C  factor.
C
C  Input parameters:
C    IELEC ..... electron-positron flag;
C                =-1 for electrons,
C                =+1 for positrons.
C    IZ ........ atomic number of the target atom.
C    MNUCL ..... nuclear charge density model.
C                  1 --> point nucleus (P),
C                  2 --> uniform distribution (U),
C                3,4 --> Helm's uniform-uniform distribution (Uu).
C    EV ........ projectile's kinetic energy (in eV).
C    IW ........ output unit (to be defined in the main program).
C
C  Output (through the common block /DCSTAB/):
C     ECS ........ total cross section (cm**2).
C     TCS1 ....... 1st transport cross section (cm**2).
C     TCS2 ....... 2nd transport cross section (cm**2).
C     TH(I) ...... scattering angles (in deg)
C     XT(I) ...... values of (1-COS(TH(I)))/2.
C     DCST(I) .... differential cross section per unit solid angle at
C                  TH(I) (in cm**2/sr).
C     ERROR(I) ... relative uncertainty of the computed DCS values.
C                  Estimated from the convergence of the series.
C     NTAB ....... number of angles in the table.
C
      IMPLICIT DOUBLE PRECISION (A-B,D-H,O-Z), COMPLEX*16 (C),
     1   INTEGER*4 (I-N)
C
      PARAMETER (SL=137.035999074D0)  ! Speed of light (1/alpha)
      PARAMETER (A0B=5.2917721092D-9)  ! Bohr radius (cm)
      PARAMETER (HREV=27.21138505D0)  ! Hartree energy (eV)
      PARAMETER (A0B2=A0B*A0B)
      PARAMETER (F2BOHR=1.0D-13/A0B)
      PARAMETER (PI=3.1415926535897932D0,FOURPI=4.0D0*PI)
C
      PARAMETER (NGT=650)
      COMMON/DCSTAB/ECS,TCS1,TCS2,TH(NGT),XT(NGT),DCST(NGT),SPOL(NGT),
     1              ERROR(NGT),NTAB
      COMMON/CTOTCS/TOTCS,ABCS
      COMMON/CDCSHE/Q2T(NGT),FQ(NGT),U1,U2,NQS,MOM
C
      DIMENSION ELAW(103)
C
      PARAMETER (NPC=1500)
      COMMON/CRMORU/CFM(NPC),CGM(NPC),DPC(NPC),DMC(NPC),
     1              CF,CG,RUTHC,WATSC,RK2,ERRF,ERRG,NPC1
C
      CHARACTER*120 SCFILE,FILE1,CS120,NULL
      CHARACTER*1 LIT10(10),LIT1,LIT2,LIT3
      CHARACTER*2 LSYMBL(103)
      DATA LIT10/'0','1','2','3','4','5','6','7','8','9'/
C
      DATA LSYMBL       /' H','He','Li','Be',' B',' C',' N',' O',
     1    ' F','Ne','Na','Mg','Al','Si',' P',' S','Cl','Ar',' K',
     2    'Ca','Sc','Ti',' V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
     3    'Ga','Ge','As','Se','Br','Kr','Rb','Sr',' Y','Zr','Nb',
     4    'Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te',
     5    ' I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu',
     6    'Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta',' W',
     7    'Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At',
     8    'Rn','Fr','Ra','Ac','Th','Pa',' U','Np','Pu','Am','Cm',
     9    'Bk','Cf','Es','Fm','Md','No','Lr'/
C
      DATA ELAW     /1.007900D0,4.002600D0,6.941000D0,9.012200D0,
     1    1.081100D1,1.201070D1,1.400670D1,1.599940D1,1.899840D1,
     2    2.017970D1,2.298980D1,2.430500D1,2.698150D1,2.808550D1,
     3    3.097380D1,3.206600D1,3.545270D1,3.994800D1,3.909830D1,
     4    4.007800D1,4.495590D1,4.786700D1,5.094150D1,5.199610D1,
     5    5.493800D1,5.584500D1,5.893320D1,5.869340D1,6.354600D1,
     6    6.539000D1,6.972300D1,7.261000D1,7.492160D1,7.896000D1,
     7    7.990400D1,8.380000D1,8.546780D1,8.762000D1,8.890590D1,
     8    9.122400D1,9.290640D1,9.594000D1,9.890630D1,1.010700D2,
     9    1.029055D2,1.064200D2,1.078682D2,1.124110D2,1.148180D2,
     1    1.187100D2,1.217600D2,1.276000D2,1.269045D2,1.312900D2,
     1    1.329054D2,1.373270D2,1.389055D2,1.401160D2,1.409076D2,
     2    1.442400D2,1.449127D2,1.503600D2,1.519640D2,1.572500D2,
     3    1.589253D2,1.625000D2,1.649303D2,1.672600D2,1.689342D2,
     4    1.730400D2,1.749670D2,1.784900D2,1.809479D2,1.838400D2,
     5    1.862070D2,1.902300D2,1.922170D2,1.950780D2,1.969666D2,
     6    2.005900D2,2.043833D2,2.072000D2,2.089804D2,2.089824D2,
     7    2.099871D2,2.220176D2,2.230197D2,2.260254D2,2.270277D2,
     8    2.320381D2,2.310359D2,2.380289D2,2.370482D2,2.440642D2,
     9    2.430614D2,2.470000D2,2.470000D2,2.510000D2,2.520000D2,
     1    2.570000D2,2.580000D2,2.590000D2,2.620000D2/
C
      EXTERNAL DCSHB
C
C  ****  Path to the ELSEPA database
      CHARACTER*100 PATHE
      PATHE='./database/'
C
      WRITE(IW,1000)
 1000 FORMAT(1X,'#',/1X,'# Subroutine HEBORN. Elastic scattering of ',
     1  'electrons and positrons',/1X,'#',20X,
     2  'by neutral atoms')
      IF(IELEC.EQ.-1) THEN
        WRITE(IW,1100)
 1100   FORMAT(1X,'#',/1X,'# Projectile: electron')
      ELSE
        WRITE(IW,1200)
 1200   FORMAT(1X,'#',/1X,'# Projectile: positron')
      ENDIF
      E=EV/HREV
      WRITE(IW,1300) EV,E
 1300 FORMAT(1X,'# Kinetic energy =',1P,E12.5,' eV =',
     1       E12.5,' a.u.')
C
      WRITE(IW,'(1X,''#'',/1X,''#  ***  WARNING: High-energy '',
     1  ''Mott-Born approximation. Neutral atom.'')')
C
      IF(IZ.LE.0) STOP 'HEBORN: Negative atomic number.'
      IF(IZ.GT.103) STOP 'HEBORN: Atomic number larger than 103.'
      AW=ELAW(IZ)
      WRITE(IW,1001) LSYMBL(IZ),IZ,AW
 1001 FORMAT(1X,'#',/1X,'# Element: ',A2,',  Z = ',I3,
     1  ',  atomic weight =',1P,E12.5,' g/mol')
C
      Z=DBLE(IZ*IELEC)
      CALL DPWAC0(Z,EV)
C
C  ****  Read screening function from data files.
      JT=IZ
      J1=JT-10*(JT/10)
      JT=(JT-J1)/10
      J2=JT-10*(JT/10)
      JT=(JT-J2)/10
      J3=JT-10*(JT/10)
      LIT1=LIT10(J1+1)
      LIT2=LIT10(J2+1)
      LIT3=LIT10(J3+1)
      FILE1=PATHE//'z_'//LIT3//LIT2//LIT1//'.dfs'
      NC=120
      CS120=' '
      I=0
      DO J=1,NC
        IF(FILE1(J:J).NE.' ') THEN
          I=I+1
          CS120(I:I)=FILE1(J:J)
        ENDIF
      ENDDO
      SCFILE=CS120
      WRITE(6,'(A)') SCFILE
      OPEN(99,FILE=SCFILE,STATUS='OLD',ERR=4)
      READ(99,'(1X,A1)') NULL
      READ(99,'(1X,A1)') NULL
      READ(99,'(1X,A1)') NULL
      NQS=0
      DO I=1,NGT
        READ(99,*,END=4) Q2T(I),FQ(I)
        NQS=I
      ENDDO
 4    CONTINUE
      CLOSE(UNIT=99)
      IF(NQS.EQ.0) THEN
        WRITE(IW,*) 'HEBORN: I/O error. SCFILE does not exist.'
        STOP 'HEBORN: I/O error. SCFILE does not exist.'
      ENDIF
C
C  ****  Nuclear charge density parameters.
C
      IF(MNUCL.EQ.1) THEN
C  ****  Point nucleus.
        WRITE(IW,1002)
 1002   FORMAT(1X,'#',/1X,'# Nuclear model: point charge')
        U1=0.0D0
        U2=0.0D0
      ELSE IF (MNUCL.EQ.2) THEN
C  ****  Uniform distribution..
        R1=1.07D0*F2BOHR*AW**0.3333333333333333D0
        R2=2.00D0*F2BOHR
        R1=R1*SQRT((1.0D0+2.5D0*(R2/R1)**2)
     1             /(1.0D0+0.75D0*(R2/R1)**2))
        WRITE(IW,1004) R1*A0B
 1004   FORMAT(1X,'#',/1X,'# Nuclear model: uniform spherical d',
     1    'istribution',/1X,'#',16X,'Nuclear radius =',1P,E12.5,
     2    ' cm')
        U1=R1**2
        U2=0.0D0
      ELSE IF(MNUCL.EQ.4.OR.MNUCL.EQ.3) THEN
C  ****  Helm's Uu distribution.
        RNUC=1.070D0*F2BOHR*AW**0.3333333333333333D0
        R1=0.962D0*RNUC+0.435D0*F2BOHR
        R2=2.0D0*F2BOHR
        IF(R2.GT.R1) THEN
          STORE=R1
          R1=R2
          R2=STORE
        ENDIF
        WRITE(IW,1105) R1*A0B,R2*A0B
 1105   FORMAT(1X,'#',/1X,'# Nuclear model: Helm''s Uu distribu',
     1    'tion',/1X,'#',16X,'  Inner radius =',1P,E12.5, ' cm',
     2    /1X,'#',16X,'Skin thickness =',E12.5,' cm')
        U1=R1**2
        U2=R2**2
      ELSE
        WRITE(IW,1003)
 1003   FORMAT(1X,'#',/1X,'# Undefined nuclear charge density model.',
     1    /1X,'# The calculation was aborted by subroutine HEBORN.')
        STOP 'HEBORN: Undefined nuclear charge density model.'
      ENDIF
C
      TH(1)=0.0D0
      TH(2)=1.0D-4
      I=2
 10   CONTINUE
      I=I+1
      IF(TH(I-1).LT.0.9999D-3) THEN
        TH(I)=TH(I-1)+2.5D-5
      ELSE IF(TH(I-1).LT.0.9999D-2) THEN
        TH(I)=TH(I-1)+2.5D-4
      ELSE IF(TH(I-1).LT.0.9999D-1) THEN
        TH(I)=TH(I-1)+2.5D-3
      ELSE IF(TH(I-1).LT.0.9999D+0) THEN
        TH(I)=TH(I-1)+2.5D-2
      ELSE IF(TH(I-1).LT.0.9999D+1) THEN
        TH(I)=TH(I-1)+1.0D-1
      ELSE IF(TH(I-1).LT.2.4999D+1) THEN
        TH(I)=TH(I-1)+2.5D-1
      ELSE
        TH(I)=TH(I-1)+5.0D-1
      ENDIF
      IF(TH(I).LT.180.0D0) GO TO 10
      NTAB=I
C
      DO I=1,NTAB
        THR=TH(I)*PI/180.0D0
        XT(I)=(1.0D0-COS(THR))/2.0D0
C  ****  Screening correction.
        Q2=4.0D0*RK2*XT(I)
        IF(Q2.LT.Q2T(NQS)) THEN
          CALL FINDI(Q2,Q2T,NQS,J)
          F=FQ(J)+(FQ(J+1)-FQ(J))*(Q2-Q2T(J))/(Q2T(J+1)-Q2T(J))
        ELSE
          F=1.0D0
        ENDIF
C  ****  Nuclear form factor.
        QR2=Q2*U1
        QR=SQRT(QR2)
        IF(QR2.LT.1.0D-8) THEN
          FR=1.0D0+QR2*(-0.1D0+QR2*3.5714285714285714D-3)
        ELSE
          FR=3.0D0*(SIN(QR)-QR*COS(QR))/(QR*QR2)
        ENDIF
        QU2=Q2*U2
        QU=SQRT(QU2)
        IF(QU2.LT.1.0D-8) THEN
          FU=1.0D0+QU2*(-0.1D0+QU2*3.5714285714285714D-3)
        ELSE
          FU=3.0D0*(SIN(QU)-QU*COS(QU))/(QU*QU2)
        ENDIF
        FN=FR*FU
C
        RMR=DPWAC(THR)
        DCST(I)=RUTHC*(F*FN)**2*RMR/(1.0D0+Q2)**2
      ENDDO
C
C  ****  Integrated cross sections.
C
      SUM0=0.0D0
      SUM1=0.0D0
      SUM2=0.0D0
      RMUL=0.0D0
      RMUU=1.0D-16
 20   CONTINUE
      TOL=1.0D-9
      MOM=0
      SUMP0=SUMGA(DCSHB,RMUL,RMUU,TOL)
      MOM=1
      SUMP1=SUMGA(DCSHB,RMUL,RMUU,TOL)
      MOM=2
      SUMP2=SUMGA(DCSHB,RMUL,RMUU,TOL)
      SUM0=SUM0+SUMP0
      SUM1=SUM1+SUMP1
      SUM2=SUM2+SUMP2
      RMUL=RMUU
      RMUU=MIN(2.0D0*RMUL,1.0D0)
      IF(RMUL.LT.0.9999999D0) GO TO 20
      ECS0=FOURPI*SUM0
      ECS1=FOURPI*SUM1
      ECS2=FOURPI*SUM2
C
      ECS=ECS0
      TCS1=2.0D0*ECS1
      TCS2=6.0D0*(ECS1-ECS2)
      WRITE(IW,2013) ECS,ECS/A0B2
 2013 FORMAT(1X,'#',/1X,'# Total elastic cross section =',1P,
     1     E12.5,' cm**2 =',E12.5,' a0**2')
      WRITE(IW,2014) TCS1,TCS1/A0B2
 2014 FORMAT(1X,'# 1st transport cross section =',1P,
     1     E12.5,' cm**2 =',E12.5,' a0**2')
      WRITE(IW,2015) TCS2,TCS2/A0B2
 2015 FORMAT(1X,'# 2nd transport cross section =',1P,
     1     E12.5,' cm**2 =',E12.5,' a0**2',/1X,'#')
C
      WRITE(IW,'(1X,''#'')')
      WRITE(IW,'(1X,''# Differential cross section'',6X,
     1  ''MU=(1-COS(THETA))/2'')')
      WRITE(IW,'(1X,''#'',/1X,''#  THETA'',8X,''MU'',10X,''DCS'',
     1  10X,''DCS'',8X,''Sherman'',7X,''error''/1X,''#  (deg)'',
     2  17X,''(cm**2/sr)'',3X,''(a0**2/sr)'',4X,''function'',
     3  /1X,''#'',71(''-''))')
      DO I=1,NTAB
        SPOL(I)=0.0D0
        ERROR(I)=1.0D-5
        WRITE(IW,2018) TH(I),XT(I),DCST(I),DCST(I)/A0B2,SPOL(I),ERROR(I)
      ENDDO
 2018 FORMAT(1X,1P,E10.3,E13.5,3E13.5,2X,E8.1)
C
      RETURN
      END
C  *********************************************************************
      FUNCTION DCSHB(RMU)
C     Mott-Born DCS for elastic scattering of high-energy electrons and
C  positrons by neutral atoms.
      IMPLICIT DOUBLE PRECISION (A-B,D-H,O-Z), COMPLEX*16 (C),
     1   INTEGER*4 (I-N)
      PARAMETER (NPC=1500)
      COMMON/CRMORU/CFM(NPC),CGM(NPC),DPC(NPC),DMC(NPC),
     1              CF,CG,RUTHC,WATSC,RK2,ERRF,ERRG,NPC1
      PARAMETER (NGT=650)
      COMMON/CDCSHE/Q2T(NGT),FQ(NGT),U1,U2,NQS,MOM
C  ****  Screening correction.
      Q2=4.0D0*RK2*RMU
      IF(Q2.LT.Q2T(NQS)) THEN
        CALL FINDI(Q2,Q2T,NQS,J)
        F=FQ(J)+(FQ(J+1)-FQ(J))*(Q2-Q2T(J))/(Q2T(J+1)-Q2T(J))
      ELSE
        F=1.0D0
      ENDIF
C  ****  (nuclear form factor)**2.
      QR2=Q2*U1
      QR=SQRT(QR2)
      IF(QR2.LT.1.0D-8) THEN
        FR=1.0D0+QR2*(-0.1D0+QR2*3.5714285714285714D-3)
      ELSE
        FR=3.0D0*(SIN(QR)-QR*COS(QR))/(QR*QR2)
      ENDIF
      QU2=Q2*U2
      QU=SQRT(QU2)
      IF(QU2.LT.1.0D-8) THEN
        FU=1.0D0+QU2*(-0.1D0+QU2*3.5714285714285714D-3)
      ELSE
        FU=3.0D0*(SIN(QU)-QU*COS(QU))/(QU*QU2)
      ENDIF
      FN=FR*FU
C
      RMR=DPWAC(ACOS(1.0D0-2.0D0*RMU))
      DCSHB=(RUTHC*(F*FN)**2*RMR/(1.0D0+Q2)**2)*RMU**MOM
      RETURN
      END
C  *********************************************************************
C                       FUNCTION VCPOL
C  *********************************************************************
      FUNCTION VCPOL(IELEC,DEN)
C
C     This function gives the correlation potential of an electron
C  (IELEC=-1) or positron (IELEC=+1) in an homogeneous electron gas of
C  density DEN (electrons per unit volume).
C
C  ****  All quantities are in atomic units.
C
      IMPLICIT REAL*8 (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER (PI=3.1415926535897932D0,FOURPI=4.0D0*PI)
C
      IF(DEN.LT.1.0D-12) THEN
        VCPOL=0.0D0
        RETURN
      ENDIF
      RS=(3.0D0/(FOURPI*DEN))**3.333333333333D-1
      RSL=LOG(RS)
C
      IF(IELEC.EQ.-1) THEN
C  ****  Electron exchange-correlation potential.
C        Ref:  Padial and Norcross, Phys. Rev. A 29(1984)1742.
C              Perdew and Zunger, Phys. Rev. B 23(1981)5048.
        IF(RS.LT.1.0D0) THEN
          VCPOL=0.0311D0*RSL-0.0584D0+0.00133D0*RS*RSL-0.0084D0*RS
        ELSE
          GAM=-0.1423D0
          BET1=1.0529D0
          BET2=0.3334D0
          RSS=SQRT(RS)
          VCPOL=GAM*(1.0D0+(7.0D0/6.0D0)*BET1*RSS
     1         +(4.0D0/3.0D0)*BET2*RS)/(1.0D0+BET1*RSS+BET2*RS)**2
        ENDIF
      ELSE
C  ****  Positron correlation potential.
C        Ref:  Jain, Phys. Rev. A 41(1990)2437.
        IF(RS.LT.0.302D0) THEN
          VCPOL=(-1.82D0/SQRT(RS))+(0.051D0*RSL-0.115D0)*RSL+1.167D0
        ELSE IF(RS.LT.0.56D0) THEN
          VCPOL=-0.92305D0-0.09098D0/RS**2
        ELSE IF(RS.LT.8.0D0) THEN
          RSD=1.0D0/(RS+2.5D0)
          VCPOL=(-8.7674D0*RS*RSD**3)+(-13.151D0+0.9552D0*RS)*RSD**2
     1         +2.8655D0*RSD-0.6298D0
        ELSE
          VCPOL=-179856.2768D0*3.0D0*DEN**2+186.4207D0*2.0D0*DEN
     1         -0.524D0
        ENDIF
        VCPOL=0.5D0*VCPOL
      ENDIF
      RETURN
      END
C  *********************************************************************
C                        SUBROUTINE XSFEG
C  *********************************************************************
      SUBROUTINE XSFEG(DEN,DELTA,IELEC,EK,MORD,XSEC,IMODE)
C
C     This subroutine computes restricted (W>DELTA) total cross sections
C  for interactions of electrons (IELEC=-1) or positrons (IELEC=+1) with
C  a degenerate free electron gas (per electron in the gas). The DCS
C  is obtained from Lindhard's dielectric function (i.e. within the
C  first Born approximation), with the Ochkur exchange correction for
C  electrons.
C
C  Ref.: F. Salvat, Phys. Rev. A 68 (2003) 012708.
C
C
C  Input arguments:
C     DEN ...... density of the electron gas (electrons per unit
C                volume).
C     DELTA .... energy gap (or minimum energy loss).
C     IELEC .... kind of projectile.
C                =-1, electron; =+1, positron.
C     EK ....... kinetic energy of the projectile.
C     MORD ..... order of the calculated cross section;
C                =0, total cross section,
C                =1, stopping cross section,
C                =2, energy straggling cross section.
C     XSEC ..... total integrated cross section.
C     IMODE .... =1, the complete DCS (for electron-hole and plasmon
C                    excitations) is calculated.
C                =2, the output value XSEC corresponds to electron-hole
C                    excitations (binary collisions) only.
C
C                                  (All quantities in atomic units).
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER(F2O3=2.0D0/3.0D0,F3O16=3.0D0/16.0D0)
      PARAMETER(PI=3.1415926535897932D0, FOURPI=4.0D0*PI)
      COMMON/CXSFEG/EF,EP,CHI2,XP,ZC,XC,XE,SXE,IELPO
      PARAMETER(NPM=50)
      COMMON/CXSPL0/Z
      COMMON/CXSPL1/ZPT(NPM),XPT(NPM),FPL(NPM),ZANAL,XANAL,
     1              AX(NPM),BX(NPM),CX(NPM),DX(NPM),
     2              AF(NPM),BF(NPM),CF(NPM),DF(NPM),MOM
C  ****  Contributions from electron-hole and plasmon excitations.
      COMMON/XSFEGO/XSEH,XSPL
C
      COMMON/CSUMGA/ERR,IERGA,NCALL  ! Error, code, function calls.
      EXTERNAL PLSTR
C  ****  Plasmon excitation functions are printed if IWR=1.
      IWR=0
      TOL=1.0D-6
C
      IF(MORD.LT.0.OR.MORD.GT.2) THEN
        STOP 'XSFEG: Wrong MORD value.'
      ENDIF
C
C  ****  Constants and energy-independent parameters.
C
      DENM=MAX(DEN,1.0D-5)
      IELPO=IELEC
      EP2=FOURPI*DENM
      EP=SQRT(EP2)
      EF=0.5D0*(3.0D0*PI**2*DENM)**F2O3
      XE=EK/EF
      IF(XE.LT.1.0D-3) THEN
        XSEC=0.0D0
        RETURN
      ENDIF
      SXE=SQRT(XE)
      XP=EP/EF
      CHI2=F3O16*XP*XP
C  ****  Plasmon cutoff momentum.
        ZL=XP-1.0D-3
 1      CONTINUE
        FL=ZL*ZL+CHI2*F1(ZL,4.0D0*ZL*(ZL+1.0D0))
        IF(FL.GT.0.0D0) THEN
          ZL=0.5D0*ZL
          GO TO 1
        ENDIF
        ZU=XP+1.0D-2
 2      CONTINUE
        FU=ZU*ZU+CHI2*F1(ZU,4.0D0*ZU*(ZU+1.0D0))
        IF(FU.LT.0.0D0) THEN
          ZU=ZU+ZU
          GO TO 2
        ENDIF
 3      ZC=0.5D0*(ZL+ZU)
        FT=ZC*ZC+CHI2*F1(ZC,4.0D0*ZC*(ZC+1.0D0))
        IF(FT.LT.0.0D0) THEN
          ZL=ZC
        ELSE
          ZU=ZC
        ENDIF
        IF(ABS(ZL-ZU).GT.1.0D-15*ZC) GO TO 3
        XC=4.0D0*ZC*(ZC+1.0D0)
C
C  ************  Electron-hole contribution.
C
      CALL SEH0(XSEH,DELTA,MORD,IWR)
C
      IF(IMODE.EQ.2) THEN
        XSEC=XSEH
        RETURN
      ENDIF
C
C  ************  Plasmon contribution.
C
      IF(XE.LT.XP) THEN
        XSPL=0.0D0
      ELSE
C  ****  Plasmon line.
        ZPT(1)=0.0D0
        XPT(1)=XP
        FPL(1)=1.0D0
        ZANAL=0.0D0
        XANAL=0.0D0
C  ****  Varying step: 2*DZ for I<NPH.
        NPH=2*NPM/3
        DFZ=0.999999999D0*ZC/DBLE(NPM+NPH-3)
        DO I=2,NPM
          IF(I.LT.NPH) THEN
            Z=ZPT(I-1)+DFZ*2.0D0
          ELSE
            Z=ZPT(I-1)+DFZ
          ENDIF
          IF(Z.GT.0.02D0*ZC) THEN
C  The starting endpoints must be outside the Lindhard continuum.
            XL=MAX(4.0D0*Z*(Z+1.0D0)+1.0D-9,0.9D0*XP)
            XU=1.1D0*XC
 4          X=0.5D0*(XL+XU)
            FT=Z*Z+CHI2*F1(Z,X)
            IF(FT.GT.0.0D0) THEN
              XU=X
            ELSE
              XL=X
            ENDIF
C           WRITE(6,'('' X,FT ='',1P,3E18.11)') X,FT
            IF(FT.GT.1.0D-6) GO TO 4
            IF(ABS(XL-XU).GT.1.0D-13*X) GO TO 4
          ELSE
            X=SQRT(XP**2+(48.0D0/5.0D0)*Z**2+16.0D0*Z**4)
          ENDIF
          XPT(I)=X
          ZPT(I)=Z
        ENDDO
        DO I=2,NPM-1
          Z=ZPT(I)
          XUP=4.0D0*Z*(Z+1.0D0)-1.0D-9
          SUM=SUMGA(PLSTR,1.0D-10,XUP,TOL)
          IF(IERGA.EQ.1) THEN
            WRITE(6,*) 'SUMGA error in XSFEG.'
            STOP 'XSFEG: SUMGA error (1).'
          ENDIF
          FPL(I)=1.0D0-SUM*(6.0D0/(16.0D0*PI))
          XAP=XP+(24.0D0/5.0D0)*Z*Z/XP
          IF(ABS(XAP-XPT(I)).LT.1.0D-3*XPT(I).AND.
     1      FPL(I).GT.0.999D0) THEN
            ZANAL=ZPT(I)
            XANAL=XPT(I)
          ENDIF
        ENDDO
        FPL(NPM)=FPL(NPM-1)+(FPL(NPM-1)-FPL(NPM-2))
     1      *(XPT(NPM)-XPT(NPM-1))/(XPT(NPM-1)-XPT(NPM-2))
C
        IF(IWR.EQ.1) THEN
          OPEN(99,FILE='plasma.dat')
          Z=1.1D0*ZC
          XLOW=MAX(0.0D0,4.0D0*Z*(Z-1.0D0))+1.0D-9
          XUP=4.0D0*Z*(Z+1.0D0)-1.0D-9
          SUM=SUMGA(PLSTR,XLOW,XUP,TOL)
          IF(IERGA.EQ.1) THEN
            WRITE(6,*) 'SUMGA error in XSFEG.'
            STOP 'XSFEG: SUMGA error (2).'
          ENDIF
          BETHE=SUM*(6.0D0/(16.0D0*PI))
          WRITE(99,*) '#  BETHE SUM =',BETHE
          WRITE(99,*) '#  AN. APPROX. VALID FOR Z <',ZANAL
          WRITE(99,*) '#  AN. APPROX. VALID FOR X <',XANAL
          DO I=1,NPM
            Z=ZPT(I)
            XAP=XP+(24.0D0/5.0D0)*Z*Z/XP
            WRITE(99,'(I4,1P,5E14.6)') I,ZPT(I),XPT(I),FPL(I),XAP
          ENDDO
          CLOSE(99)
        ENDIF
C
        CALL SPLINE(ZPT,XPT,AX,BX,CX,DX,0.0D0,0.0D0,NPM)
        CALL SPLINE(ZPT,FPL,AF,BF,CF,DF,0.0D0,0.0D0,NPM)
        CALL SPL0(XSPL,DELTA,MORD)
      ENDIF
C
      XSEC=XSEH+XSPL
      RETURN
      END
C  *********************************************************************
C                       FUNCTION PLSTR
C  *********************************************************************
      FUNCTION PLSTR(X)
C
C     Integrand of the DDCS for a point (Z,X) within the Lindhard
C  continuum.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER(PI=3.1415926535897932D0)
      COMMON/CXSFEG/EF,EP,CHI2,XP,ZC,XC,XE,SXE,IELPO
      COMMON/CXSPL0/Z
C
      PLSTR=0.0D0
      IF(Z.LT.1.0D-8) RETURN
C  ****  F2 function.
      IF(X.LT.1.0D0) THEN
        IF(X.LT.4.0D0*Z*(1.0D0-Z)) THEN
          F2=PI*X*0.125D0/Z  ! Region a.
        ELSE
          ZIN=1.0D0/Z
          ZM=Z-X*ZIN*0.25D0
          F2=PI*0.125D0*ZIN*(1.0D0-ZM*ZM)  ! Region b.
        ENDIF
      ELSE
        ZIN=1.0D0/Z
        ZM=Z-X*ZIN*0.25D0
        F2=PI*0.125D0*ZIN*(1.0D0-ZM*ZM)  ! Region b.
      ENDIF
C
      PLSTR=X*Z*Z*F2/((Z*Z+CHI2*F1(Z,X))**2+(CHI2*F2)**2)
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE SPL0
C  *********************************************************************
      SUBROUTINE SPL0(XSPL,DELTA,MORD)
C
C     Restricted total cross sections for plasmon excitation.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER(PI=3.1415926535897932D0)
      PARAMETER(NPM=50)
      COMMON/CXSFEG/EF,EP,CHI2,XP,ZC,XC,XE,SXE,IELPO
      COMMON/CXSPL1/ZPT(NPM),XPT(NPM),FPL(NPM),ZANAL,XANAL,
     1              AX(NPM),BX(NPM),CX(NPM),DX(NPM),
     2              AF(NPM),BF(NPM),CF(NPM),DF(NPM),MOM
      COMMON/CSUMGA/ERR,IERGA,NCALL  ! Error, code, function calls.
      EXTERNAL SPL1
C
      TOL=1.0D-6
      XSPL=0.0D0
      IF(XE.LT.XP+1.0D-8) THEN
        WRITE(6,*) 'WARNING: X is less than XP (SPL0).'
        RETURN
      ENDIF
C  ****  Minimum and maximum allowed Z-values.
      I1=0
      IN=NPM
      DO I=2,NPM
        IF(IELPO.EQ.-1) THEN
          XUP=MIN(4.0D0*ZPT(I)*(SXE-ZPT(I)),XE-1.0D0)
        ELSE
          XUP=4.0D0*ZPT(I)*(SXE-ZPT(I))
        ENDIF
        IF(XUP.GT.XPT(I)) THEN
          IF(I1.EQ.0) I1=I
        ENDIF
        IF(XUP.LT.XPT(I).AND.I1.GT.0) THEN
          IN=I
          GO TO 1
        ENDIF
      ENDDO
 1    CONTINUE
      IF(I1.EQ.0) RETURN
C
      I=I1-1
      ZL=ZPT(I)
      ZU=ZPT(I+1)
 2    Z=0.5D0*(ZL+ZU)
      X=AX(I)+Z*(BX(I)+Z*(CX(I)+Z*DX(I)))
      IF(IELPO.EQ.-1) THEN
        XMIN=MIN(4.0D0*Z*(SXE-Z),XE-1.0D0)
      ELSE
        XMIN=4.0D0*Z*(SXE-Z)
      ENDIF
      IF(XMIN.GT.X) THEN
        ZU=Z
      ELSE
        ZL=Z
      ENDIF
C       WRITE(6,'('' Z1,X-XCON ='',1P,3E18.11)') Z,X-XMIN
      IF(ABS(ZU-ZL).GT.1.0D-14*Z) GO TO 2
      ZMIN=Z
C
      IF(IN.LT.NPM) THEN
        I=IN-1
        ZL=ZPT(I)
        ZU=ZPT(I+1)
 3      Z=0.5D0*(ZL+ZU)
        X=AX(I)+Z*(BX(I)+Z*(CX(I)+Z*DX(I)))
        IF(IELPO.EQ.-1) THEN
          XMAX=MIN(4.0D0*Z*(SXE-Z),XE-1.0D0)
        ELSE
          XMAX=4.0D0*Z*(SXE-Z)
        ENDIF
        IF(XMAX.LT.X) THEN
          ZU=Z
        ELSE
          ZL=Z
        ENDIF
C         WRITE(6,'('' Z2,X-XCON ='',1P,3E18.11)') Z,X-XMAX
        IF(ABS(ZU-ZL).GT.1.0D-14*Z) GO TO 3
        ZMAX=Z
      ELSE
        XMAX=XC
        ZMAX=ZC
      ENDIF
C
      XDEL=DELTA/EF
      IF(XDEL.GT.XMAX) RETURN
      IF(XDEL.GT.XMIN) THEN
        CALL FINDI(XDEL,XPT,NPM,I)
        ZL=ZPT(I)
        ZU=ZPT(I+1)
 4      Z=0.5D0*(ZL+ZU)
        X=AX(I)+Z*(BX(I)+Z*(CX(I)+Z*DX(I)))
        IF(XDEL.LT.X) THEN
          ZU=Z
        ELSE
          ZL=Z
        ENDIF
C         WRITE(6,'('' Z1,X-XCON ='',1P,3E18.11)') Z,X-XDEL
        IF(ABS(ZU-ZL).GT.1.0D-14*Z) GO TO 4
        ZMIN=Z
        XMIN=XDEL
      ENDIF
C
      IF(XMIN.GT.XMAX) RETURN
C
C  ****  Soft plasmon excitation.
C
      FACT= 3.0D0/(16.0D0*CHI2)
      SUMP=0.0D0
      IF(XMIN.LT.XANAL.AND.XMAX.GT.XANAL) THEN
        IF(MORD.EQ.0) THEN
          X=XANAL
          S0U=X+(XP/2.0D0)*LOG((X-XP)/(X+XP))
          X=XMIN
          S0L=X+(XP/2.0D0)*LOG((X-XP)/(X+XP))
          SUMP=FACT*(S0U-S0L)
          ZMIN=ZANAL
        ELSE IF(MORD.EQ.1) THEN
          X=XANAL
          S1U=(X**2/2.0D0)+(XP**2/2.0D0)*LOG(X*X-XP*XP)
          X=XMIN
          S1L=(X**2/2.0D0)+(XP**2/2.0D0)*LOG(X*X-XP*XP)
          SUMP=FACT*(S1U-S1L)
          ZMIN=ZANAL
        ELSE IF(MORD.EQ.2) THEN
          X=XANAL
          S2U=(X**3/3.0D0)+XP**2*X+(XP**3/2.0D0)
     1       *LOG((X-XP)/(X+XP))
          X=XMIN
          S2L=(X**3/3.0D0)+XP**2*X+(XP**3/2.0D0)
     1       *LOG((X-XP)/(X+XP))
          SUMP=FACT*(S2U-S2L)
          ZMIN=ZANAL
        ELSE
          STOP 'SPL0: Wrong MORD value.'
        ENDIF
      ENDIF
C
      IF(ZMIN.LT.ZMAX) THEN
        MOM=MORD
        SUM=SUMGA(SPL1,ZMIN,ZMAX,TOL)
        IF(IERGA.NE.0) THEN
          OPEN(99,FILE='plasma.dat')
          DO I=1,NPM
            WRITE(99,'(I4,1P,5E14.6)') I,XPT(I),ZPT(I),FPL(I)
          ENDDO
          CLOSE(99)
          WRITE(6,*) 'Accumulated numerical errors...'
          WRITE(6,*) 'SUMGA error in SPL0.'
          STOP 'SPL0: SUMGA error.'
        ENDIF
      ELSE
        SUM=0.0D0
      ENDIF
      XSPL=(2.0D0*PI/(XE*EF*EF))*(SUM+SUMP)*EF**MORD
      RETURN
      END
C  *********************************************************************
C                       FUNCTION SPL1
C  *********************************************************************
      FUNCTION SPL1(Z)
C
C     DCS for plasmon excitations.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER(R96O5=96.0D0/5.0D0,R3O4=3.0D0/4.0D0)
      COMMON/CXSFEG/EF,EP,CHI2,XP,ZC,XC,XE,SXE,IELPO
      PARAMETER(NPM=50)
      COMMON/CXSPL1/ZPT(NPM),XPT(NPM),FPL(NPM),ZANAL,XANAL,
     1              AX(NPM),BX(NPM),CX(NPM),DX(NPM),
     2              AF(NPM),BF(NPM),CF(NPM),DF(NPM),MOM
C
      CALL FINDI(Z,ZPT,NPM,I)
      X=AX(I)+Z*(BX(I)+Z*(CX(I)+Z*DX(I)))
      FP=AF(I)+Z*(BF(I)+Z*(CF(I)+Z*DF(I)))
      SPL1=FP*X**MOM/(Z*X)
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE SEH0
C  *********************************************************************
      SUBROUTINE SEH0(XSEH,DELTA,MORD,IWR)
C
C  Restricted total cross sections for electron-hole excitations.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER(PI=3.1415926535897932D0)
      PARAMETER(NHM=150)
      DIMENSION XT(NHM),DW(NHM)
      COMMON/CXSFEG/EF,EP,CHI2,XP,ZC,XC,XE,SXE,IELPO
C
      IF(IELPO.EQ.-1) THEN
        XMAX=0.5D0*(XE-1.0D0)
C       XMAX=XE-1.0D-8  !!!! NO EXCHANGE !!!!
      ELSE
        XMAX=XE-1.0D-8
      ENDIF
      XMIN=MAX(DELTA/EF,1.0D-10)
      IF(XMIN.GT.XMAX) THEN
        XSEH=0.0D0
        RETURN
      ENDIF
C
      FACTL=6.0D0/(16.0D0*PI)
      FACTR=0.5D0
      NP=1
      XT(1)=XMIN
      DW(1)=SEH1(XT(1))
      IF(XMIN.LT.1.2D0*XC) THEN
        NS1=2*NHM/3
        DX=(MIN(1.2D0*XC,XMAX)-XMIN)/DBLE(NS1-1)
        DO I=2,NS1
          NP=I
          XT(I)=XT(I-1)+DX
          DW(I)=FACTL*SEH1(XT(I))
        ENDDO
      ENDIF
      IF(XT(NP).LT.XMAX-1.0D-10) THEN
        DFX=EXP(LOG((XMAX)/XT(NP))/DBLE(NHM-NP))
        NP1=NP+1
        ICALC=0
        DO I=NP1,NHM
          NP=I
          XT(I)=XT(I-1)*DFX
          IF(ICALC.EQ.0) THEN
            DW(I)=FACTL*SEH1(XT(I))
            DWA=FACTR/XT(I)**2
            IF(IELPO.EQ.-1) THEN  ! Exchange correction.
              FEXP=XT(I)/(XE-XT(I))
C             FEXP=1.0D0  !!!! NO EXCHANGE !!!!
              DWA=DWA*(1.0D0-FEXP*(1.0D0-FEXP))
            ENDIF
            IF(ABS(DW(I)-DWA).LT.1.0D-4*DWA) ICALC=1
          ELSE
C  ****  High-Z electron-hole excitations. Moller or Rutherford
C        differential cross section.
            DW(I)=FACTR/XT(I)**2
            IF(IELPO.EQ.-1) THEN  ! Exchange correction.
              FEXP=XT(I)/(XE-XT(I))
C             FEXP=1.0D0  !!!! NO EXCHANGE !!!!
              DW(I)=DW(I)*(1.0D0-FEXP*(1.0D0-FEXP))
            ENDIF
          ENDIF
        ENDDO
      ENDIF
      IF(NP.LT.3) THEN
        XSEH=0.0D0
        WRITE(6,*) 'WARNING: NP is too small (SEH0).'
        RETURN
      ENDIF
      DW(NP)=EXP(LOG(DW(NP-1))+LOG(DW(NP-1)/DW(NP-2))
     1      *(XT(NP)-XT(NP-1))/(XT(NP-1)-XT(NP-2)))
C
      IF(IWR.EQ.1) THEN
        OPEN(99,FILE='ehdcs.dat')
        DO I=1,NP
          WRITE(99,'(1X,1P,5E14.6)') XT(I)/XC,DW(I)
        ENDDO
        CLOSE(99)
      ENDIF
C
      FACT=2.0D0*PI/(XE*EF*EF)
      XSEH=FACT*SMOMLL(XT,DW,XT(1),XT(NP),NP,MORD,0)*EF**MORD
      RETURN
      END
C  *********************************************************************
C                       FUNCTION SEH1
C  *********************************************************************
      FUNCTION SEH1(X)
C
C     Integral of the DDCS over Z within the Lindhard continuum.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      COMMON/CXSFEG/EF,EP,CHI2,XP,ZC,XC,XE,SXE,IELPO
      COMMON/CXSEH1/XX
      COMMON/CSUMGA/ERR,IERGA,NCALL  ! Error, code, function calls.
      EXTERNAL SEH2
C
      TOL=1.0D-6
      SEH1=0.0D0
      SXP1=SQRT(X+1.0D0)
      SXEX=SQRT(XE-X)
      ZMIN=MAX(0.5D0*(SXE-SXEX),0.5D0*(SXP1-1.0D0))+1.0D-10
      ZMAX=MIN(0.5D0*(SXE+SXEX),0.5D0*(SXP1+1.0D0))-1.0D-10
      IF(ZMIN.GT.ZMAX) RETURN
C
      XX=X
      IF(ABS(X-XC).LT.2.0D-2*XC) THEN
        ZMINM=ZMIN+1.0D-7*(ZMAX-ZMIN)
        SUM=SUMGA(SEH2,ZMINM,ZMAX,TOL)
        IF(IERGA.EQ.1) THEN
          WRITE(6,*) 'SUMGA error in SEH1 (1).'
          STOP 'SUMGA error in SEH1 (1).'
        ENDIF
        SEH1=SUM
      ELSE
        SUM=SUMGA(SEH2,ZMIN,ZMAX,TOL)
        IF(IERGA.EQ.1) THEN
          WRITE(6,*) 'SUMGA error in SEH1 (2).'
          STOP 'SUMGA error in SEH1 (2).'
        ENDIF
        SEH1=SUM
      ENDIF
      RETURN
      END
C  *********************************************************************
C                       FUNCTION SEH2
C  *********************************************************************
      FUNCTION SEH2(Z)
C
C     Integrand of the DDCS for a point (Z,X) within the Lindhard
C  continuum.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      PARAMETER(PI=3.1415926535897932D0)
      COMMON/CXSFEG/EF,EP,CHI2,XP,ZC,XC,XE,SXE,IELPO
      COMMON/CXSEH1/X
C
      SEH2=0.0D0
      IF(Z.LT.1.0D-8) RETURN
C  ****  F2 function.
      IF(X.LT.1.0D0) THEN
        IF(X.LT.4.0D0*Z*(1.0D0-Z)) THEN
          F2=PI*X*0.125D0/Z  ! Region a.
        ELSE
          ZIN=1.0D0/Z
          ZM=Z-X*ZIN*0.25D0
          F2=PI*0.125D0*ZIN*(1.0D0-ZM*ZM)  ! Region b.
        ENDIF
      ELSE
        ZIN=1.0D0/Z
        ZM=Z-X*ZIN*0.25D0
        F2=PI*0.125D0*ZIN*(1.0D0-ZM*ZM)  ! Region b.
      ENDIF
C
      SEH2=Z*F2/((Z*Z+CHI2*F1(Z,X))**2+(CHI2*F2)**2)
      IF(IELPO.EQ.-1) THEN  ! Exchange correction for electrons.
        FEXP=4.0D0*Z*Z/(XE-X)
C       FEXP=1.0D0  !!!! NO EXCHANGE !!!!
        SEH2=SEH2*(1.0D0-FEXP*(1.0D0-FEXP))
      ENDIF
      RETURN
      END
C  *********************************************************************
C                       FUNCTION F1
C  *********************************************************************
      FUNCTION F1(Z,X)
C
C     Lindhard's f_1(z,x) function.
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
C
      IF(Z.LT.1.0D-5*X) THEN
        R=(Z/X)**2
        F1=-((16.0D0/3.0D0)+(256.0D0/5.0D0)*R)*R
        RETURN
      ENDIF
C
      ZIN=1.0D0/Z
      ZM=Z-X*ZIN*0.25D0
      IF(ABS(ZM).LT.1.0D-8) THEN
        AUX1=2.0D0*ZM-(4.0D0/3.0D0)*ZM**3-(4.0D0/15.0D0)*ZM**5
      ELSE
        ARGL=ABS((1.0D0+ZM)/(1.0D0-ZM))
        IF(ARGL.LT.1.0D-25.OR.ARGL.GT.1.0D25) THEN
          AUX1=0.0D0
        ELSE
          AUX1=(1.0D0-ZM**2)*LOG(ARGL)
        ENDIF
      ENDIF
C
      ZP=Z+X*ZIN*0.25D0
      IF(ABS(ZP).LT.1.0D-8) THEN
        AUX2=2.0D0*ZP-(4.0D0/3.0D0)*ZP**3-(4.0D0/15.0D0)*ZP**5
      ELSE
        ARGL=ABS((1.0D0+ZP)/(1.0D0-ZP))
        IF(ARGL.LT.1.0D-25.OR.ARGL.GT.1.0D25) THEN
          AUX2=0.0D0
        ELSE
          AUX2=(1.0D0-ZP**2)*LOG(ARGL)
        ENDIF
      ENDIF
C
      F1=0.5D0+0.125D0*(AUX1+AUX2)*ZIN
      RETURN
      END


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
      PARAMETER (NP=10, NP2=2*NP, NP4=4*NP, NOIT=256, NCALLT=100000)
      DIMENSION X(NP),W(NP),XM(NP),XP(NP)
      DIMENSION S(NOIT),SN(NOIT),XR(NOIT),XRN(NOIT)
C  Output error codes:
C     IERGA = 0, no problem, the calculation has converged.
C           = 1, too many open subintervals.
C           = 2, too many function calls.
      COMMON/CSUMGA/ERR,IERGA,NCALL  ! Error, code, function calls.
      DATA IWR/6/
C
C  ****  Gauss 20-point quadrature formula.
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
C
C                  *****************************
C                  *  SUBROUTINE PACKAGE DPWA  *
C                  *****************************
C
C
C                                          Francesc Salvat.
C                                          Universitat de Barcelona.
C                                          January 2, 2017
C
C
C     Dirac Partial Wave Analysis for elastic scattering of electrons
C  and positrons by Coulomb fields with short-range central
C  modifications. The radial Dirac equation is solved by using the
C  Fortran subroutine package RADIAL described in
C
C      F. Salvat and J.M. Fernandez-Varea,
C      'RADIAL: a FORTRAN subroutine package for the solution of the
C      radial Schrodinger and Dirac wave equations'.
C      Internal report, University of Barcelona, 2017.
C
C     The calling sequence from the main program is:
C
C****   CALL DPWA0(EV,NDELTA,ISCH)
C
C     This subroutine determines the phase shifts. It acts as the
C  initialization routine for the evaluation of scattering amplitudes
C  and differential cross sections.
C
C****   CALL DPWA(TH,CF,CG,DCS,SPL,ERRF,ERRG)
C
C     Subroutine DPWA gives elastic scattering functions at the
C  scattering angle TH obtained from the phase shifts calculated
C  previously by subroutine DPWA0.
C
C
C            ****  All I/O energies and lengths in eV and cm, resp.
C
C  *********************************************************************
C                      SUBROUTINE DPWA0
C  *********************************************************************
      SUBROUTINE DPWA0(EV,NDELTA,ISCH)
C
C     This subroutine computes Dirac phase shifts, differential cross
C  sections and scattering amplitudes for elastic scattering of
C  electrons in central fields.
C
C  Input arguments:
C     EV ....... effective kinetic energy of the projectile (eV).
C     NDELTA ... number of required phase shifts (LT.25000).
C     ISCH ..... =1: all phase shifts are computed by solving the radial
C                    equation.
C                =2: only phase shifts of selected orders are computed
C                    from the solution of the radial equation, the
C                    others are obtained by lin-log natural cubic spline
C                    interpolation. For high energies, ISCH=2 leads to a
C                    considerable reduction of the calculation time.
C
C  Input (through the common block /FIELD/):
C     R(I) .... radial grid points (radii in increasing order). The
C               first point in the grid must be the origin, i.e. R(1)=0.
C               Repeated values are interpreted as discontinuities.
C     RV(I).... R(I) times the potential energy at R=R(I). The last
C               component, RV(NP), is assumed to be equal to the
C               asymptotic value.
C     NP ...... number of input grid points.
C
C *** NOTE: The radii and potential values, R(I) and RV(I), are in
C           atomic units.
C
C  Output (through the common block /DCSTAB/):
C     ECS ........ total cross section (cm**2)
C                    (only for finite range fields).
C     TCS1 ....... 1st transport cross section (cm**2)
C                    (only for finite range fields).
C     TCS2 ....... 2nd transport cross section (cm**2)
C                    (only for finite range fields).
C     TH(I) ...... scattering angles (in deg)
C     XT(I) ...... values of (1-COS(TH(I)))/2.0D0.
C     DCST(I) .... differential cross section per unit solid angle at
C                    TH(I) (cm**2/sr).
C     ERROR(I) ... estimated relative uncertainty of the computed DCS
C                    value.
C     NTAB ....... number of angles in the table.
C
C  NOTE: The values of ECS, TCS1 and TCS2 are computed from the DCS
C  table. This introduces a certain error (of the order of 0.01 per
C  cent) but ensures consistency of multiple scattering simulations
C  using the DCS table.
C
C
      USE CONSTANTS
C
      IMPLICIT DOUBLE PRECISION (A-B,D-H,O-Z), COMPLEX*16 (C),
     1   INTEGER*4 (I-N)
      PARAMETER (A0B2=A0B*A0B)
      PARAMETER (PI=3.1415926535897932D0,FOURPI=4.0D0*PI)
C  ****  Input-output.
      COMMON/FIELD/R(NDIM),RV(NDIM),NP
      PARAMETER (NGT=650)
      COMMON/DCSTAB/ECS,TCS1,TCS2,TH(NGT),XT(NGT),DCST(NGT),SPOL(NGT),
     1              ERROR(NGT),NTAB
C  ****  Link with the RADIAL package.
      COMMON/RADWF/RRR(NDIM),P(NDIM),Q(NDIM),NRT,ILAST,IER
      PARAMETER (NPPG=NDIM+1)
      COMMON/VGRID/RG(NPPG),RVG(NPPG),VA(NPPG),VB(NPPG),VC(NPPG),
     1             VD(NPPG),NVT
C  ****  Phase shifts and partial wave series coefficients.
      PARAMETER (NPC=1500,NDM=25000)
      DIMENSION XL(NDM),DPI(NDM),DMI(NDM)
      COMMON/PHASES/DP(NDM),DM(NDM),NPH,ISUMP
      DIMENSION X(NDM),Y(NDM),SA(NDM),SB(NDM),SC(NDM),SD(NDM)
      COMMON/CSA/CFL(NDM),CGL(NDM),CFM(NDM),CGM(NDM),NPHM,IZINF
      COMMON/CRMORU/CFMC(NPC),CGMC(NPC),DPC(NPC),DMC(NPC),
     1              CFC,CGC,RUTHC,WATSC,RK2,ERRFC,ERRGC,NPC1
C
      CI=DCMPLX(0.0D0,1.0D0)
      ISUMP=0
C
      OPEN(98,FILE='dpwa.dat')
      WRITE(98,2000)
 2000 FORMAT(//2X,'**** Partial wave analysis (DPWA0) ',
     1  42('*')/)
C
      NDELT=NDELTA
      IF(NDELT.GT.NDM) THEN
        WRITE(98,2001)
 2001   FORMAT(/2X,'Warning: NDELTA is too large.')
        NDELT=NDM
      ENDIF
      IF(NDELT.LT.6) NDELT=6
      EPS=1.0D-15
      EPSCUT=1.0D-9
C
      E=EV/HREV
C
C  ****  Initialization of the RADIAL package.
C
      CALL VINT(R,RV,NP)
      ZINF=RVG(NVT)
      IF(ABS(ZINF).GT.1.0D-10) THEN
        IZINF=1
        CALL DPWAC0(ZINF,EV)
        IF(NDELT.GT.NPC) NDELT=NPC
      ELSE
        IZINF=0
      ENDIF
C
      WRITE(98,2002) EV
 2002 FORMAT(/2X,'Kinetic energy =',1P,E12.5,' eV')
      IF(E.LT.0.0D0) THEN
        WRITE(6,2002) EV
        WRITE(98,2003)
        WRITE(6,2003)
 2003   FORMAT(//2X,'Negative energy. Stop.')
        STOP 'DPWA0: Negative energy.'
      ENDIF
      RK=SQRT(E*(E+2.0D0*SL*SL))/SL
C
      IF(IZINF.EQ.1) WRITE(98,2004)
 2004 FORMAT(/2X,'Only inner phase shifts are tabulated')
      WRITE(98,2005)
 2005 FORMAT(/6X,'L',7X,'Phase(spin UP)',5X,'Phase(spin DOWN)',
     1       /2X,47('-'))
      ISCH0=ISCH
      IF(ISCH0.EQ.2.AND.EV.GT.1000.0D0) GO TO 1
C
C  ****  ISCH0=1, all phase shifts are computed by solving the radial
C        equation.
C
      L=0
      CALL DFREE(E,EPS,PHP,-1,0)
      IF(IER.NE.0) STOP 'DPWA0: Error in DFREE (1).'
      PHM=0.0D0
      WRITE(98,2006) L,PHP,PHM
      WRITE(6,2006) L,PHP,PHM
 2006 FORMAT(3X,I5,4X,1P,E16.8,4X,E16.8)
      DP(1)=PHP
      DM(1)=0.0D0
      NPH=1
C
      IFIRST=2
 33   CONTINUE
      ISUMP=1
      TST=0.0D0
      DO I=IFIRST,NDELT
        L=I-1
        CALL DFREE(E,EPS,PHP,-L-1,0)
        IF(IER.NE.0) STOP 'DPWA0: Error in DFREE (2).'
        CALL DFREE(E,EPS,PHM,L,0)
        IF(IER.NE.0) STOP 'DPWA0: Error in DFREE (3).'
        DP(I)=PHP
        DM(I)=PHM
        TST=MAX(ABS(PHP),ABS(PHM),ABS(DP(I-1)))
        NPH=I
        WRITE(98,2006) L,PHP,PHM
        WRITE(6,2006) L,PHP,PHM
        IF(TST.LT.EPSCUT.AND.L.GT.10) GO TO 6
C  ****  When the last phase shift (spin up) differs in more than 20 per
C  cent from the quadratic extrapolation, accumulated roundoff errors
C  may be important and the calculation of phase shifts is discontinued.
        IF(I.GT.500) THEN
          DPEXT=DP(I-3)+3.0D0*(DP(I-1)-DP(I-2))
          DPMAX=MAX(ABS(DP(I-3)),ABS(DP(I-2)),ABS(DP(I-1)),ABS(DP(I)))
          IF(ABS(DP(I)-DPEXT).GT.0.20D0*DPMAX) THEN
            NPH=I-1
            WRITE(98,2107)
            WRITE(6,2107)
 2107 FORMAT(/2X,'WARNING: Possible accumulation of round-off errors.')
            GO TO 6
          ENDIF
        ENDIF
      ENDDO
      WRITE(98,2007) TST
      WRITE(6,2007) TST
 2007 FORMAT(/2X,'WARNING: TST =',1P,E11.4,'. Check convergence.')
      GO TO 6
C
C  ****  ISCH0=2, only inner phase shifts of orders L in a given grid
C        are computed from the solution of the radial equation. Phase
C        shifts of orders not included in this grid are obtained by
C        lin-log cubic spline interpolation.
C          The adopted grid is: 0(1)100(5)300(10) ...
C
C        This is a somewhat risky procedure, which is based on the
C        observed variation of the calculated phase shifts with L for
C        atomic scattering fields. When a change of sign is found, all
C        the phases are recalculated.
C
 1    CONTINUE
      L=0
      CALL DFREE(E,EPS,PHP,-1,0)
      IF(IER.NE.0) STOP 'DPWA0: Error in DFREE (1).'
      PHM=0.0D0
      WRITE(98,2006) L,PHP,PHM
      WRITE(6,2006) L,PHP,PHM
      DP(1)=PHP
      DM(1)=0.0D0
C
      LMAX=NDELT-1
      IND=0
      IADD=1
      NADD=0
      LPP=1
 2    CONTINUE
      L=LPP
      CALL DFREE(E,EPS,PHP,-L-1,0)
      IF(IER.NE.0) STOP 'DPWA0: Error in DFREE (2).'
      CALL DFREE(E,EPS,PHM,L,0)
      IF(IER.NE.0) STOP 'DPWA0: Error in DFREE (3).'
      WRITE(6,2006) L,PHP,PHM
C
      DP(L+1)=PHP
      DM(L+1)=PHM
C
      IF(L.LT.95) THEN
        WRITE(98,2006) L,PHP,PHM
      ELSE
        IND=IND+1
        NADD=NADD+1
        XL(IND)=L
        DPI(IND)=PHP
        DMI(IND)=PHM
        IF(IND.GT.1) THEN
          TST1=DPI(IND)*DPI(IND-1)
          TST2=DMI(IND)*DMI(IND-1)
          IF(TST1.LT.0.0D0.OR.TST2.LT.0.0D0) THEN
            IF(L.LT.600) THEN
              ISCH0=1
              IFIRST=MIN(L,94)
              GO TO 33
            ELSE
              IND=IND-1
              L=XL(IND)+0.5D0
              GO TO 3
            ENDIF
          ENDIF
C
          IF(L.GT.600.AND.NADD.GT.3) THEN
            I=IND
            DPEXT=DPI(I-3)+3.0D0*(DPI(I-1)-DPI(I-2))
            DPMAX=MAX(ABS(DPI(I-3)),ABS(DPI(I-2)),ABS(DPI(I-1)),
     1            ABS(DPI(I)))
            IF(ABS(DPI(I)-DPEXT).GT.0.20D0*DPMAX) THEN
              IND=I-1
              L=XL(IND)+0.5D0
              WRITE(98,2107)
              WRITE(6,2107)
              GO TO 3
            ENDIF
            DMEXT=DMI(I-3)+3.0D0*(DMI(I-1)-DMI(I-2))
            DMMAX=MAX(ABS(DMI(I-3)),ABS(DMI(I-2)),ABS(DMI(I-1)),
     1            ABS(DMI(I)))
            IF(ABS(DMI(I)-DMEXT).GT.0.20D0*DMMAX) THEN
              IND=I-1
              L=XL(IND)+0.5D0
              WRITE(98,2107)
              WRITE(6,2107)
              GO TO 3
            ENDIF
          ENDIF
        ENDIF
      ENDIF
      TST=MAX(ABS(PHP),ABS(PHM))
      IF(TST.LT.EPSCUT) THEN
        IF(L.LT.100) THEN
          NPH=L+1
          GO TO 6
        ELSE
          GO TO 3
        ENDIF
      ENDIF
      IF(L.GE.LMAX) GO TO 3
C
      IADDO=IADD
      IF(L.GT.99) IADD=5
      IF(L.GT.299) IADD=10
      IF(L.GT.599) IADD=20
      IF(L.GT.1199) IADD=50
      IF(L.GT.2999) IADD=100
      IF(L.GT.9999) IADD=250
      IF(IADD.NE.IADDO) NADD=0
      LPP=L+IADD
      IF(LPP.GT.LMAX) LPP=LMAX
      GO TO 2
C
C  ****  Check consistency of sparsely tabulated phase shifts.
C        A discontinuity larger than 0.25*PI is considered as
C        a symptom of numerical inconsistencies.
C
 3    CONTINUE
      IF(IND.LT.5.OR.IADD.EQ.1) GO TO 6
      NPH=XL(IND)+1.5D0
      TST=0.0D0
      DO I=1,IND
        WRITE(98,2008) INT(XL(I)+0.5D0),DPI(I),DMI(I)
 2008   FORMAT(3X,I5,4X,1P,E16.8,4X,E16.8,'  i')
        IF(I.GT.1) THEN
          TST=MAX(TST,ABS(DPI(I)-DPI(I-1)),ABS(DMI(I)-DMI(I-1)))
        ENDIF
      ENDDO
      IF(TST.GT.0.25D0*PI) THEN
        WRITE(98,2009)
        WRITE(6,2009)
 2009   FORMAT(/2X,'ERROR: Directly computed phase shifts show',
     1    ' large discontinuities.')
        STOP 'DPWA0: Phase shifts do not vary continuously with L.'
      ENDIF
C
C  ****  Interpolated phase shifts (lin-log cubic spline).
C
      IF(DPI(4).GT.0.0D0) THEN
        ITRAN=+1
      ELSE
        ITRAN=-1
      ENDIF
      JT=0
      DO I=1,IND
        IF(DPI(I)*ITRAN.LT.0.0D0.OR.ABS(DPI(I)).LT.1.0D-12) THEN
          GO TO 4
        ELSE
          JT=JT+1
          X(JT)=XL(I)
          Y(JT)=LOG(ABS(DPI(I)))
        ENDIF
      ENDDO
 4    CONTINUE
      NUP=X(JT)+1.5D0
      NUP=MIN(NUP,NPH)
      CALL SPLINE(X,Y,SA,SB,SC,SD,0.0D0,0.0D0,JT)
      DO I=95+1,NUP
        RL=I-1
        CALL FINDI(RL,X,JT,J)
        DP(I)=ITRAN*EXP(SA(J)+RL*(SB(J)+RL*(SC(J)+RL*SD(J))))
      ENDDO
      IF(NUP.LT.NPH) THEN
        DO I=NUP+1,NPH
          DP(I)=0.0D0
        ENDDO
      ENDIF
C
      IF(DMI(4).GT.0.0D0) THEN
        ITRAN=+1
      ELSE
        ITRAN=-1
      ENDIF
      JT=0
      DO I=1,IND
        IF(DMI(I)*ITRAN.LT.0.0D0.OR.ABS(DMI(I)).LT.1.0D-12) THEN
          GO TO 5
        ELSE
          JT=JT+1
          X(JT)=XL(I)
          Y(JT)=LOG(ABS(DMI(I)))
        ENDIF
      ENDDO
 5    CONTINUE
      NUP=X(JT)+1.5D0
      NUP=MIN(NUP,NPH)
      CALL SPLINE(X,Y,SA,SB,SC,SD,0.0D0,0.0D0,JT)
      DO I=95+1,NUP
        RL=I-1
        CALL FINDI(RL,X,JT,J)
        DM(I)=ITRAN*EXP(SA(J)+RL*(SB(J)+RL*(SC(J)+RL*SD(J))))
      ENDDO
      IF(NUP.LT.NPH) THEN
        DO I=NUP+1,NPH
          DM(I)=0.0D0
        ENDDO
      ENDIF
C
      TST=MAX(ABS(DP(NPH)),ABS(DM(NPH)))
      IF(TST.GT.10.0D0*EPSCUT) THEN
        WRITE(98,2007) TST
        WRITE(6,2007) TST
      ENDIF
C
C  ************  Coefficients in the partial-wave expansion.
C
 6    CONTINUE
      CFACT=1.0D0/(2.0D0*CI*RK)
      IF(IZINF.EQ.1) THEN
        CXP=CDEXP(2.0D0*CI*DP(1))
        CXPC=CDEXP(2.0D0*CI*DPC(1))
        CFL(1)=CXPC*(CXP-1)*CFACT
        CGL(1)=0.0D0
        DO I=2,NPH
          L=I-1
          CXP=CDEXP(2.0D0*CI*DP(I))
          CXM=CDEXP(2.0D0*CI*DM(I))
          CXPC=CDEXP(2.0D0*CI*DPC(I))
          CXMC=CDEXP(2.0D0*CI*DMC(I))
          CFL(I)=((L+1)*CXPC*(CXP-1)+L*CXMC*(CXM-1))*CFACT
          CGL(I)=(CXMC*(CXM-1)-CXPC*(CXP-1))*CFACT
        ENDDO
      ELSE
        CXP=CDEXP(2*CI*DP(1))
        CFL(1)=(CXP-1.0D0)*CFACT
        CGL(1)=0.0D0
        DO I=2,NPH
          L=I-1
          CXP=CDEXP(2.0D0*CI*DP(I))
          CXM=CDEXP(2.0D0*CI*DM(I))
          CFL(I)=((L+1)*(CXP-1)+L*(CXM-1))*CFACT
          CGL(I)=CXM*(1.0D0-CDEXP(2.0D0*CI*(DP(I)-DM(I))))*CFACT
        ENDDO
      ENDIF
C
C  ****  Reduced series (two iterations).
C
      IF(NPH.GE.250.AND.ISUMP.EQ.0) THEN
        DO I=1,NPH
          CFM(I)=CFL(I)
          CGM(I)=CGL(I)
        ENDDO
C
        NPHM=NPH
        DO 7 NTR=1,2
          NPHM=NPHM-1
          CFC=0.0D0
          CFP=CFM(1)
          CGC=0.0D0
          CGP=CGM(1)
          DO I=1,NPHM
            RL=I-1
            CFA=CFC
            CFC=CFP
            CFP=CFM(I+1)
            CFM(I)=CFC-CFP*(RL+1)/(RL+RL+3)-CFA*RL/(RL+RL-1)
            CGA=CGC
            CGC=CGP
            CGP=CGM(I+1)
            CGM(I)=CGC-CGP*(RL+2)/(RL+RL+3)-CGA*(RL-1)/(RL+RL-1)
          ENDDO
 7      CONTINUE
      ENDIF
C
*     OPEN(99, file='pwa-coefs0.dat')
*     WRITE(99,'(A)') '# Coefficients in the partial-wave expansions'
*     WRITE(99,'(A,I6)') '# NPH =',NPH
*     WRITE(99,'(A)') '# L, CFL(L),CGL(L) all in a.u.'
*     DO I=1,NPH
*       WRITE(99,'(I5,1P,2(2X,E14.6,E14.6))') I-1,CFL(I),CGL(I)
*     ENDDO
*     CLOSE(99)
*     IF(NPH.GE.250.AND.ISUMP.EQ.0) THEN
*       OPEN(99, file='pwa-coefs2.dat')
*       WRITE(99,'(A)') '# Coefficients in the partial-wave expansions'
*       WRITE(99,'(A)') '#   after the reduced-series transformation'
*       WRITE(99,'(A,I6)') '# NPHM =',NPHM
*       WRITE(99,'(A)') '# L, CFM(L),CGM(L) all in a.u.'
*       DO I=1,NPHM
*         WRITE(99,'(I5,1P,2(2X,E14.6,E14.6))') I-1,CFM(I),CGM(I)
*       ENDDO
*       CLOSE(99)
*     ENDIF
C
C  ****  Scattering amplitudes and DCS.
C
      WRITE(98,2010)
 2010 FORMAT(//2X,'*** Scattering amplitudes and different',
     1  'ial cross section ***')
      WRITE(98,2011)
 2011 FORMAT(/4X,'Angle',6X,'DCS',7X,'Asymmetry',4X,'Direct amplitu',
     1  'de',7X,'Spin-flip amplitude',5X,'error',/4X,'(deg)',3X,
     2  '(cm**2/sr)',22X,'(cm)',20X,'(cm)',/2X,91('-'))
C
C  ****  Angular grid (TH in deg).
C
      TH(1)=0.0D0
      TH(2)=1.0D-4
      I=2
 10   CONTINUE
      I=I+1
      IF(TH(I-1).LT.0.9999D-3) THEN
        TH(I)=TH(I-1)+2.5D-5
      ELSE IF(TH(I-1).LT.0.9999D-2) THEN
        TH(I)=TH(I-1)+2.5D-4
      ELSE IF(TH(I-1).LT.0.9999D-1) THEN
        TH(I)=TH(I-1)+2.5D-3
      ELSE IF(TH(I-1).LT.0.9999D+0) THEN
        TH(I)=TH(I-1)+2.5D-2
      ELSE IF(TH(I-1).LT.0.9999D+1) THEN
        TH(I)=TH(I-1)+1.0D-1
      ELSE IF(TH(I-1).LT.2.4999D+1) THEN
        TH(I)=TH(I-1)+2.5D-1
      ELSE
        TH(I)=TH(I-1)+5.0D-1
      ENDIF
      IF(I.GT.NGT) STOP 'DPWA0. The NGT parameter is too small.'
      IF(TH(I).LT.180.0D0) GO TO 10
      NTAB=I
C
      DO I=1,NTAB
        THR=TH(I)*PI/180.0D0
        XT(I)=(1.0D0-COS(THR))/2.0D0
        CALL DPWA(THR,CF,CG,DCS,SPL,ERRF,ERRG)
        IF(MAX(ERRF,ERRG).GT.0.95D0) THEN
          ERR=1.0D0
        ELSE
          ACF=CDABS(CF)**2
          ACG=CDABS(CG)**2
          ERR=2.0D0*(ACF*ERRF+ACG*ERRG)/MAX(DCS,1.0D-45)
        ENDIF
        DCST(I)=DCS
        ERROR(I)=MAX(ERR,1.0D-7)
        SPOL(I)=SPL
        WRITE(98,2012) TH(I),DCST(I),SPOL(I),CF,CG,ERROR(I)
 2012   FORMAT(1X,1P,E10.3,E12.5,1X,E10.3,2(1X,'(',E10.3,',',
     1    E10.3,')'),E10.2)
      ENDDO
C
C  ************  Total and momentum transfer cross sections.
C                Convergence test (only for finite range fields).
C
      IF(IZINF.EQ.0) THEN
        INC=5
        IF(ISUMP.EQ.1) INC=1
        TST1=0.0D0
        TST2=0.0D0
        ECS=4.0D0*PI*CFL(1)*DCONJG(CFL(1))
        TCS=0.0D0
        ECSO=ECS
        TCSO=TCS
        DO I=2,NPH
          L=I-1
          RL=L
          DECS=CFL(I)*DCONJG(CFL(I))+RL*(L+1)*CGL(I)*DCONJG(CGL(I))
          DECS=4.0D0*PI*DECS/(L+L+1)
          DTCS=CFL(L)*DCONJG(CFL(I))+DCONJG(CFL(L))*CFL(I)
     1        +(L-1)*(RL+1)*(CGL(L)*DCONJG(CGL(I))
     2        +DCONJG(CGL(L))*CGL(I))
          DTCS=4.0D0*PI*DTCS*L/((RL+L-1)*(L+L+1))
          ECS=ECS+DECS
          TCS=TCS+DTCS
C  ****  Convergence test.
          ITW=L-(L/INC)*INC
          IF(ITW.EQ.0) THEN
            TST1=ABS(ECS-ECSO)/(ABS(ECS)+1.0D-35)
            TST2=ABS(TCS-TCSO)/(ABS(TCS)+1.0D-35)
            ECSO=ECS
            TCSO=TCS
          ENDIF
        ENDDO
        TST=MAX(TST1,TST2)
        TCS=ECS-TCS
        IF(TST.GT.1.0D-5.AND.NPH.GT.40) THEN
          WRITE(98,2007) TST
          WRITE(6,2007) TST
        ENDIF
        ECS=ECS*A0B2
        TCS=TCS*A0B2
C
C  ****  ECS and TCSs are evaluated from the DCS table.
C
        ECS0=FOURPI*SMOMLL(XT,DCST,XT(1),XT(NTAB),NTAB,0,0)
        ECS1=FOURPI*SMOMLL(XT,DCST,XT(1),XT(NTAB),NTAB,1,0)
        ECS2=FOURPI*SMOMLL(XT,DCST,XT(1),XT(NTAB),NTAB,2,0)
        TST1=ABS(ECS-ECS0)/(ABS(ECS)+1.0D-35)
        WRITE(98,2013) ECS,ECS0,TST1
        WRITE(6,2013) ECS,ECS0,TST1
 2013   FORMAT(/2X,'Total elastic cross section =',1P,E13.6,' cm**2',
     1         /2X,'             from DCS table =',E13.6,
     2         '  (rel. dif. =',E9.2,')')
        TCS1=2.0D0*ECS1
        TCS2=6.0D0*(ECS1-ECS2)
        TST2=ABS(TCS-TCS1)/(ABS(TCS)+1.0D-35)
        WRITE(98,2014) TCS,TCS1,TST2
        WRITE(6,2014) TCS,TCS1,TST2
 2014   FORMAT(/2X,'1st transport cross section =',1P,E13.6,' cm**2',
     1         /2X,'             from DCS table =',E13.6,
     2         '  (rel. dif. =',E9.2,')')
        WRITE(98,2015) TCS2
        WRITE(6,2015) TCS2
 2015   FORMAT(/2X,'2nd transport cross section =',1P,E13.6,' cm**2')
        TST=MAX(TST1,TST2)
        IF(TST.GT.2.0D-3) THEN
          WRITE(98,2016)
          WRITE(6,2016)
        ENDIF
      ENDIF
 2016 FORMAT(/2X,'WARNING: relative differences are too large.',
     1       /11X,'The dcs table is not consistent.')
C
      WRITE(98,2017)
 2017 FORMAT(/2X,'**** DPWA0 ended ',60('*')/)
      CLOSE(UNIT=98)
C
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE DPWA
C  *********************************************************************
      SUBROUTINE DPWA(TH,CF,CG,DCS,SPL,ERRF,ERRG)
C
C    This subroutine gives various elastic scattering functions at the
C  scattering angle TH (in radians) computed from Dirac phase shifts.
C  It should be previously initialized by calling subroutine DPWA0.
C
C  Input argument:
C     TH ....... scattering angle (in rad)
C
C  Output arguments:
C     CF ....... F scattering amplitude (cm).
C     CG ....... G scattering amplitude (cm).
C     DCS ...... differential cross section per unit solid angle for
C                unpolarized beams.
C     SPL ...... asymmetry function.
C     ERRF ..... relative uncertainty of CF.
C     ERRG ..... relative uncertainty of CG.
C
      USE CONSTANTS
C
      IMPLICIT DOUBLE PRECISION (A-B,D-H,O-Z),COMPLEX*16 (C),
     1   INTEGER*4 (I-N)
      PARAMETER (A0B2=A0B*A0B)
      PARAMETER (NPC=1500,NDM=25000,TOL=5.0D-8)
C  ****  Phase shifts and partial wave series coefficients.
      COMMON/PHASES/DP(NDM),DM(NDM),NPH,ISUMP
      COMMON/CSA/CFL(NDM),CGL(NDM),CFM(NDM),CGM(NDM),NPHM,IZINF
      COMMON/CRMORU/CFMC(NPC),CGMC(NPC),DPC(NPC),DMC(NPC),
     1              CFC,CGC,RUTHC,WATSC,RK2,ERRFC,ERRGC,NPC1
C
      X=COS(TH)
      Y=SIN(TH)
      TST=1.0D35
      CTERM=0.0D0
C
C  ************  Reduced series method. Only when TH is greater
C                than 0.5 deg and NPH.ge.250.
C
      IF(TH.LT.0.008726D0.OR.NPH.LT.250.OR.ISUMP.EQ.1) THEN
        CFO=0.0D0
        CGO=0.0D0
        ERRFO=1.0D10
        ERRGO=1.0D10
        GO TO 10
      ENDIF
      FACT=1.0D0/(1.0D0-X)**2
C
C  ****  F scattering amplitude.
C
      P2=1.0D0
      P3=X
      CFS=CFM(1)
      CFSO=CFS
      CFS=CFS+CFM(2)*P3
      DO I=3,NPHM
        L=I-1
        P1=P2
        P2=P3
        P3=((L+L-1)*X*P2-(L-1)*P1)/L
        CTERM=CFM(I)*P3
        CFS=CFS+CTERM
C  ****  Convergence test.
        IF(L.LT.149) THEN
          INC=1
        ELSE IF(L.LT.999) THEN
          INC=5
        ELSE
          INC=15
        ENDIF
        ITW=L-(L/INC)*INC
        IF(ITW.EQ.0) THEN
          TST=CDABS(CFS-CFSO)/MAX(CDABS(CFS),1.0D-45)
          CFSO=CFS
        ENDIF
      ENDDO
      CF=FACT*CFS
      ERRF=TST
C
C  ****  G scattering amplitude.
C
      IF(Y.LT.1.0D-30) THEN
        CG=0.0D0
        ERRG=0.0D0
      ELSE
        P2=1.0D0
        P3=3*X
        CGS=CGM(2)
        CGSO=CGS
        CGS=CGS+CGM(3)*P3
        DO I=4,NPHM
          L=I-1
          P1=P2
          P2=P3
          P3=((L+L-1)*X*P2-L*P1)/(L-1)
          CTERM=CGM(I)*P3
          CGS=CGS+CTERM
C  ****  Convergence test.
          IF(L.LT.149) THEN
            INC=1
          ELSE IF(L.LT.999) THEN
            INC=5
          ELSE
            INC=15
          ENDIF
          ITW=L-(L/INC)*INC
          IF(ITW.EQ.0) THEN
            TST=CDABS(CGS-CGSO)/MAX(CDABS(CGS),1.0D-45)
            CGSO=CGS
          ENDIF
        ENDDO
        CG=FACT*Y*CGS
        ERRG=TST
      ENDIF
C
      IF(ERRF.LT.TOL.AND.ERRG.LT.TOL) GO TO 20
      CFO=CF
      ERRFO=ERRF
      CGO=CG
      ERRGO=ERRG
C
C  ************  TH smaller than 0.5 deg or NPH.LT.250 or ISUMP=1.
C
 10   CONTINUE
C  ****  If IZINF=1, scattering functions are calculated only for
C        TH larger than 0.5 deg.
      IF(IZINF.EQ.1.AND.TH.LT.0.008726D0) THEN
        CF=0.0D0
        CG=0.0D0
        ERRF=1.0D0
        ERRG=1.0D0
        DCS=1.0D-45
        SPL=0.0D0
        RETURN
      ENDIF
C
C  ****  F scattering amplitude.
C
      P2=1.0D0
      P3=X
      CFS=CFL(1)
      CFSO=CFS
      CFS=CFS+CFL(2)*P3
      DO I=3,NPH
        L=I-1
        P1=P2
        P2=P3
        P3=((L+L-1)*X*P2-(L-1)*P1)/L
        CTERM=CFL(I)*P3
        CFS=CFS+CTERM
C  ****  Convergence test.
        IF(L.LT.149) THEN
          INC=1
        ELSE IF(L.LT.999) THEN
          INC=5
        ELSE
          INC=15
        ENDIF
        ITW=L-(L/INC)*INC
        IF(ITW.EQ.0) THEN
          TST=CDABS(CFS-CFSO)/MAX(CDABS(CFS),1.0D-45)
          CFSO=CFS
        ENDIF
      ENDDO
      CF=CFS
      ERRF=TST
C
C  ****  G scattering amplitude.
C
      IF(Y.LT.1.0D-30) THEN
        CG=0.0D0
        ERRG=0.0D0
      ELSE
        P2=1.0D0
        P3=3*X
        CGS=CGL(2)
        CGSO=CGS
        CGS=CGS+CGL(3)*P3
        DO I=4,NPH
          L=I-1
          P1=P2
          P2=P3
          P3=((L+L-1)*X*P2-L*P1)/(L-1)
          CTERM=CGL(I)*P3
          CGS=CGS+CTERM
C  ****  Convergence test.
          IF(L.LT.149) THEN
            INC=1
          ELSE IF(L.LT.999) THEN
            INC=5
          ELSE
            INC=15
          ENDIF
          ITW=L-(L/INC)*INC
          IF(ITW.EQ.0) THEN
            TST=CDABS(CGS-CGSO)/MAX(CDABS(CGS),1.0D-45)
            CGSO=CGS
          ENDIF
        ENDDO
        CG=Y*CGS
        ERRG=TST
      ENDIF
C  ****  The following four sentences are introduced to prevent abnormal
C        termination of the calculation when the number of (inner) phase
C        shifts is small. This solves the problem found by M. Berger.
      IF(NPH.LT.20.AND.CDABS(CTERM).LT.TOL) THEN
        ERRF=0.0D0
        ERRG=0.0D0
      ENDIF
C
C  ****  Select the most accurate method.
C
      IF(ERRFO.LT.ERRF) THEN
        CF=CFO
        ERRF=ERRFO
      ENDIF
      IF(ERRGO.LT.ERRG) THEN
        CG=CGO
        ERRG=ERRGO
      ENDIF
C
C  ****  Differential cross section (unpolarized beam).
C
 20   CONTINUE
      CF=CF*A0B
      CG=CG*A0B
      IF(IZINF.EQ.1) THEN
        XAUX=DPWAC(TH)
        CFC=CFC*A0B
        CGC=CGC*A0B
        DCSM=CDABS(CFC)**2+CDABS(CGC)**2
        CF=CF+CFC
        CG=CG+CGC
        ACF=CDABS(CF)**2
        ACG=CDABS(CG)**2
        DCS=ACF+ACG
C  ****  Scattering amplitudes that are much smaller than the Coulomb
C        ones may not be correct due to rounding off.
C        (Modified Coulomb fields only).
        IF(DCS.LT.1.0D-10*DCSM.OR.ERRFC+ERRGC.GT.1.0D0.OR.
     1    TH.LT.0.008726D0) THEN
          CF=0.0D0
          CG=0.0D0
          ERRF=1.0D0
          ERRG=1.0D0
          DCS=1.0D-45
          SPL=0.0D0
          RETURN
        ENDIF
        ERRF=ERRF+ERRFC
        ERRG=ERRG+ERRGC
      ELSE
        ACF=CDABS(CF)**2
        ACG=CDABS(CG)**2
        DCS=ACF+ACG
      ENDIF
C
      ERR=2.0D0*(ACF*ERRF+ACG*ERRG)/MAX(DCS,1.0D-45)
      IF(ERR.GT.0.10D0) THEN
        CF=0.0D0
        CG=0.0D0
        ERRF=1.0D0
        ERRG=1.0D0
        DCS=1.0D-45
        SPL=0.0D0
        RETURN
      ENDIF
C
C  ****  Asymmetry function.
C
      CSPL1=DCMPLX(0.0D0,1.0D0)*CF*DCONJG(CG)
      CSPL2=DCMPLX(0.0D0,1.0D0)*CG*DCONJG(CF)
      TST=CDABS(CSPL1-CSPL2)/MAX(CDABS(CSPL1),1.0D-45)
      IF(TST.GT.1.0D-3.AND.ERR.LT.0.01D0) THEN
        SPL=(CSPL1-CSPL2)/DCS
      ELSE
        SPL=0.0D0
      ENDIF
      RETURN
      END
C  *********************************************************************
C                       SUBROUTINE DPWAC0
C  *********************************************************************
      SUBROUTINE DPWAC0(ZZP,EV)
C
C     This subroutine computes Coulomb phase shifts and initializes the
C  calculation of the Mott differential cross section for electron or
C  positron elastic scattering by a bare point nucleus.
C
C  Input:
C     ZZP....... product of nuclear and projectile charges, that is, R
C                times the interaction energy at the distance R.
C                Negative for electrons, positive for positrons.
C     EV ....... kinetic energy of the projectile (eV).
C
C  After calling DPWAC0, the function DPWAC(TH) delivers the ratio
C  (Mott DCS / Rutherford DCS) for the scattering angle TH (rad).
C
      USE CONSTANTS
C
      IMPLICIT DOUBLE PRECISION (A-B,D-H,O-Z),COMPLEX*16 (C),
     1   INTEGER*4 (I-N)
C
      PARAMETER (SL2=SL*SL)
      PARAMETER (A0B2=A0B*A0B)
      PARAMETER (PI=3.1415926535897932D0,PIH=0.5D0*PI)
C  ****  Phase shifts and partial-wave series coefficients.
      PARAMETER (NPC=1500)
      COMMON/CRMORU/CFM(NPC),CGM(NPC),DPC(NPC),DMC(NPC),
     1              CF,CG,RUTHC,WATSC,RK2,ERRFC,ERRGC,NPC1
C
      CI=DCMPLX(0.0D0,1.0D0)
C
C  ************  Coulomb phase shifts.
C
      E=EV/HREV
      RUTHC=(A0B*2.0D0*ZZP*(1.0D0+E/SL2))**2
      PC=SQRT(E*(E+2.0D0*SL*SL))
      RK=PC/SL
      RK2=RK*RK
      ZETA=ZZP/SL
      W=E+SL2
      ETA=ZETA*W/PC
      RNUR=ZETA*(W+SL2)
C  ****  Negative kappa.
      DO I=1,NPC
        L=I-1
        K=-L-1
        RLAMB=SQRT(K*K-ZETA*ZETA)
        RNUI=-1.0D0*(K+RLAMB)*PC
        RNU=ATAN2(RNUI,RNUR)
        DELTAC=-CI*CLGAM(RLAMB+CI*ETA)
        DPC(I)=RNU-(RLAMB-(L+1))*PIH+DELTAC
      ENDDO
C  ****  Positive kappa.
      DMC(1)=0.0D0
      DO I=2,NPC
        L=I-1
        K=L
        RLAMB=SQRT(K*K-ZETA*ZETA)
        RNUI=-1.0D0*(K+RLAMB)*PC
        RNU=ATAN2(RNUI,RNUR)
        DELTAC=-CI*CLGAM(RLAMB+CI*ETA)
        DMC(I)=RNU-(RLAMB-(L+1))*PIH+DELTAC
      ENDDO
C
C  ****  Prints Coulomb phase shifts in file CPHASES.DAT.
C
*     OPEN(99,FILE='cphases.dat')
*     WRITE(99,1000)
*1000 FORMAT(2X,'# COULOMB PHASE SHIFTS',/2X,'#')
*     WRITE(99,1001) ZZP,E
*1001 FORMAT(2X,'# Z =',1P,E12.5,5X,'KINETIC ENERGY =',E12.5,/2X,'#')
*     WRITE(99,1002)
*1002 FORMAT(2X,'#   L',7X,'PHASE(SPIN UP)',4X,
*    1  'PHASE(SPIN DOWN)',/2X,'# ',45('-'))
*     DO I=1,NPC
*       DPI=DPC(I)
*       TT=ABS(DPI)
*       IF(TT.GT.PIH) DPI=DPI*(1.0D0-PI/TT)
*       DMI=DMC(I)
*       TT=ABS(DMI)
*       IF(TT.GT.PIH) DMI=DMI*(1.0D0-PI/TT)
*       WRITE(99,1003) I,DPI,DMI
*1003   FORMAT(3X,I5,4X,1P,E16.8,4X,E16.8)
*     ENDDO
*     CLOSE(UNIT=99)
C
C  ************  Coefficients in the partial wave expansion.
C
      CXP=CDEXP(2*CI*DPC(1))
      CFACT=1.0D0/(2.0D0*CI*RK)
      CFM(1)=(CXP-1.0D0)*CFACT
      CGM(1)=0.0D0
      DO I=2,NPC
        L=I-1
        RL=L
        CXP=CDEXP(2.0D0*CI*DPC(I))
        CXM=CDEXP(2.0D0*CI*DMC(I))
        CFM(I)=((L+1)*(CXP-1)+L*(CXM-1))*CFACT
        CGM(I)=(CXM-CXP)*CFACT
      ENDDO
C
C  ****  Reduced series.
C
      NPC1=NPC
      DO NTR=1,2
        NPC1=NPC1-1
        CFC=0.0D0
        CFP=CFM(1)
        CGC=0.0D0
        CGP=CGM(1)
        DO I=1,NPC1
          RL=I-1
          CFA=CFC
          CFC=CFP
          CFP=CFM(I+1)
          CFM(I)=CFC-CFP*(RL+1)/(RL+RL+3)-CFA*RL/(RL+RL-1)
          CGA=CGC
          CGC=CGP
          CGP=CGM(I+1)
          CGM(I)=CGC-CGP*(RL+2)/(RL+RL+3)-CGA*(RL-1)/(RL+RL-1)
        ENDDO
      ENDDO
C
C  ****  Bartlett and Watson's formula for small angles.
C
      TARG=-2.0D0*CLGAM(DCMPLX(0.5D0,ETA))*CI
      C5=CDEXP(TARG*CI)
      TARG=-2.0D0*CLGAM(DCMPLX(1.0D0,ETA))*CI
      C1=CDEXP(TARG*CI)
      BETA2=E*(E+2.0D0*SL2)/(E+SL2)**2
      WATSC=-PI*BETA2*ETA*(C5/C1)
C
      RETURN
      END
C  *********************************************************************
C                       FUNCTION DPWAC
C  *********************************************************************
      FUNCTION DPWAC(TH)
C
C     Ratio (Mott DCS / Rutherford DCS) for collisions with scattering
C  angle TH (rad). Additional information is provided through the common
C  block /CRMORU/.
C
      USE CONSTANTS
C
      IMPLICIT DOUBLE PRECISION (A-B,D-H,O-Z),COMPLEX*16 (C),
     1   INTEGER*4 (I-N)
      PARAMETER (A0B2=A0B*A0B)
      PARAMETER (NPC=1500)
      COMMON/CRMORU/CFM(NPC),CGM(NPC),DPC(NPC),DMC(NPC),
     1              CF,CG,RUTHC,WATSC,RK2,ERRF,ERRG,NPC1
C
      X=COS(TH)
      Q2=2.0D0*RK2*(1.0D0-X)
      NTEST=NPC1-5
C
C  ****  TH greater than 0.5 deg.
C
      IF(TH.LT.0.008726D0) GO TO 2
      FACT=1.0D0/(1.0D0-X)**2
C  ****  Direct scattering amplitude.
      P2=1.0D0
      P3=X
      CF=CFM(1)
      CF=CF+CFM(2)*P3
      CFA=0.0D0
      DO I=3,NPC1
        L=I-1
        P1=P2
        P2=P3
        P3=((L+L-1)*X*P2-(L-1)*P1)/L
        CF=CF+CFM(I)*P3
        IF(I.EQ.NTEST) CFA=CF
      ENDDO
      ERRF=CDABS(CFA-CF)/MAX(CDABS(CF),1.0D-15)
      CF=FACT*CF
C  ****  Spin-flip scattering amplitude.
      Y=SIN(TH)
      IF(Y.LT.1.0D-20) THEN
        CG=0.0D0
        ERRG=ERRF
        GO TO 1
      ENDIF
      P2=1.0D0
      P3=3*X
      CG=CGM(2)
      CG=CG+CGM(3)*P3
      CGA=0.0D0
      DO I=4,NPC1
        L=I-1
        P1=P2
        P2=P3
        P3=((L+L-1)*X*P2-L*P1)/(L-1)
        CG=CG+CGM(I)*P3
        IF(I.EQ.NTEST) CGA=CG
      ENDDO
      ERRG=CDABS(CGA-CG)/MAX(CDABS(CG),1.0D-15)
    1 CG=FACT*Y*CG
      PAV1=CDABS(CF)**2
      PAV2=CDABS(CG)**2
      ERR=2.0D0*(PAV1*ERRF+PAV2*ERRG)/(PAV1+PAV2)
      DCS=(PAV1+PAV2)*A0B2
      DPWAC=DCS*(Q2*Q2/RUTHC)
      IF(ERR.LT.1.0D-3.OR.TH.GT.0.08726D0) RETURN
C
C  ****  Bartlett and Watson's formula; used only for TH less than
C        5 deg, if needed. The computed DPWAC value may have slight
C        discontinuities, of the order of 0.01 per cent, between
C        0.5 and 5 deg.
C
 2    CONTINUE
      CF=0.0D0
      CG=0.0D0
      ERRF=1.0D10
      ERRG=1.0D10
      DPWAC=1.0D0+WATSC*SIN(0.5D0*TH)
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
      PARAMETER (EPS=1.0D-12, ONEM=1.0D0-EPS, ZERO=1.0D-98)
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

C >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
C >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
C
C  The following subroutine performs Dirac partial-wave calculations of
C  scattering of electrons and positrons in a complex central field with
C  an imaginary (absorptive) part.
C
C >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
C >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

C  *********************************************************************
C                      SUBROUTINE DPWAI0
C  *********************************************************************
      SUBROUTINE DPWAI0(EV,TOTCS,ABCS,NDELTA,ISCH)
C
C     This subroutine computes Dirac phase shifts, differential cross
C  sections and scattering amplitudes for elastic scattering of
C  electrons in central fields with an imaginary (absorptive) part.
C
C  Input/output arguments:
C     EV ....... effective kinetic energy of the projectile (eV).
C     TOTCS .... total cross section (cm**2).
C     ABCS ..... absorption cross section (cm**2).
C     NDELTA ... number of required phase shifts (LT.25000).
C     ISCH ..... =1: all phase shifts are computed by solving the radial
C                    equation.
C                =2: only phase shifts of selected orders are computed
C                    from the solution of the radial equation, the
C                    others are obtained by lin-log natural cubic spline
C                    interpolation. For high energies, ISCH=2 leads to a
C                    considerable reduction of the calculation time.
C
C  Input (through the common block /FIELDI/):
C     R(I) .... radial grid points (radii in increasing order). The
C               first point in the grid must be the origin, i.e. R(1)=0.
C               Repeated values are interpreted as discontinuities.
C     RV(I).... R(I) times the potential energy at R=R(I). The last
C               component, RV(NP), is assumed to be equal to the
C               asymptotic value.
C     RW(I).... R(I) times the imaginary potential (it must be negative
C               or zero).
C     IAB ..... 0 if the potential is real, 1 if it has an imaginary
C               part.
C     NP ...... number of input grid points.
C
C *** NOTE: The radii and potential values, R(I) and RV(I), are in
C           atomic units.
C
C  Output (through the common block /DCSTAB/):
C     ECS ........ total cross section (cm**2)
C                    (only for finite range fields).
C     TCS1 ....... 1st transport cross section (cm**2)
C                    (only for finite range fields).
C     TCS2 ....... 2nd transport cross section (cm**2)
C                    (only for finite range fields).
C     TH(I) ...... scattering angles (in deg)
C     XT(I) ...... values of (1-COS(TH(I)))/2.0D0.
C     DCST(I) .... differential cross section per unit solid angle at
C                    TH(I) (cm**2/sr).
C     ERROR(I) ... estimated relative uncertainty of the computed DCS
C                    value.
C     NTAB ....... number of angles in the table.
C
      USE CONSTANTS
C
      IMPLICIT DOUBLE PRECISION (A-B,D-H,O-Z), COMPLEX*16 (C),
     1   INTEGER*4 (I-N)
      PARAMETER (A0B2=A0B*A0B)
      PARAMETER (PI=3.1415926535897932D0,FOURPI=4.0D0*PI)
C  ****  Input-output.
      COMMON/FIELDI/R(NDIM),RV(NDIM),RW(NDIM),IAB,NP
      PARAMETER (NGT=650)
      COMMON/DCSTAB/ECS,TCS1,TCS2,TH(NGT),XT(NGT),DCST(NGT),SPOL(NGT),
     1              ERROR(NGT),NTAB
C  ****  Link with the RADIAL package.
      COMMON/RADWF/RRR(NDIM),P(NDIM),Q(NDIM),NRT,ILAST,IER
      COMMON/RADWFI/PIM(NDIM),QIM(NDIM)
      PARAMETER (NPPG=NDIM+1)
      COMMON/VGRID/RG(NPPG),RVG(NPPG),VA(NPPG),VB(NPPG),VC(NPPG),
     1             VD(NPPG),NVT
C  ****  Phase shifts and partial wave series coefficients.
      PARAMETER (NPC=1500,NDM=25000)
      DIMENSION CXP(NDM),CXM(NDM)
      COMMON/PHASES/DP(NDM),DM(NDM),NPH,ISUMP
      COMMON/PHASEI/DPJ(NDM),DMJ(NDM)
      DIMENSION XL(NDM),DPI(NDM),DMI(NDM),DPJI(NDM),DMJI(NDM)
      DIMENSION X(NDM),Y(NDM),SA(NDM),SB(NDM),SC(NDM),SD(NDM)
      COMMON/CSA/CFL(NDM),CGL(NDM),CFM(NDM),CGM(NDM),NPHM,IZINF
      COMMON/CRMORU/CFMC(NPC),CGMC(NPC),DPC(NPC),DMC(NPC),
     1              CFC,CGC,RUTHC,WATSC,RK2,ERRFC,ERRGC,NPC1
C
      CI=DCMPLX(0.0D0,1.0D0)
      ISUMP=0
C
      OPEN(98,FILE='dpwai.dat')
      WRITE(98,2000)
 2000 FORMAT(//2X,'**** Partial wave analysis (DPWAI0) ',
     1  42('*')/)
C
      NDELT=NDELTA
      IF(NDELT.GT.NDM) THEN
        WRITE(98,2001)
 2001   FORMAT(/2X,'WARNING: NDELTA is too large')
        NDELT=NDM
      ENDIF
      IF(NDELT.LT.6) NDELT=6
      EPS=1.0D-15
      EPSCUT=1.0D-9
C
      E=EV/HREV
C
C  ****  Initialization of the RADIAL package.
C
      IF(IAB.EQ.0) THEN
        CALL VINT(R,RV,NP)
      ELSE
        CALL ZVINT(R,RV,RW,NP)
      ENDIF
      ZINF=RVG(NVT)
      IF(ABS(ZINF).GT.1.0D-10) THEN
        IZINF=1
        CALL DPWAC0(ZINF,EV)
        IF(NDELT.GT.NPC) NDELT=NPC
      ELSE
        IZINF=0
      ENDIF
C
      WRITE(98,2002) EV
 2002 FORMAT(/2X,'Kinetic energy =',1P,E12.5,' eV')
      IF(E.LT.0.0D0) THEN
        WRITE(6,2002) EV
        WRITE(98,2003)
        WRITE(6,2003)
 2003   FORMAT(//2X,'Negative energy. Stop.')
        STOP 'DPWAI0: Negative energy.'
      ENDIF
      RK=SQRT(E*(E+2.0D0*SL*SL))/SL
C
      IF(IZINF.EQ.1) WRITE(98,2004)
 2004 FORMAT(/2X,'Only inner phase shifts are tabulated')
      WRITE(98,2005)
 2005 FORMAT(/14X,'--------- Spin UP ---------',6X,
     1  '-------- Spin DOWN --------',/6X,'L',9X,
     2  'Re(phase)      Im(phase)',9X,'Re(phase)      Im(phase)',
     3  /2X,74('-'))
      ISCH0=ISCH
      IF(ISCH0.EQ.2.AND.EV.GT.1000.0D0) GO TO 1
C
C  ****  ISCH0=1, all phase shifts are computed by solving the radial
C        equation.
C
      L=0
      IF(IAB.EQ.0) THEN
        CALL DFREE(E,EPS,PHP,-1,0)
        DP(1)=PHP
        DPJ(1)=0.0D0
        DM(1)=0.0D0
        DMJ(1)=0.0D0
        CXP(1)=CDEXP(2.0D0*CI*PHP)
      ELSE
        CALL ZDFREE(E,EPS,PHPR,PHPI,-1,0)
        IF(ABS(PHPI).LT.1.0D-12) PHPI=0.0D0
        DP(1)=PHPR
        DPJ(1)=PHPI
        DM(1)=0.0D0
        DMJ(1)=0.0D0
        CXP(1)=CDEXP(2.0D0*(CI*PHPR-PHPI))
      ENDIF
      IF(IER.NE.0) STOP 'DPWAI0: Error in ZDPHAS (1).'
      WRITE(98,2006) L,DP(1),DPJ(1),DM(1),DMJ(1)
      WRITE(6,2006) L,DP(1),DPJ(1),DM(1),DMJ(1)
 2006 FORMAT(3X,I5,4X,1P,E16.8,1X,E12.5,4X,E16.8,1X,E12.5)
      NPH=1
C
      IFIRST=2
 33   CONTINUE
      ISUMP=1
      TST=0.0D0
      DO I=IFIRST,NDELT
        L=I-1
        IF(IAB.EQ.0) THEN
          CALL DFREE(E,EPS,PHP,-L-1,0)
          DP(I)=PHP
          DPJ(I)=0.0D0
          CXP(I)=CDEXP(2.0D0*CI*PHP)
        ELSE
          CALL ZDFREE(E,EPS,PHPR,PHPI,-L-1,0)
          IF(ABS(PHPI).LT.1.0D-12) PHPI=0.0D0
          DP(I)=PHPR
          DPJ(I)=PHPI
          CXP(I)=CDEXP(2.0D0*(CI*PHPR-PHPI))
        ENDIF
        IF(IER.NE.0) STOP 'DPWAI0: Error in ZDPHAS (2).'
        IF(IAB.EQ.0) THEN
          CALL DFREE(E,EPS,PHM,L,0)
          DM(I)=PHM
          DMJ(I)=0.0D0
          CXM(I)=CDEXP(2.0D0*CI*PHM)
        ELSE
          CALL ZDFREE(E,EPS,PHMR,PHMI,L,0)
          IF(ABS(PHMI).LT.1.0D-12) PHMI=0.0D0
          DM(I)=PHMR
          DMJ(I)=PHMI
          CXM(I)=CDEXP(2.0D0*(CI*PHMR-PHMI))
        ENDIF
        IF(IER.NE.0) STOP 'DPWAI0: Error in ZDPHAS (3).'
        NPH=I
        WRITE(98,2006) L,DP(I),DPJ(I),DM(I),DMJ(I)
        WRITE(6,2006) L,DP(I),DPJ(I),DM(I),DMJ(I)
        TST=MAX(SQRT(DP(I)**2+DPJ(I)**2),SQRT(DM(I)**2+DMJ(I)**2),
     1    SQRT(DP(I-1)**2+DM(I-1)**2))
        IF(TST.LT.EPSCUT.AND.L.GT.10) GO TO 6
C  ****  When the last phase shift (spin up) differs in more than 20 per
C  cent from the quadratic extrapolation, accumulated roundoff errors
C  may be important and the calculation of phase shifts is discontinued.
        IF(I.GT.500) THEN
          DPEXT=DP(I-3)+3.0D0*(DP(I-1)-DP(I-2))
          DPMAX=MAX(ABS(DP(I-3)),ABS(DP(I-2)),ABS(DP(I-1)),
     1              ABS(DP(I)))
          IF(ABS(DP(I)-DPEXT).GT.0.20D0*DPMAX) THEN
            NPH=I-1
            WRITE(98,2107)
            WRITE(6,2107)
 2107 FORMAT(/2X,'WARNING: Possible accumulation of round-off errors.')
            GO TO 6
          ENDIF
        ENDIF
      ENDDO
      WRITE(98,2007) TST
      WRITE(6,2007) TST
 2007 FORMAT(/2X,'WARNING: TST =',1P,E11.4,'. Check convergence.')
      GO TO 6
C
C  ****  ISCH0=2, only inner phase shifts of orders L in a given grid
C        are computed from the solution of the radial equation. Phase
C        shifts of orders not included in this grid are obtained by
C        lin-log cubic spline interpolation.
C          The adopted grid is: 0(1)100(5)300(10) ...
C
C        This is a somewhat risky procedure, which is based on the
C        observed variation of the calculated phase shifts with L for
C        atomic scattering fields. When a change of sign is found, all
C        the phases are recalculated.
C
 1    CONTINUE
      L=0
      IF(IAB.EQ.0) THEN
        CALL DFREE(E,EPS,PHP,-1,0)
        DP(1)=PHP
        DPJ(1)=0.0D0
        DM(1)=0.0D0
        DMJ(1)=0.0D0
        CXP(1)=CDEXP(2.0D0*CI*PHP)
      ELSE
        CALL ZDFREE(E,EPS,PHPR,PHPI,-1,0)
        IF(ABS(PHPI).LT.1.0D-12) PHPI=0.0D0
        DP(1)=PHPR
        DPJ(1)=PHPI
        DM(1)=0.0D0
        DMJ(1)=0.0D0
        CXP(1)=CDEXP(2.0D0*(CI*PHPR-PHPI))
      ENDIF
      IF(IER.NE.0) STOP 'DPWAI0: Error in ZDPHAS (4).'
      WRITE(98,2006) L,DP(1),DPJ(1),DM(1),DMJ(1)
      WRITE(6,2006) L,DP(1),DPJ(1),DM(1),DMJ(1)
C
      LMAX=NDELT-1
      IND=0
      IADD=1
      NADD=0
      LPP=1
 2    CONTINUE
      L=LPP
      IF(IAB.EQ.0) THEN
        CALL DFREE(E,EPS,PHP,-L-1,0)
        DP(L+1)=PHP
        DPJ(L+1)=0.0D0
        CXP(L+1)=CDEXP(2.0D0*CI*PHP)
      ELSE
        CALL ZDFREE(E,EPS,PHPR,PHPI,-L-1,0)
        IF(ABS(PHPI).LT.1.0D-12) PHPI=0.0D0
        DP(L+1)=PHPR
        DPJ(L+1)=PHPI
        CXP(L+1)=CDEXP(2.0D0*(CI*PHPR-PHPI))
      ENDIF
      IF(IER.NE.0) STOP 'DPWAI0: Error in ZDPHAS (5).'
      IF(IAB.EQ.0) THEN
        CALL DFREE(E,EPS,PHM,L,0)
        DM(L+1)=PHM
        DMJ(L+1)=0.0D0
        CXM(L+1)=CDEXP(2.0D0*CI*PHM)
      ELSE
        CALL ZDFREE(E,EPS,PHMR,PHMI,L,0)
        IF(ABS(PHMI).LT.1.0D-12) PHMI=0.0D0
        DM(L+1)=PHMR
        DMJ(L+1)=PHMI
        CXM(L+1)=CDEXP(2.0D0*(CI*PHMR-PHMI))
      ENDIF
      IF(IER.NE.0) STOP 'DPWAI0: Error in ZDPHAS (6).'
      WRITE(6,2006) L,DP(L+1),DPJ(L+1),DM(L+1),DMJ(L+1)
C
      IF(L.LT.95) THEN
        WRITE(98,2006) L,DP(L+1),DPJ(L+1),DM(L+1),DMJ(L+1)
      ELSE
        IND=IND+1
        NADD=NADD+1
        XL(IND)=L
        DPI(IND)=DP(L+1)
        DPJI(IND)=DPJ(L+1)
        DMI(IND)=DM(L+1)
        DMJI(IND)=DMJ(L+1)
        IF(IND.GT.1) THEN
          TST1=DPI(IND)*DPI(IND-1)
          TST2=DMI(IND)*DMI(IND-1)
          IF(TST1.LT.0.0D0.OR.TST2.LT.0.0D0) THEN
            IF(L.LT.600) THEN
              ISCH0=1
              IFIRST=MIN(L,94)
              GO TO 33
            ELSE
              IND=IND-1
              L=XL(IND)+0.5D0
              GO TO 3
            ENDIF
          ENDIF
C
          IF(L.GT.600.AND.NADD.GT.3) THEN
            I=IND
            DPEXT=DPI(I-3)+3.0D0*(DPI(I-1)-DPI(I-2))
            DPMAX=MAX(ABS(DPI(I-3)),ABS(DPI(I-2)),ABS(DPI(I-1)),
     1            ABS(DPI(I)))
            IF(ABS(DPI(I)-DPEXT).GT.0.20D0*DPMAX) THEN
              IND=I-1
              L=XL(IND)+0.5D0
              WRITE(98,2107)
              WRITE(6,2107)
              GO TO 3
            ENDIF
            DMEXT=DMI(I-3)+3.0D0*(DMI(I-1)-DMI(I-2))
            DMMAX=MAX(ABS(DMI(I-3)),ABS(DMI(I-2)),ABS(DMI(I-1)),
     1            ABS(DMI(I)))
            IF(ABS(DMI(I)-DMEXT).GT.0.20D0*DMMAX) THEN
              IND=I-1
              L=XL(IND)+0.5D0
              WRITE(98,2107)
              WRITE(6,2107)
              GO TO 3
            ENDIF
          ENDIF
        ENDIF
      ENDIF
      TST=MAX(SQRT(DP(L+1)**2+DPJ(L+1)**2),SQRT(DM(L+1)**2+DMJ(L+1)**2))
      IF(TST.LT.EPSCUT) THEN
        IF(L.LT.100) THEN
          NPH=L+1
          GO TO 6
        ELSE
          GO TO 3
        ENDIF
      ENDIF
      IF(L.GE.LMAX) GO TO 3
C
      IADDO=IADD
      IF(L.GT.99) IADD=5
      IF(L.GT.299) IADD=10
      IF(L.GT.599) IADD=20
      IF(L.GT.1199) IADD=50
      IF(L.GT.2999) IADD=100
      IF(L.GT.9999) IADD=250
      IF(IADD.NE.IADDO) NADD=0
      LPP=L+IADD
      IF(LPP.GT.LMAX) LPP=LMAX
      GO TO 2
C
C  ****  Check consistency of sparsely tabulated phase shifts.
C        A discontinuity larger than 0.25*PI is considered as
C        a symptom of numerical inconsistencies.
C
 3    CONTINUE
      IF(IND.LT.5.OR.IADD.EQ.1) GO TO 6
      NPH=XL(IND)+1.5D0
      TST=0.0D0
      DO I=1,IND
        WRITE(98,2008) INT(XL(I)+0.5D0),DPI(I),DPJI(I),DMI(I),DMJI(I)
 2008   FORMAT(3X,I5,4X,1P,E16.8,1X,E12.5,4X,E16.8,1X,E12.5,'  i')
        IF(I.GT.1) THEN
          TST=MAX(TST,ABS(DPI(I)-DPI(I-1)),ABS(DMI(I)-DMI(I-1)))
        ENDIF
      ENDDO
      IF(TST.GT.0.25D0*PI) THEN
        WRITE(98,2009)
        WRITE(6,2009)
 2009   FORMAT(/2X,'ERROR: Directly computed phase shifts show',
     1    ' large discontinuities.')
        STOP 'DPWAI0: Phase shifts do not vary continuously with L.'
      ENDIF
C
C  ****  Interpolated phase shifts (lin-log cubic spline).
C
      IF(DPI(4).GT.0.0D0) THEN
        ITRAN=+1
      ELSE
        ITRAN=-1
      ENDIF
      JT=0
      DO I=1,IND
        IF(DPI(I)*ITRAN.LT.0.0D0.OR.ABS(DPI(I)).LT.1.0D-12) THEN
          GO TO 4
        ELSE
          JT=JT+1
          X(JT)=XL(I)
          Y(JT)=LOG(ABS(DPI(I)))
        ENDIF
      ENDDO
 4    CONTINUE
      NUP=X(JT)+1.5D0
      NUP=MIN(NUP,NPH)
      CALL SPLINE(X,Y,SA,SB,SC,SD,0.0D0,0.0D0,JT)
      DO I=95+1,NUP
        RL=I-1
        CALL FINDI(RL,X,JT,J)
        DP(I)=ITRAN*EXP(SA(J)+RL*(SB(J)+RL*(SC(J)+RL*SD(J))))
      ENDDO
      IF(NUP.LT.NPH) THEN
        DO I=NUP+1,NPH
          DP(I)=0.0D0
        ENDDO
      ENDIF
C
      JT=0
      DO I=1,IND
        IF(ABS(DPJI(I)).LT.1.0D-12) THEN
          GO TO 41
        ELSE
          JT=JT+1
          X(JT)=XL(I)
          Y(JT)=DPJI(I)
        ENDIF
      ENDDO
 41   CONTINUE
      NUP=X(JT)+1.5D0
      NUP=MIN(NUP,NPH)
      CALL SPLINE(X,Y,SA,SB,SC,SD,0.0D0,0.0D0,JT)
      DO I=95+1,NUP
        RL=I-1
        CALL FINDI(RL,X,JT,J)
        DPJ(I)=SA(J)+RL*(SB(J)+RL*(SC(J)+RL*SD(J)))
      ENDDO
      IF(NUP.LT.NPH) THEN
        DO I=NUP+1,NPH
          DPJ(I)=0.0D0
        ENDDO
      ENDIF
C
      IF(DMI(4).GT.0.0D0) THEN
        ITRAN=+1
      ELSE
        ITRAN=-1
      ENDIF
      JT=0
      DO I=1,IND
        IF(DMI(I)*ITRAN.LT.0.0D0.OR.ABS(DMI(I)).LT.1.0D-12) THEN
          GO TO 5
        ELSE
          JT=JT+1
          X(JT)=XL(I)
          Y(JT)=LOG(ABS(DMI(I)))
        ENDIF
      ENDDO
 5    CONTINUE
      NUP=X(JT)+1.5D0
      NUP=MIN(NUP,NPH)
      CALL SPLINE(X,Y,SA,SB,SC,SD,0.0D0,0.0D0,JT)
      DO I=95+1,NUP
        RL=I-1
        CALL FINDI(RL,X,JT,J)
        DM(I)=ITRAN*EXP(SA(J)+RL*(SB(J)+RL*(SC(J)+RL*SD(J))))
      ENDDO
      IF(NUP.LT.NPH) THEN
        DO I=NUP+1,NPH
          DM(I)=0.0D0
        ENDDO
      ENDIF
C
      JT=0
      DO I=1,IND
        IF(ABS(DMJI(I)).LT.1.0D-12) THEN
          GO TO 51
        ELSE
          JT=JT+1
          X(JT)=XL(I)
          Y(JT)=DMJI(I)
        ENDIF
      ENDDO
 51   CONTINUE
      NUP=X(JT)+1.5D0
      NUP=MIN(NUP,NPH)
      CALL SPLINE(X,Y,SA,SB,SC,SD,0.0D0,0.0D0,JT)
      DO I=95+1,NUP
        RL=I-1
        CALL FINDI(RL,X,JT,J)
        DMJ(I)=SA(J)+RL*(SB(J)+RL*(SC(J)+RL*SD(J)))
      ENDDO
      IF(NUP.LT.NPH) THEN
        DO I=NUP+1,NPH
          DMJ(I)=0.0D0
        ENDDO
      ENDIF
C
      TST=MAX(ABS(DP(NPH)),ABS(DM(NPH)))
      IF(TST.GT.10.0D0*EPSCUT) THEN
        WRITE(98,2007) TST
        WRITE(6,2007) TST
      ENDIF
      DO I=1,NPH
        CXP(I)=CDEXP(2.0D0*CI*DCMPLX(DP(I),DPJ(I)))
        CXM(I)=CDEXP(2.0D0*CI*DCMPLX(DM(I),DMJ(I)))
      ENDDO
C
C  ************  Coefficients in the partial-wave expansion.
C
 6    CONTINUE
      CFACT=1.0D0/(2.0D0*CI*RK)
      IF(IZINF.EQ.1) THEN
        CXPC=CDEXP(2*CI*DPC(1))
        CFL(1)=CXPC*(CXP(1)-1)*CFACT
        CGL(1)=0.0D0
        DO I=2,NPH
          L=I-1
          CXPC=CDEXP(2.0D0*CI*DPC(I))
          CXMC=CDEXP(2.0D0*CI*DMC(I))
          CFL(I)=((L+1)*CXPC*(CXP(I)-1)+L*CXMC*(CXM(I)-1))*CFACT
          CGL(I)=(CXMC*(CXM(I)-1)-CXPC*(CXP(I)-1))*CFACT
        ENDDO
      ELSE
        CFL(1)=(CXP(1)-1.0D0)*CFACT
        CGL(1)=0.0D0
        DO I=2,NPH
          L=I-1
          CFL(I)=((L+1)*(CXP(I)-1)+L*(CXM(I)-1))*CFACT
          CGL(I)=(CXM(I)-CXP(I))*CFACT
        ENDDO
      ENDIF
C
C  ****  Reduced series (two iterations).
C
      IF(NPH.GE.250.AND.ISUMP.EQ.0) THEN
        DO I=1,NPH
          CFM(I)=CFL(I)
          CGM(I)=CGL(I)
        ENDDO
C
        NPHM=NPH
        DO 7 NTR=1,2
          NPHM=NPHM-1
          CFC=0.0D0
          CFP=CFM(1)
          CGC=0.0D0
          CGP=CGM(1)
          DO I=1,NPHM
            RL=I-1
            CFA=CFC
            CFC=CFP
            CFP=CFM(I+1)
            CFM(I)=CFC-CFP*(RL+1)/(RL+RL+3)-CFA*RL/(RL+RL-1)
            CGA=CGC
            CGC=CGP
            CGP=CGM(I+1)
            CGM(I)=CGC-CGP*(RL+2)/(RL+RL+3)-CGA*(RL-1)/(RL+RL-1)
          ENDDO
 7      CONTINUE
      ENDIF
C
*     OPEN(99, file='pwa-coefs0.dat')
*     WRITE(99,'(A)') '# Coefficients in the partial-wave expansions'
*     WRITE(99,'(A,I6)') '# NPH =',NPH
*     WRITE(99,'(A)') '# L, CFL(L),CGL(L) all in a.u.'
*     DO I=1,NPH
*       WRITE(99,'(I5,1P,2(2X,E14.6,E14.6))') I-1,CFL(I),CGL(I)
*     ENDDO
*     CLOSE(99)
*     IF(NPH.GE.250.AND.ISUMP.EQ.0) THEN
*       OPEN(99, file='pwa-coefs2.dat')
*       WRITE(99,'(A)') '# Coefficients in the partial-wave expansions'
*       WRITE(99,'(A)') '#   after the reduced-series transformation'
*       WRITE(99,'(A,I6)') '# NPHM =',NPHM
*       WRITE(99,'(A)') '# L, CFM(L),CGM(L) all in a.u.'
*       DO I=1,NPHM
*         WRITE(99,'(I5,1P,2(2X,E14.6,E14.6))') I-1,CFM(I),CGM(I)
*       ENDDO
*       CLOSE(99)
*     ENDIF
C
C  ****  Scattering amplitudes and DCS.
C
      WRITE(98,2010)
 2010 FORMAT(//2X,'*** Scattering amplitudes and different',
     1  'ial cross section ***')
      WRITE(98,2011)
 2011 FORMAT(/4X,'Angle',6X,'DCS',7X,'Asymmetry',4X,'Direct amplitu',
     1  'de',7X,'Spin-flip amplitude',5X,'Error',/4X,'(deg)',3X,
     2  '(cm**2/sr)',22X,'(cm)',20X,'(cm)',/2X,91('-'))
C
C  ****  Angular grid (TH in deg).
C
      TH(1)=0.0D0
      TH(2)=1.0D-4
      I=2
 10   CONTINUE
      I=I+1
      IF(TH(I-1).LT.0.9999D-3) THEN
        TH(I)=TH(I-1)+2.5D-5
      ELSE IF(TH(I-1).LT.0.9999D-2) THEN
        TH(I)=TH(I-1)+2.5D-4
      ELSE IF(TH(I-1).LT.0.9999D-1) THEN
        TH(I)=TH(I-1)+2.5D-3
      ELSE IF(TH(I-1).LT.0.9999D+0) THEN
        TH(I)=TH(I-1)+2.5D-2
      ELSE IF(TH(I-1).LT.0.9999D+1) THEN
        TH(I)=TH(I-1)+1.0D-1
      ELSE IF(TH(I-1).LT.2.4999D+1) THEN
        TH(I)=TH(I-1)+2.5D-1
      ELSE
        TH(I)=TH(I-1)+5.0D-1
      ENDIF
      IF(I.GT.NGT) STOP 'DPWAI0. The NGT parameter is too small.'
      IF(TH(I).LT.180.0D0) GO TO 10
      NTAB=I
C
      DO I=1,NTAB
        THR=TH(I)*PI/180.0D0
        XT(I)=(1.0D0-COS(THR))/2.0D0
        CALL DPWA(THR,CF,CG,DCS,SPL,ERRF,ERRG)
        IF(MAX(ERRF,ERRG).GT.0.95D0) THEN
          ERR=1.0D0
        ELSE
          ACF=CDABS(CF)**2
          ACG=CDABS(CG)**2
          ERR=2.0D0*(ACF*ERRF+ACG*ERRG)/MAX(DCS,1.0D-45)
        ENDIF
        DCST(I)=DCS
        ERROR(I)=MAX(ERR,1.0D-7)
        SPOL(I)=SPL
        WRITE(98,2012) TH(I),DCST(I),SPOL(I),CF,CG,ERROR(I)
 2012   FORMAT(1X,1P,E10.3,E12.5,1X,E10.3,2(1X,'(',E10.3,',',
     1    E10.3,')'),E10.2)
      ENDDO
C
C  ************  Total and momentum transfer cross sections.
C                Convergence test (only for finite range fields).
C
      IF(IZINF.EQ.0) THEN
        INC=5
        IF(ISUMP.EQ.1) INC=1
        TST1=0.0D0
        TST2=0.0D0
        ECS=4.0D0*PI*CFL(1)*DCONJG(CFL(1))
        TCS=0.0D0
        ECSO=ECS
        TCSO=TCS
        DO I=2,NPH
          L=I-1
          RL=L
          DECS=CFL(I)*DCONJG(CFL(I))+RL*(L+1)*CGL(I)*DCONJG(CGL(I))
          DECS=4.0D0*PI*DECS/(L+L+1)
          DTCS=CFL(L)*DCONJG(CFL(I))+DCONJG(CFL(L))*CFL(I)
     1        +(L-1)*(RL+1)*(CGL(L)*DCONJG(CGL(I))
     2        +DCONJG(CGL(L))*CGL(I))
          DTCS=4.0D0*PI*DTCS*L/((RL+L-1)*(L+L+1))
          ECS=ECS+DECS
          TCS=TCS+DTCS
C  ****  Convergence test.
          ITW=L-(L/INC)*INC
          IF(ITW.EQ.0) THEN
            TST1=ABS(ECS-ECSO)/(ABS(ECS)+1.0D-35)
            TST2=ABS(TCS-TCSO)/(ABS(TCS)+1.0D-35)
            ECSO=ECS
            TCSO=TCS
          ENDIF
        ENDDO
        TST=MAX(TST1,TST2)
        TCS=ECS-TCS
        IF(TST.GT.1.0D-5.AND.NPH.GT.40) THEN
          WRITE(98,2007) TST
          WRITE(6,2007) TST
        ENDIF
        ECS=ECS*A0B2
        TCS=TCS*A0B2
C
C  ****  ECS and TCSs are evaluated from the DCS table.
C
        ECS0=FOURPI*SMOMLL(XT,DCST,XT(1),XT(NTAB),NTAB,0,0)
        ECS1=FOURPI*SMOMLL(XT,DCST,XT(1),XT(NTAB),NTAB,1,0)
        ECS2=FOURPI*SMOMLL(XT,DCST,XT(1),XT(NTAB),NTAB,2,0)
        TST1=ABS(ECS-ECS0)/(ABS(ECS)+1.0D-35)
        WRITE(98,2013) ECS,ECS0,TST1
        WRITE(6,2013) ECS,ECS0,TST1
 2013   FORMAT(/2X,'Total elastic cross section =',1P,E13.6,' cm**2',
     1         /2X,'             From DCS table =',E13.6,
     2         '  (Rel. dif. =',E9.2,')')
        TCS1=2.0D0*ECS1
        TCS2=6.0D0*(ECS1-ECS2)
        TST2=ABS(TCS-TCS1)/(ABS(TCS)+1.0D-35)
        WRITE(98,2014) TCS,TCS1,TST2
        WRITE(6,2014) TCS,TCS1,TST2
 2014   FORMAT(/2X,'1ST transport cross section =',1P,E13.6,' cm**2',
     1         /2X,'             From DCS table =',E13.6,
     2         '  (REL. DIF. =',E9.2,')')
        WRITE(98,2015) TCS2
        WRITE(6,2015) TCS2
 2015   FORMAT(/2X,'2ND transport cross section =',1P,E13.6,' cm**2')
        TST=MAX(TST1,TST2)
        IF(TST.GT.2.0D-3) THEN
          WRITE(98,2016)
          WRITE(6,2016)
        ENDIF
 2016   FORMAT(/2X,'WARNING: Relative differences are too large.',
     1         /11X,'The DCS table is not consistent.')
C
C  ****  Absorption cross section.
C
        CALL DPWA(0.0D0,CF,CG,DCS,SPL,ERRF,ERRG)
        TOTCS=(FOURPI*(A0B/RK))*(-CI*CF)
        ABCS=TOTCS-ECS
        WRITE(98,2018) TOTCS
        WRITE(6,2018) TOTCS
 2018   FORMAT(/2X,'  Grand total cross section =',1P,E13.6,' cm**2')
        WRITE(98,2019) ABCS
        WRITE(6,2019) ABCS
 2019   FORMAT(2X,'   Absorption cross section =',1P,E13.6,' cm**2')
      ENDIF
C
      WRITE(98,2017)
 2017 FORMAT(/2X,'**** DPWAI0 ended ',60('*')/)
      CLOSE(UNIT=98)
C
      RETURN
      END
