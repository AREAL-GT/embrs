import numpy as np
from embrs.utilities.unit_conversions import *
from embrs.fire_simulator.cell import Cell

MAXNO = 20
MAXKL = int((MAXNO * ( MAXNO + 1 ) / 2 + MAXNO ))
MXSTEP = 20
tpdry = 353.0
ch2o = 4186.0
t_small = 1e-6
t_big = 1e6
FintSwitch = 15.0

class Burnup():
    def __init__(self, cell: Cell):
        self.gd_Fudge1 = 0.0
        self.gd_Fudge2 = 0.0

        self.now = 0
        self.tis = 0

        self.ntimes = 0
        self.nruns = 0

        self.tdf = 0
        self.dfi = 0

        self.key = np.zeros(MAXNO)
        self.diam = np.zeros(MAXKL)
        self.xmat = np.zeros(MAXKL)

        self.wd0 = None        

        self.Smoldering = np.zeros(MAXNO + 1) # last element contains duff mass burning rate
        self.Flaming = np.zeros(MAXNO)

        self.fout = np.zeros(MAXNO)
        self.work = np.zeros(MAXNO)
        self.flit = np.zeros(MAXNO)
        self.alfa = np.zeros(MAXNO)
        self.fint = np.zeros(MAXNO)

        self.wo = np.zeros(MAXKL)
        self.tign = np.zeros(MAXKL)
        self.tout = np.zeros(MAXKL)
        self.qcum = np.zeros(MAXKL)
        self.tcum = np.zeros(MAXKL)
        self.tdry = np.zeros(MAXKL)
        self.ddot = np.zeros(MAXKL)
        self.wodot = np.zeros(MAXKL)
        self.acum = np.zeros(MAXKL)

        self.qdot = np.zeros((MAXKL, MXSTEP))

        self.hf = None
        self.hb = None
        self.en = None

        self.number = cell._fuel.num_classes # Number of fuel size categories

        self.wdry = Lbsft2_to_KiSq(cell.wdry) # Dry loading (kg/m2)
        self.fmois = cell.fmois[0:self.number] # fuel moisture (fraction)
        self.sigma = cell.sigma * 3.2808 # SAV (1/m)
        self.ash = [0.05] * self.number # ash content (fraction)
        self.htval = [18600000] * self.number # heat content(J/kg)
        self.dendry = [512.591] * self.number # dry mass density (kg/m3)
        self.cheat = [2750] * self.number # heat capacity (J/kg/K)
        self.condry = [0.133] * self.number # thermal conductivity (W/m/K)
        self.tpig = [327 + 273] * self.number # ignition tempurature (C)
        self.tchar = [377 + 273] * self.number # char end pyrolisis temperature (C)

        self.fi_min = 0.1

    def set_fire_data(self, NumIter, Fi, Ti, U, D, Tamb, Dt, Wdf, Dfm):
        self.ntimes = NumIter
        fi_kwm2 = BTU_ft2_min_to_kW_m2(Fi) # convert intensity to kW/m2
        self.fistart = fi_kwm2 # igniting fire intensity (kW/m2)
        self.fi = fi_kwm2
        self.ti = Ti  # igniting surface fire res. time (s)
        self.u = U # windspeed at top of fuelbed (m/s)
        self.d = ft_to_m(D) # depth of fuel bed (m)
        TambC = F_to_C(Tamb) # ambient temperature (C)
        self.tamb = TambC + 273 # ambient temperature (K)
        self.r0 = 1.83
        self.dr = 0.4
        self.dt = Dt 
        wdf = TPA_to_KiSq(Wdf) # duff dry weight loading (kg/m2)
        self.wdf = wdf
        self.dfm = Dfm # duff moisture (fraction)

    def arrays(self):
        indices = list(range(self.number))
        indices.sort(key=lambda i: (1.0 / self.sigma[i], self.fmois[i], self.dendry[i]))
        self.key = indices

        self.wdry   = [self.wdry[i] for i in self.key]
        self.ash    = [self.ash[i] for i in self.key]
        self.htval  = [self.htval[i] for i in self.key]
        self.fmois  = [self.fmois[i] for i in self.key]
        self.dendry = [self.dendry[i] for i in self.key]
        self.sigma  = [self.sigma[i] for i in self.key]
        self.cheat  = [self.cheat[i] for i in self.key]
        self.condry = [self.condry[i] for i in self.key]
        self.tpig   = [self.tpig[i] for i in self.key]
        self.tchar  = [self.tchar[i] for i in self.key]

        self.overlaps()

        for k in range(1, self.number + 1):
            diak = 4.0 / self.sigma[k - 1]
            wtk = self.wdry[k - 1]
            kl = self.loc(k, 0)
            self.diam[kl] = diak
            self.xmat[kl] = self.alone[k - 1]
            self.wo[kl] = wtk * self.xmat[kl]
            for j in range(1, k + 1):
                kj = self.loc(k, j)
                self.diam[kj] = diak
                self.xmat[kj] = self.elam[k - 1][j - 1]
                self.wo[kj] = wtk * self.xmat[kj]

    def overlaps(self):
        self.elam = np.zeros((MAXNO, MAXNO))
        self.alone = np.zeros(MAXNO)
        self.area = np.zeros(MAXNO)

        for j in range(1, self.number + 1):
            self.alone[j-1] = 0.0
            for k in range(1, j + 1):
                kj = self.loc(j, k)
                self.xmat[kj] = 0.0
            for k in range(1, self.number + 1):
                self.elam[j-1][k-1] = 0.0

        for k in range(1, self.number + 1):
            for l in range(1, k+1):
                ak = 3.25 * np.exp(-20.0 * (self.fmois[l - 1]**2))
                siga = ak * self.sigma[k - 1] / np.pi
                kl = self.loc(k, l)
                a = siga * self.wdry[l - 1] / self.dendry[l - 1]
                if k == l:
                    bb = 1 - np.exp(-a)
                    bb = max(1e-30, bb)
                    self.area[k - 1] = bb

                else:
                    bb = min(1.0, a)
                
                self.xmat[kl] = bb

        if self.number == 1:
            self.elam[0][0] = self.xmat[1]
            self.alone[0] = 1 - self.elam[0][0]

            return
        
        for k in range(1, self.number + 1):
            frac = 0.0
            for l in range(1, k+1):
                kl = self.loc(k, l)
                frac += self.xmat[kl]
            
            if frac > 1.0:
                for l in range(1, k+1):
                    kl = self.loc(k,l)
                    self.elam[k-1][l-1] = self.xmat[kl] / frac
                
                self.alone[k - 1] = 0.0
            
            else:
                for l in range(1, k+1):
                    kl = self.loc(k,l)
                    self.elam[k-1][l-1] = self.xmat[kl]
                
                self.alone[k-1] = 1.0 - frac

    def loc(self, k, l):
        return int(k * (k + 1) / 2 + l - 1)
    
    def duff_burn(self):
        if self.wdf <= 0.0 or self.dfm >= 1.96:
            return
        
        self.dfi = 11.25 - 4.05 * self.dfm
        ff = 0.837 - 0.426 * self.dfm
        self.tdf = 1e4 * ff * self.wdf / (7.5 - 2.7 * self.dfm)

        self.Smoldering[MAXNO] += ((ff * self.wdf) / (self.tdf))

    def start(self, dt, now):
        rindef = 1e30

        for k in range(1, self.number + 1):
            self.fout[k - 1] = 0.0
            self.flit[k - 1] = 0.0
            self.alfa[k - 1] = self.condry[k - 1] / (self.dendry[k - 1] * self.cheat[k - 1])

            delm = 1.67 * self.fmois[k - 1]

            heatk = self.dendry[k - 1] / 446.0

            heatk *= 2.01e6 * (1 + delm)

            self.work[k - 1] = 1.0 / (255.0 * heatk)

            for l in range(k + 1):
                kl = self.loc(k, l)
                self.tout[kl] = rindef
                self.tign[kl] = rindef
                self.tdry[kl] = rindef
                self.tcum[kl] = 0.0
                self.qcum[kl] = 0.0

        r = self.r0 + 0.25 * self.dr
        tf = self.TempF(self.fi, r)
        ts = self.tamb

        if (tf <= (tpdry + 10.0)):
            return False

        thd = (tpdry - ts) / (tf - ts)
        tx = 0.5 * (ts + tpdry)

        for k in range(1, self.number + 1):
            factor = self.dendry[k - 1] * self.fmois[k - 1]
            conwet = self.condry[k - 1] + 4.27e-4 * factor
            for l in range(k + 1):
                kl = self.loc(k, l)
                dia = self.diam[kl]
                self.heat_exchange(dia, tf, tx, conwet)
                dryt = self.dry_time(self.en, thd)
                cpwet = self.cheat[k - 1] + self.fmois[k - 1] * ch2o
                fac = ((0.5 * dia) ** 2) / conwet
                fac = fac * self.dendry[k - 1] * cpwet
                dryt = fac * dryt
                self.tdry[kl] = dryt

        tsd = tpdry

        for k in range(1, self.number+1):
            c = self.condry[k - 1]
            tigk = self.tpig[k - 1]
            for l in range(k+1):
                kl = self.loc(k, l)
                dryt = self.tdry[kl]
                if dryt >= dt:
                    continue
                dia = self.diam[kl]
                ts = 0.5 * (tsd + tigk)
                self.heat_exchange(dia, tf, ts, c)
                self.tcum[kl] = max((tf - ts) * (dt - dryt), 0.0)
                self.qcum[kl] = self.hb * self.tcum[kl]
                if (tf <= (tigk + 10.0)):
                    continue
                dtign = self.TIgnite(tpdry, self.tpig[k- 1], tf, self.condry[k-1],
                                     self.cheat[k-1], self.fmois[k-1], self.dendry[k-1], self.hb)
                trt = dryt + dtign
                self.tign[kl] = 0.5 * trt
                if (dt > trt):
                    self.flit[k - 1] += self.xmat[kl]

        nlit = 0
        trt = rindef

        for k in range(1, self.number + 1):
            if self.flit[k - 1] > 0.0:
                nlit += 1
            
            for l in range(k + 1):
                kl = self.loc(k, l)
                trt = min(trt, self.tign[kl])

        if nlit == 0:
            return False

        for k in range(1, self.number + 1):
            for l in range(k + 1):
                kl = self.loc(k, l)
                if (self.tdry[kl] < rindef):
                    self.tdry[kl] -= trt
                if (self.tign[kl] < rindef):
                    self.tign[kl] -= trt

        for k in range(1, self.number + 1):
            if self.flit[k - 1] == 0.0:
                for l in range(k + 1):
                    kl = self.loc(k, l)
                    self.ddot[kl] = 0.0
                    self.tout[kl] = rindef
                    self.wodot[kl] = 0.0
            else:
                ts = self.tchar[k - 1]
                c = self.condry[k - 1]
                for l in range(k + 1):
                    kl = self.loc(k, l)
                    dia = self.diam[kl]
                    self.heat_exchange(dia, tf, ts, c)

                    self.qdot[kl][now - 1] = self.hb * max((tf - ts), 0.0)
                    aint = (c / self.hb) ** 2
                    ddt = dt - self.tign[kl]
                    self.acum[kl] = aint * ddt
                    self.ddot[kl] = self.qdot[kl][now - 1] * self.work[k - 1]
                    self.tout[kl] = dia / self.ddot[kl]
                    dnext = max(0.0, (dia - ddt * self.ddot[kl]))
                    wnext = self.wo[kl] * (dnext / dia) ** 2
                    self.wodot[kl] = (self.wo[kl] - wnext) / ddt
                    self.diam[kl] = dnext
                    self.wo[kl] = wnext

                    df = 0.0
                    if (dnext <= 0.0):
                        df = self.xmat[kl]
                        if (kl == 0):
                            self.gd_Fudge1 = self.wodot[kl]
                        else:
                            self.gd_Fudge2 = self.wodot[kl]
                        
                        self.wodot[kl] = 0.0
                        self.ddot[kl] = 0.0
                    self.flit[k-1] -= df
                    self.fout[k-1] += df

        self.nruns = 0
        return True

    def TIgnite(self, tpdr, tpig, tpfi, cond, chtd, fmof, dend, hbar):
        pinv = 2.125534
        hvap = 2.177e6
        cpm = 4186.0
        conc = 4.27e-4

        def ff(x, tpfi, tpig):
            a03 = -1.3371565
            a13 = 0.4653628
            a23 = -0.1282064

            b03 = a03 * (tpfi - tpig) / (tpfi - self.tamb)

            return b03 + x * (a13 + x * (a23 + x))

        xlo = 0.0
        xhi = 1.0

        xav = 0.5 * (xlo + xhi)
        fav = ff(xav, tpfi, tpig)

        while abs(fav) > t_small:
            xav = 0.5 * (xlo + xhi)
            fav = ff(xav, tpfi, tpig)
            if (abs(fav) > t_small):
                if fav < 0.0:
                    xlo = xav
                if fav > 0.0:
                    xhi = xav

        beta = pinv * (1.0 - xav) / xav
        conw = cond + conc * dend * fmof
        dtb = tpdr - self.tamb
        dti = tpig - self.tamb
        ratio = (hvap + cpm * dtb) / (chtd * dti)
        rhoc = dend * chtd * (1.0 + fmof * ratio)
        tmig = (beta / hbar)**2 * conw * rhoc

        return tmig

    def dry_time(self, enu, theta):
        p = 0.47047
        xl = 0.0
        xh = 1.0

        def func(h, theta):
            a = 0.7478556
            b = 0.4653628
            c = 0.1282064

            return h * (b - h * (c - h)) - (1.0 - theta) / a

        for _ in range(15):
            xm = 0.5 * (xl + xh)
            if func(xm, theta) < 0.0:
                xl = xm
            else:
                xh = xm

        x = (1 / xm - 1.0)/ p
        tau = (0.5 * x / enu) ** 2

        return tau
    
    def heat_exchange(self, dia, tf, ts, cond):
        g = 9.8
        vis = 7.5e-5
        a = 8.75e-3
        b = 5.75e-5
        rad = 5.67e-8
        fmfac = 0.382
        hradf = 0.5

        self.hfm = 0
        if (dia > b):
            v = np.sqrt(self.u * self.u + 0.53 * g * self.d)
            re = v * dia / vis
            enuair = 0.344 * (re ** 0.56)
            conair = a + b * tf
            fac = np.sqrt(abs(tf - ts) / dia)
            hfmin = fmfac * np.sqrt(fac)
            self.hfm = max((enuair * conair / dia), hfmin)

        hrad = hradf * rad * (tf + ts) * (tf * tf + ts * ts)
        self.hb = self.hfm + hrad
        self.en = self.hb * dia / cond

    def step(self, dt, tin, fid):
        rindef = 1e30

        self.nruns += 1
        now = self.nruns
        tnow = tin
        tnext = tnow + dt

        tifi = tnow - (now - 1) * dt

        for k in range(1, self.number + 1):
            c = self.condry[k - 1]

            for l in range(k + 1):
                kl = self.loc(k, l)
                tdun = self.tout[kl]

                if tnow >= tdun:
                    self.ddot[kl] = 0.0
                    self.wodot[kl] = 0.0

                    continue
            
                if tnext >= tdun:
                    tgo = tdun - tnow
                    self.ddot[kl] = self.diam[kl] / tgo
                    self.wodot[kl] = self.wo[kl] / tgo
                    self.wo[kl] = 0.0
                    self.diam[kl] = 0.0

                    continue

                tlit = self.tign[kl]
                if tnow >= tlit:
                    ts = self.tchar[k - 1]
                    if l == 0:
                        r = self.r0 + 0.5 * self.dr
                        gi = self.fi + fid
                    elif l == k:
                        r = self.r0 + 0.5 * (1.0 + self.flit[k - 1]) * self.dr
                        gi = self.fi + self.flit[k - 1] * self.fint[k - 1]
                    else:
                        r = self.r0 + 0.5 * (1.0 + self.flit[l - 1]) * self.dr
                        gi = self.fi + self.fint[k - 1] + self.flit[l - 1] * self.fint[l - 1]
                    tf = self.TempF(gi, r)
                    dia = self.diam[kl]
                    self.heat_exchange(dia, tf, ts, c)
                    qqq = self.hb * max(tf - ts, 0.0)
                    tst = max(tlit, tifi)
                    nspan = max(1, int(round((tnext - tst) / dt)))
                    if nspan <= MXSTEP:
                        self.qdot[kl][nspan - 1] = qqq
                    elif nspan > MXSTEP:
                        for mu in range(2, MXSTEP + 1):
                            self.qdot[kl][mu - 2] = self.qdot[kl][mu - 1]
                        self.qdot[kl][MXSTEP - 1] = qqq
                    aint = (c / self.hb) ** 2
                    self.acum[kl] += (aint * dt)
                    tav1 = tnext - tlit
                    tav2 = self.acum[kl] / self.alfa[k - 1]
                    tav3 = (dia / 4.0) ** 2 / self.alfa[k - 1]
                    tavg = tav1
                    if tav2 < tavg:
                        tavg = tav2
                    if tav3 < tavg:
                        tavg = tav3
                    index = min(nspan, MXSTEP)
                    qdsum = 0.0
                    tspan = 0.0
                    deltim = dt

                    while index > 0:
                        index -= 1
                        if index == 0:
                            deltim = tnext - tspan - tlit
                        if (tspan + deltim) >= tavg:
                            deltim = tavg - tspan
                        qdsum += (self.qdot[kl][index] * deltim)
                        tspan += deltim
                        if tspan >= tavg:
                            break

                    qdavg = max(qdsum / tspan, 0.0)
                    self.ddot[kl] = qdavg * self.work[k - 1]
                    dnext = max(0.0, dia - dt * self.ddot[kl])
                    wnext = self.wo[kl] * (dnext / dia) ** 2
                    if dnext == 0.0 and self.ddot[kl] > 0.0:
                        self.tout[kl] = tnow + dia / self.ddot[kl]
                    elif dnext > 0.0 and dnext < dia:
                        rate = dia / (dia - dnext)
                        self.tout[kl] = tnow + rate * dt
                    if qdavg <= MXSTEP:
                        self.tout[kl] = 0.5 * (tnow + tnext)
                    ddt = min(dt, (self.tout[kl] - tnow))
                    self.wodot[kl] = (self.wo[kl] - wnext) / ddt
                    self.diam[kl] = dnext
                    self.wo[kl] = wnext
                    
                    continue

                dryt = self.tdry[kl]
                if tnow >= dryt and tnow < tlit:
                    if l == 0:
                        r = self.r0
                        gi = self.fi + fid
                    elif l == k:
                        r = self.r0
                        gi = self.fi
                    else:
                        r = self.r0 + 0.5 * self.flit[l - 1] * self.dr
                        gi = self.fi + self.flit[l - 1] * self.fint[l - 1]

                    tf = self.TempF(gi, r)
                    ts = self.tamb
                    dia = self.diam[kl]
                    self.heat_exchange(dia, tf, ts, c)

                    dtemp = max(0.0, tf - ts)
                    dqdt = self.hb * dtemp

                    self.qcum[kl] += (dqdt * dt)
                    self.tcum[kl] += (dtemp * dt)
                    dteff = self.tcum[kl] / (tnext - dryt)
                    heff = self.qcum[kl] / self.tcum[kl]
                    tfe = ts + dteff
                    dtlite = rindef
                    if tfe > self.tpig[k - 1] + 10:
                        dtlite = self.TIgnite(tpdry, self.tpig[k - 1], tfe, self.condry[k - 1],
                                              self.cheat[k - 1], self.fmois[k - 1], self.dendry[k-1],
                                              heff)
                    self.tign[kl] = 0.5 * (dryt + dtlite)

                    if (tnext > self.tign[kl]):
                        ts = self.tchar[k - 1]
                        self.heat_exchange(dia, tf, ts, c)
                        self.qdot[kl][0] = self.hb * max(tf -ts, 0.0)
                        qd = self.qdot[kl][0]
                        self.ddot[kl] = qd * self.work[k - 1]
                        delt = tnext - self.tign[kl]
                        dnext = max(0.0, dia - delt * self.ddot[kl])
                        wnext = self.wo[kl] * (dnext / dia) ** 2
                        if dnext == 0.0:
                            self.tout[kl] = tnow + dia / self.ddot[kl]
                        elif (dnext > 0.0) and dnext < dia:
                            rate = dia / (dia - dnext)
                            self.tout[kl] = tnow + rate * dt
                        if self.tout[kl] > tnow:
                            ddt = min(dt, (self.tout[kl] - tnow))
                            self.wodot[kl] = (self.wo[kl] - wnext) / ddt
                        else:
                            self.wodot[kl] = 0.0
                        self.diam[kl] = dnext
                        self.wo[kl] = wnext

                    continue

                if tnow < dryt:
                    factor = self.fmois[k - 1] * self.dendry[k - 1]
                    conwet = self.condry[k - 1] + 4.27e-4 * factor
                    if l == 0:
                        r = self.r0
                        gi = self.fi * fid
                    elif l == k:
                        r = self.r0
                        gi = self.fi
                    elif l != 0 and l != k:
                        r = self.r0 + 0.5 * self.flit[l - 1] * self.dr
                        gi = self.fi + self.flit[l - 1] * self.fint[l - 1]
                    tf = self.TempF(gi, r)
                    if tf <= tpdry + 10.0:
                        continue
                    dia = self.diam[kl]
                    ts = 0.5 * (self.tamb + tpdry)
                    self.heat_exchange(dia, tf, ts, c)
                    dtcum = max((tf - ts) * dt, 0.0)
                    self.tcum[kl] += dtcum
                    self.qcum[kl] += (self.hb * dtcum)
                    he = self.qcum[kl] / self.tcum[kl]
                    dtef = self.tcum[kl] / tnext
                    thd = (tpdry - self.tamb) / dtef
                    if (thd > 0.9):
                        continue
                    biot = he * dia / conwet

                    dryt = self.dry_time(biot, thd)
                    cpwet = self.cheat[k - 1] + ch2o * self.fmois[k - 1]
                    fac = ((0.5 * dia) ** 2) / conwet
                    fac = fac * cpwet * self.dendry[k - 1]
                    self.tdry[kl] = fac * dryt
                    if (self.tdry[kl] < tnext):
                        ts = tpdry
                        self.heat_exchange(dia, tf, ts, c)
                        dqdt = self.hb * (tf - ts)
                        delt = tnext - self.tdry[kl]
                        self.qcum[kl] = dqdt * delt
                        self.tcum[kl] = (tf - ts) * delt

                        if tf <= self.tpig[k - 1] + 10:
                            continue
                        dtlite = self.TIgnite(tpdry, self.tpig[k - 1], tf, self.condry[k - 1],
                                              self.cheat[k - 1], self.fmois[k - 1], self.dendry[k - 1], self.hb)
                        self.tign[kl] = 0.5 * (self.tdry[kl] + dtlite)
                        if (tnext > self.tign[kl]):
                            ts = self.tchar[k - 1]
                            self.qdot[kl][0] = self.hb * (max(tf - ts, 0.0))

        for k in range(1, self.number + 1):
            self.flit[k - 1] = 0.0
            self.fout[k - 1] = 0.0
            for l in range(k + 1):
                kl = self.loc(k, l)
                if (tnext >= self.tign[kl]):
                    flag = True
                else:
                    flag = False
                if flag and tnext <= self.tout[kl]:
                    self.flit[k - 1] += self.xmat[kl]
                if tnext > self.tout[kl]:
                    self.fout[k - 1] += self.xmat[kl]

    def get_flaming_front_consumption(self):
        fuel_loadings = []

        for m in range(1, self.number + 1):
            consumed = (self.fistart * self.ti) / 18608
            rem = self.wdry[m - 1] - consumed
            rem = max(rem, 0.0)
            rem = KiSq_to_Lbsft2(rem)
            fuel_loadings.append(rem)

        inverse_key = np.argsort(self.key)
        fuel_loadings = np.array(fuel_loadings)[inverse_key]

        return fuel_loadings

    def get_updated_fuel_loading(self):
        fuel_loadings = []

        for m in range(1, self.number + 1):
            rem = 0.0
            tf = 0.0

            for n in range(m + 1):
                mn = self.loc(m, n)
                to = self.tout[mn]
                tf = max(to, tf)
                wd = self.wo[mn]
                rem = rem + wd

            if tf < self.ti:
                # set to amount consumed in flaming front
                consumed = (self.fistart * self.ti) / 18608
                rem = self.wdry[m - 1] - consumed
                rem = max(rem, 0.0)
                rem = KiSq_to_Lbsft2(rem)
                fuel_loadings.append(rem)
            else:
                rem = KiSq_to_Lbsft2(rem)
                fuel_loadings.append(rem)

        inverse_key = np.argsort(self.key)
        fuel_loadings = np.array(fuel_loadings)[inverse_key]

        return fuel_loadings

    def fire_intensity(self):
        sum = 0
        noduffsum = 0
        wnoduffsum = 0

        if (self.gd_Fudge1 != 0.0):
            self.wodot[0] = self.gd_Fudge1
            self.gd_Fudge1 = 0.0
        if (self.gd_Fudge2 != 0.0):
            self.wodot[1] = self.gd_Fudge2
            self.gd_Fudge2 = 0.0

        for k in range(1, self.number + 1):
            wdotk = 0
            wnoduff = 0
            for l in range(k + 1):
                kl = self.loc(k, l)
                wdotk += self.wodot[kl]

            term = (1.0 - self.ash[k - 1]) * self.htval[k - 1] * wdotk * 1e-3
            ark = self.area[k - 1]
            if (ark > t_small):
                self.fint[k - 1] = term / ark - term

            else:
                self.fint[k - 1] = 0.0

            k0 = self.loc(k, 0)

            self.Smoldering[k - 1] = self.wodot[k0]
            wnoduff = wdotk - self.Smoldering[k-1]
            noduffterm = (1.0 - self.ash[k - 1]) * self.htval[k - 1] * wnoduff * 1e-3
            if wnoduff > 0.0:
                fracf = wnoduff / wdotk
                test = fracf * self.fint[k - 1]

            else:
                test = 0.0

            if test > (FintSwitch / ark - FintSwitch):
                self.Flaming[k - 1] += wnoduff
            else:
                self.Smoldering[k - 1] += wnoduff
            
            sum += term
            noduffsum += noduffterm
            wnoduffsum += wnoduff

        return sum

    def TempF(self, q, r):
        test = 1e3
        err = 1e-4
        aa = 20.0

        term = r / (aa * q)
        rlast = r

        while test >= err:
            den = 1.0 + term * (rlast + 1.0) * (rlast * rlast + 1.0)
            rnext = 0.5 * (rlast + 1.0 + r / den)
            test = abs(rnext - rlast)
            if (test < err):
                tempf = rnext * self.tamb
                break
        
            rlast = rnext

        return tempf
