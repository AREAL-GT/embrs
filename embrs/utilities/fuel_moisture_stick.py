import numpy as np

Aks = 2.0e-13
Ap = 0.000772
Aw = 0.8
Hfs = 0.99
Kelvin = 273.2
Pi = 3.14159
Pr = 0.7
Sc = 0.58
Smv = 94.743
St = 72.8
Tcd = 6.0
Tcn = 3.0
Thdiff = 8.0
Wl = 0.0023
Srf = 14.82052
Wsf = 4.60517
Hrd = 0.116171
Hrn = 0.112467
Sir = 0.0714285
Scr = 0.285714

def Fms_CreateParameters(fms):
    pass

def Fms_Diffusivity(fms):
    pass

def Fms_Create1Hour(name):
    return Fms(
        name,           # Name for the new stick instance.
        11,             # Number of calculation nodes.
        0.20,           # Stick radius (cm)
        25.0,           # Stick length (cm)
        0.40,           # Stick density (gm/cm3)
        0.004,          # Moisture computation time step (h)
        0.05,           # Diffusivity computation time step (h)
        0.0218,         # Barometric pressure (cal/cm3)
        0.85,           # Max local moisture due to rain (g/g)
        2.5,            # Planar heat transfer (cal/cm2-h-C)
        0.065,          # Surface mass transfer - adsorption ((cm3/cm2)/h)
        0.08,           # Surface mass transfer - desorption ((cm3/cm2)/h)
        5.0,            # Runoff factor during initial rainfall observation
        10.0,           # Runoff factor after initial rainfall observation
        0.006,          # Storm transition value (cm/h)
        0.10            # Water film contribution to moisture content (gm/gm)
    )

def Fms_Create10Hour(name):
    return Fms(
        name,           # Name for the new stick instance.
        11,             # Number of calculation nodes.
        0.64,           # Stick radius (cm)
        50.0,           # Stick length (cm)
        0.40,           # Stick density (gm/cm3)
        0.10,           # Moisture computation time step (h)
        0.25,           # Diffusivity computation time step (h)
        0.0218,         # Barometric pressure (cal/cm3)
        0.60,           # Max local moisture due to rain (g/g)
        0.38,           # Planar heat transfer (cal/cm2-h-C)
        0.02,           # Surface mass transfer - adsorption ((cm3/cm2)/h)
        0.06,           # Surface mass transfer - desorption ((cm3/cm2)/h)
        0.55,           # Runoff factor during initial rainfall observation
        0.15,           # Runoff factor after initial rainfall observation
        0.05,           # Storm transition value (cm/h)
        0.05            # Water film contribution to moisture content (gm/gm)
    )

def Fms_Create100Hour(name):
    return Fms(
        name,           # Name for the new stick instance.
        11,             # Number of calculation nodes.
        2.0,            # Stick radius (cm)
        100.0,          # Stick length (cm)
        0.40,           # Stick density (gm/cm3)
        0.20,           # Moisture computation time step (h)
        0.25,           # Diffusivity computation time step (h)
        0.0218,         # Barometric pressure (cal/cm3)
        0.40,           # Max local moisture due to rain (g/g)
        0.30,           # Planar heat transfer (cal/cm2-h-C)
        0.012,          # Surface mass transfer - adsorption ((cm3/cm2)/h)
        0.06,           # Surface mass transfer - desorption ((cm3/cm2)/h)
        0.05,           # Runoff factor during initial rainfall observation
        0.12,           # Runoff factor after initial rainfall observation
        5.0,            # Storm transition value (cm/h)
        0.005           # Water film contribution to moisture content (gm/gm)
    )


def Fms_Create1000Hour(name):
    return Fms(
        name,           # Name for the new stick instance.
        11,             # Number of calculation nodes.
        6.4,            # Stick radius (cm)
        200.0,          # Stick length (cm)
        0.40,           # Stick density (gm/cm3)
        0.25,           # Moisture computation time step (h)
        0.25,           # Diffusivity computation time step (h)
        0.0218,         # Barometric pressure (cal/cm3)
        0.32,           # Max local moisture due to rain (g/g)
        0.12,           # Planar heat transfer (cal/cm2-h-C)
        0.00001,        # Surface mass transfer - adsorption ((cm3/cm2)/h)
        0.06,           # Surface mass transfer - desorption ((cm3/cm2)/h)
        0.07,           # Runoff factor during initial rainfall observation
        0.12,           # Runoff factor after initial rainfall observation
        7.5,            # Storm transition value (cm/h)
        0.003           # Water film contribution to moisture content (gm/gm)
    )

class Fms:
    def __init__(self, name, nodes, radius, length, density, mdt, ddt, pressure,
                    wmx, hc, stca, stcd, rai0, rai1, stv, wfilmk):
        self.name = name
        self.n = nodes
        self.a = radius
        self.al = length
        self.dp = density
        self.mdt = mdt
        self.ddt = ddt
        self.bp = pressure
        self.wmx = wmx
        self.hc = hc
        self.stca = stca
        self.stcd = stcd
        self.rai0 = rai0
        self.rai1 = rai1
        self.stv = stv
        self.wfilmk = wfilmk

        self.x = [0.0] * self.n
        self.d = [0.0] * self.n
        self.t = [0.0] * self.n
        self.s = [0.0] * self.n
        self.w = [0.0] * self.n
        self.v = [0.0] * self.n

        self.dx = self.a / (self.n - 1)
        self.wmax = (1.0 / self.dp) - (1.0 / 1.53)
        self.init = 0
        self.updates = 0
        self.state = None

        self.hwf = 0.622 * self.hc * pow((Pr / Sc), 0.667)
        self.amlf = self.hwf / (0.24 * self.dp * self.a)
        rcav = 0.5 * Aw * Wl
        self.capf = 3600.0 * Pi * St * rcav * rcav / (16.0 * self.a * self.a * self.al * self.dp)
        self.sf = 3600.0 * self.mdt / (2.0 * self.dx * self.dp)
        self.vf = St / (self.dp * Wl * Scr)
        self.dx_2 = self.dx * 2.0
        self.mdt_2 = self.mdt * 2.0

        self.ta0 = 0.0
        self.ta1 = 0.0
        self.ha0 = 0.0
        self.ha1 = 0.0
        self.sv0 = 0.0
        self.sv1 = 0.0
        self.rc0 = 0.0
        self.rc1 = 0.0
        self.ra0 = 0.0
        self.ra1 = 0.0
        self.hf = 0.0
        self.wfilm = 0.0
        self.wsa = 0.0
        self.sem = 0.0
        self.jdate = 0.0

    def Fms_CreateParameters(self):
        self.x = [0.0] * self.n
        self.d = [0.0] * self.n
        self.t = [0.0] * self.n
        self.s = [0.0] * self.n
        self.w = [0.0] * self.n
        self.v = [0.0] * self.n

        self.dx = self.a / (self.n - 1)
        self.wmax = (1.0 / self.dp) - (1.0 / 1.53)

        for i in range(self.n):
            self.x[i] = self.a - (self.dx * i)
            self.d[i] = 0.0
            self.t[i] = 20.0
            self.s[i] = 0.5 * self.wmx
            self.w[i] = 0.5 * self.wmx
            self.v[i] = 0.0

        ro = self.a
        ri = ro - 0.5 * self.dx
        a2 = self.a * self.a
        self.v[0] = (ro * ro - ri * ri) / a2
        vwt = self.v[0]
        for i in range(1, self.n - 1):
            ro = ri
            ri = ro - self.dx
            self.v[i] = (ro * ro - ri * ri) / a2
            vwt += self.v[i]
        self.v[self.n - 1] = ri * ri / a2
        vwt += self.v[self.n - 1]

        self.init = 0
        self.updates = 0
        self.state = None

        self.hwf = 0.622 * self.hc * pow((Pr / Sc), 0.667)
        self.amlf = self.hwf / (0.24 * self.dp * self.a)
        rcav = 0.5 * Aw * Wl
        self.capf = 3600.0 * Pi * St * rcav * rcav / (16.0 * self.a * self.a * self.al * self.dp)
        self.sf = 3600.0 * self.mdt / (2.0 * self.dx * self.dp)
        self.vf = St / (self.dp * Wl * Scr)
        self.dx_2 = self.dx * 2.0
        self.mdt_2 = self.mdt * 2.0

        return 0

    def Fms_MeanMoisture(self):
        wec = self.w[0]
        wei = self.dx / (3.0 * self.a)
        for i in range(1, self.n - 1, 2):
            wea = 4.0 * self.w[i]
            web = 2.0 * self.w[i + 1]
            if (i + 1) == (self.n - 1):
                web = self.w[self.n - 1]
            wec += web + wea
        wbr = wei * wec
        wbr += self.wfilm
        if wbr > self.wmx:
            wbr = self.wmx
        return wbr

    def Fms_MeanWtdMoisture(self):
        wbr = 0.0
        for i in range(self.n):
            wbr += self.w[i] * self.v[i]
        wbr += self.wfilm
        if wbr > self.wmx:
            wbr = self.wmx
        return wbr

    def Fms_MeanWtdTemperature(self):
        wbr = 0.0
        for i in range(self.n):
            wbr += self.t[i] * self.v[i]
        return wbr

    def Fms_Diffusivity(self):
        for i in range(self.n):
            tk = self.t[i] + 273.2
            qv = 13550.0 - 10.22 * tk
            cpv = 7.22 + 0.002374 * tk + 2.67e-07 * tk * tk
            dv = 0.22 * 3600.0 * (0.0242 / self.bp) * pow((tk / 273.2), 1.75)
            ps1 = 0.0000239 * np.exp(20.58 - (5205.0 / tk))
            c1 = 0.1617 - 0.001419 * self.t[i]
            c2 = 0.4657 + 0.003578 * self.t[i]
            if self.w[i] < self.wsa:
                wc = self.w[i]
                dhdm = (1.0 - self.hf) * pow(-np.log(1.0 - self.hf), (1.0 - c2)) / (c1 * c2)
            else:
                wc = self.wsa
                dhdm = (1.0 - Hfs) * pow(Wsf, (1.0 - c2)) / (c1 * c2)
            daw = 1.3 - 0.64 * wc
            svaw = 1.0 / daw
            vfaw = svaw * wc / (0.685 + svaw * wc)
            vfcw = (0.685 + svaw * wc) / ((1.0 / self.dp) + svaw * wc)
            rfcw = 1.0 - np.sqrt(1.0 - vfcw)
            fac = 1.0 / (rfcw * vfcw)
            con = 1.0 / (2.0 - vfaw)
            qw = 5040.0 * np.exp(-14.0 * wc)
            e = (qv + qw - cpv * tk) / 1.2
            dvpr = 18.0 * 0.016 * (1.0 - vfcw) * dv * ps1 * dhdm / (self.dp * 1.987 * tk)
            self.d[i] = dvpr + 3600.0 * 0.0985 * con * fac * np.exp(-e / (1.987 * tk))

    def Fms_Initialize(self, ta, ha, sr, rc, ti, hi, wi):
        self.ta0 = self.ta1 = ta
        self.ha0 = self.ha1 = ha
        self.sv0 = self.sv1 = sr / Smv
        self.rc0 = self.rc1 = rc
        self.ra0 = self.ra1 = 0.0

        self.hf = hi
        self.wfilm = 0.0
        for i in range(self.n):
            self.t[i] = ti
            self.w[i] = wi
            self.s[i] = 0.0
        self.wsa = wi + 0.1
        self.Fms_Diffusivity()
        self.init = 1

    def Fms_InitializeAt(self, year, month, day, hour, minute, second, millisecond, ta, ha, sr, rc, ti, hi, wi):
        # Set the initial modified Julian date and milliseconds of the day.
        self.jdate = self.CDT_JulianDate(year, month, day, hour, minute, second, millisecond)
        # Now initialize the stick's environmental variables.
        self.Fms_Initialize(ta, ha, sr, rc, ti, hi, wi)

    def CDT_JulianDate(self, year, month, day, hour, minute, second, millisecond):
        jdate = 10000 * year + 100 * month + day
        if month <= 2:
            year -= 1
            month += 12
        a = 0
        b = 0
        if jdate >= 15821015.0:
            a = year // 100
            b = 2 - a + a // 4
        c = int(365.25 * year)
        d = int(30.6001 * (month + 1))
        jdate = b + c + d + day + self.CDT_DecimalDay(hour, minute, second, millisecond) + 1720994.5
        return jdate

    def CDT_DecimalDay(self, hour, minute, second, millisecond):
        return (millisecond + 1000 * second + 60000 * minute + 3600000 * hour) / 86400000.0

    def Fms_MeanMoisture(self):
        wec = self.w[0]
        wei = self.dx / (3.0 * self.a)

        for i in range(1, self.n - 1, 2):
            wea = 4.0 * self.w[i]
            web = 2.0 * self.w[i + 1]
            if (i + 1) == (self.n - 1):
                web = self.w[self.n - 1]
            wec += web + wea
        wbr = wei * wec

        # Add water film.
        wbr += self.wfilm
        if wbr > self.wmx:
            wbr = self.wmx
        return wbr

    def Fms_MeanWtdMoisture(self):
        wbr = 0.0
        for i in range(self.n):
            wbr += self.w[i] * self.v[i]
        wbr += self.wfilm
        if wbr > self.wmx:
            wbr = self.wmx
        return wbr

    def Fms_MeanWtdTemperature(self):
        wbr = 0.0
        for i in range(self.n):
            wbr += self.t[i] * self.v[i]
        return wbr

    def Fms_UpdateAt(self, year, month, day, hour, minute, second, millisecond, aC, hFtn, sW, rcumCm):
        # If stick is not initialized, do so using environmental variables.
        if not self.init:
            self.Fms_InitializeAt(year, month, day, hour, minute, second, millisecond, aC, hFtn, sW, rcumCm, aC, hFtn, 0.5 * self.wmx)

        # Get the current modified Julian date and milliseconds of the day.
        last = self.jdate
        self.jdate = self.CDT_JulianDate(year, month, day, hour, minute, second, millisecond)

        # Determine elapsed hours to nearest second and call Fms_Update().
        sec = int(86400.0 * (self.jdate - last))
        eHr = sec / 3600.0
        self.Fms_Update(eHr, aC, hFtn, sW, rcumCm)

    def Fms_Update(self, eHr, aC, hFtn, sW, rcumCm):
        self.updates += 1

        if rcumCm < self.ra1:
            self.rc1 = rcumCm
            self.ra0 = 0.0
            return
        if hFtn < 0.001 or hFtn > 1.0:
            return
        if aC < -40.0 or aC > 50.0:
            return
        if sW < 0.0 or sW > 2000.0:
            return

        self.ta0 = self.ta1
        self.ha0 = self.ha1
        self.sv0 = self.sv1
        self.rc0 = self.rc1
        self.ra0 = self.ra1

        self.ta1 = aC
        self.ha1 = hFtn
        self.sv1 = sW / Smv
        self.rc1 = rcumCm

        self.ra1 = (self.rc1 - self.rc0) / Pi

        if eHr < 0.0000027:
            return

        if self.ra1 > 0.00001 and self.ra0 < 0.00001:
            rai = self.rai0 * (1.0 - np.exp(-100.0 * self.ra1))
            if self.ha1 < self.ha0:
                rai *= 0.15
        else:
            rai = self.rai1 * self.ra1 / eHr
        rai *= self.mdt

        ddtNext = self.ddt

        for tt in np.arange(self.mdt, eHr + self.mdt, self.mdt):
            tfract = tt / eHr
            ta = self.ta0 + (self.ta1 - self.ta0) * tfract
            ha = self.ha0 + (self.ha1 - self.ha0) * tfract
            sv = self.sv0 + (self.sv1 - self.sv0) * tfract

            fsc = 0.07 * sv
            tka = ta + Kelvin
            tdw = 5205.0 / ((5205.0 / tka) - np.log(ha))
            tdp = tdw - Kelvin

            if fsc < 0.000001:
                tsk = Tcn + Kelvin
                hr = Hrn
                sr = 0
            else:
                tsk = Tcd + Kelvin
                hr = Hrd
                sr = Srf * fsc

            psa = 0.0000239 * np.exp(20.58 - (5205.0 / tka))
            pa = ha * psa

            tfd = ta + (sr - hr * (ta - tsk + Kelvin)) / (hr + self.hc)
            qv = 13550.0 - 10.22 * (tfd + Kelvin)
            qw = 5040.0 * np.exp(-14.0 * self.w[0])
            hw = (self.hwf * Ap / 0.24) * qv / 18.0

            self.t[0] = tfd - (hw * (tfd - ta) / (hr + self.hc + hw))
            tkf = self.t[0] + Kelvin

            c1 = 0.1617 - 0.001419 * self.t[0]
            c2 = 0.4657 + 0.003578 * self.t[0]

            self.wsa = c1 * pow(Wsf, c2)
            wdiff = self.wmax - self.wsa

            ps1 = 0.0000239 * np.exp(20.58 - (5205.0 / tkf))
            p1 = pa + Ap * self.bp * (qv / (qv + qw)) * (tka - tkf)
            if p1 < 0.000001:
                p1 = 0.000001

            self.hf = min(p1 / ps1, Hfs)
            hf_log = -np.log(1.0 - self.hf)
            self.sem = c1 * pow(hf_log, c2)

            if self.ra1 > 0.0:
                self.t[0] = tfd
                self.hf = Hfs
                if self.ra1 >= self.stv:
                    self.state = 6  # FMS_STATE_Rainstorm
                    self.wfilm = self.wfilmk
                    self.w[0] = self.wmx
                else:
                    self.state = 5  # FMS_STATE_Rainfall
                    self.wfilm = 0.0
                    self.w[0] += rai
                    if self.w[0] > self.wmx:
                        self.w[0] = self.wmx
                    self.s[0] = (self.w[0] - self.wsa) / wdiff
                    if self.s[0] < 0.0:
                        self.s[0] = 0.0
            else:
                self.wfilm = 0.0
                if self.w[0] > self.wsa:
                    p1 = ps1
                    self.hf = Hfs
                    psd = 0.0000239 * np.exp(20.58 - (5205.0 / tdw))
                    if self.t[0] <= tdp and p1 > psd:
                        aml = 0.0
                    else:
                        aml = self.amlf * (ps1 - psd) / self.bp
                    oldw = self.w[0]
                    self.w[0] -= aml * self.mdt_2
                    if aml > 0.0:
                        gnu = 0.00439 + 0.00000177 * pow((338.76 - tkf), 2.1237)
                        self.w[0] -= self.mdt * self.capf / gnu
                    if self.w[0] > self.wmx:
                        self.w[0] = self.wmx
                    if self.w[0] > oldw:
                        self.state = 3  # FMS_STATE_Condensation
                        self.s[0] = (self.w[0] - self.wsa) / wdiff
                    elif self.w[0] == oldw:
                        self.state = 7  # FMS_STATE_Stagnation
                    else:
                        self.state = 4  # FMS_STATE_Evaporation
                        self.s[0] = (self.w[0] - self.wsa) / wdiff
                        if self.s[0] < 0.0:
                            self.s[0] = 0.0
                elif self.t[0] <= tdp:
                    self.state = 3  # FMS_STATE_Condensation
                    psd = 0.0000239 * np.exp(20.58 - (5205.0 / tdw))
                    if p1 > psd:
                        aml = 0.0
                    else:
                        aml = self.amlf * (p1 - psd) / self.bp
                    self.w[0] -= aml * self.mdt_2
                    self.s[0] = (self.w[0] - self.wsa) / wdiff
                    if self.s[0] < 0.0:
                        self.s[0] = 0.0
                else:
                    if self.w[0] != self.sem:
                        if self.w[0] > self.sem:
                            self.state = 2  # FMS_STATE_Desorption
                            bi = self.stcd * self.dx / self.d[0]
                        else:
                            self.state = 1  # FMS_STATE_Adsorption
                            bi = self.stca * self.dx / self.d[0]
                        self.w[0] = (self.w[1] + bi * self.sem) / (1.0 + bi)
                        self.s[0] = 0.0

            wold = self.w.copy()
            sold = self.s.copy()
            told = self.t.copy()
            v = [Thdiff * x for x in self.x]
            o = [0.0] * self.n
            g = [0.0] * self.n

            if self.state != 7:  # FMS_STATE_Stagnation
                do_gnu = True
                for i in range(self.n):
                    par = (self.w[i] - self.wsa) / wdiff
                    self.s[i] = 0.0
                    if par > 0.0:
                        self.s[i] = par
                        if 0.0714285 < par < 0.285714:  # Sir < par < Scr
                            ak = 2.0e-13 * (2.0 * np.sqrt(par / 0.285714) - 1.0)
                            if do_gnu:
                                gnu = 0.00439 + 0.00000177 * pow((338.76 - tkf), 2.1237)
                                do_gnu = False
                            g[i] = (ak / (gnu * wdiff)) * self.x[i] * self.vf * pow((0.285714 / self.s[i]), 1.5)
                    o[i] = self.x[i] * self.d[i]

                for i in range(1, self.n - 1):
                    ae = g[i + 1] / self.dx
                    aw = g[i - 1] / self.dx
                    ar = self.x[i] * self.dx / self.mdt
                    ap = ae + aw + ar
                    self.s[i] = (ae * sold[i + 1] + aw * sold[i - 1] + ar * sold[i]) / ap
                    if self.s[i] > 1.0:
                        self.s[i] = 1.0
                self.s[self.n - 1] = self.s[self.n - 2]

                less = any(self.s[i] < 0.0714285 for i in range(1, self.n - 1))  # Sir

                if not less:
                    for i in range(1, self.n - 1):
                        self.w[i] = self.wsa + self.s[i] * wdiff
                        if self.w[i] > self.wmx:
                            self.w[i] = self.wmx
                else:
                    for i in range(1, self.n - 1):
                        ae = o[i + 1] / self.dx
                        aw = o[i - 1] / self.dx
                        ar = self.x[i] * self.dx / self.mdt
                        ap = ae + aw + ar
                        self.w[i] = (ae * wold[i + 1] + aw * wold[i - 1] + ar * wold[i]) / ap
                        if self.w[i] > self.wmx:
                            self.w[i] = self.wmx
                self.w[self.n - 1] = self.w[self.n - 2]

            for i in range(1, self.n - 1):
                ae = v[i + 1] / self.dx
                aw = v[i - 1] / self.dx
                ar = self.x[i] * self.dx / self.mdt
                ap = ae + aw + ar
                self.t[i] = (ae * told[i + 1] + aw * told[i - 1] + ar * told[i]) / ap
                if self.t[i] > 71.0:
                    self.t[i] = 71.0
            self.t[self.n - 1] = self.t[self.n - 2]

            if (ddtNext - tt) < (0.5 * self.mdt):
                self.Fms_Diffusivity()
                ddtNext += self.ddt


