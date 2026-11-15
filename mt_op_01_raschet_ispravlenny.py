"""МТ-ОП-01: воспроизводимый инженерный скрининг, модель жилета ε-NTU и оценка точки росы.

Важно:
- это не реализация PHS по ISO 7933:2023 и не расчёт Dlim;
- результаты относятся только к заданным сценариям и инженерным допущениям;
- активное охлаждение жилетом не вычитается из исходного баланса PHS.
Зависимости: только стандартная библиотека Python.
"""
import math

# ---------- 1. Антропометрия ----------
m, H = 75.0, 1.80
AD = 0.202 * m**0.425 * H**0.725

# ---------- 2. Точка росы (Магнус) ----------
a, b = 17.62, 243.12
def dew(t, rh):
    g = math.log(rh/100.0) + a*t/(b+t)
    return b*g/(a-g)
def pa_kpa(t, rh):               # парциальное давление пара, кПа
    return rh/100.0 * 0.6112*math.exp(a*t/(b+t))
def psat_skin(t):                # ISO 7933: p_sk,s
    return 0.1333*math.exp(18.6686 - 4030.183/(t+235.0))

SC = {"A": dict(ta=25, tr=27, rh=50),
      "B": dict(ta=31, tr=38, rh=45),
      "C": dict(ta=36, tr=45, rh=35),
      "D": dict(ta=33, tr=33, rh=70)}

# ---------- 3. Скрининг стационарного баланса ----------
def screening(ta, tr, rh, M=87.0, clo=0.6, v=0.2, im=0.38, wmax=0.85, tsk=35.0, W=0.0):
    pa = pa_kpa(ta, rh)
    Icl = clo*0.155
    fcl = 1 + 0.3*clo
    eps, sig, Ar = 0.97, 5.67e-8, 0.70
    tcl = 0.5*(tsk+ta)
    for _ in range(200):
        hc = max(2.38*abs(tcl-ta)**0.25, 3.5+5.2*v, 8.7*v**0.6)
        Tm = 273.15 + 0.5*(tcl+tr)
        hr = 4*eps*sig*Ar*Tm**3
        # (tsk-tcl)/Icl = fcl*(hc(tcl-ta)+hr(tcl-tr))
        tcl_new = (tsk/Icl + fcl*(hc*ta+hr*tr)) / (1/Icl + fcl*(hc+hr))
        if abs(tcl_new-tcl) < 1e-6: tcl = tcl_new; break
        tcl = 0.5*(tcl+tcl_new)
    CR = (tsk-tcl)/Icl                              # Вт/м²
    # дыхание (ISO 7933)
    Cres = 0.001*M*(28.56 + 0.115*ta - 0.641*pa)
    Eres = 0.00127*M*(59.34 + 0.53*ta - 11.63*pa)
    Ereq = M - W - CR - Cres - Eres                 # Вт/м²
    # паровое сопротивление
    hc_b = max(2.38*abs(tsk-ta)**0.25, 3.5+5.2*v, 8.7*v**0.6)
    Ia = 1.0/(hc_b + 4*eps*sig*Ar*(273.15+0.5*(tsk+tr))**3)
    Rt = Icl + Ia/fcl
    Ret = Rt/(im*16.5)                              # м²·кПа/Вт
    Emax = wmax*(psat_skin(tsk)-pa)/Ret
    w = Ereq/Emax if Emax > 0 else 1.7
    w = min(max(w, 0.0), 1.7)
    r = (1 - w**2/2) if w <= 1 else (2-w)**2/2
    r = max(r, 0.05)
    Ep = min(max(Ereq,0), Emax)
    SW = Ep/r
    deficit = max(0.0, Ereq-Emax)*AD
    sweat_gh = SW*AD*3600/2426.0
    t38 = (260e3/deficit/3600) if deficit > 0 else float('inf')   # ч, 1 К запаса, c·m=260 кДж/К
    return dict(pa=pa, dew=dew(ta, rh), CR=CR*AD, Ereq=Ereq*AD, Emax=Emax*AD, w=w,
                deficit=deficit, sweat=sweat_gh, water8=sweat_gh*8/1000.0, t38=t38, Cres_Eres=(Cres+Eres)*AD)

# ---------- 4. Жилет: ε-NTU ----------
cpw = 4180.0
def vest_Q(U, A, Tin, tsk, mdot):
    Cf = mdot*cpw; UA = U*A
    eps = 1-math.exp(-UA/Cf)
    return eps*Cf*(tsk-Tin)
def vest_Q_coupled(UA, Tin, tsk0, mdot, Gs):
    Cf = mdot*cpw; eps = 1-math.exp(-UA/Cf)
    return eps*Cf*(tsk0-Tin)/(1+eps*Cf/Gs)
def UA_required(Q, Tin, tsk, mdot):
    Cf = mdot*cpw; e = Q/(Cf*(tsk-Tin))
    return -math.log(1-e)*Cf if e < 1 else float('inf')

# ---------- 5. Труба и критический радиус ----------
def q_per_m(r1, r2, k, h, dT):
    if r2 <= r1: return 2*math.pi*r1*h*dT
    R = math.log(r2/r1)/(2*math.pi*k) + 1/(h*2*math.pi*r2)
    return dT/R

if __name__ == "__main__":
    out = {"AD": AD}
    print("A_D = %.3f м2; M_tot = %.1f Вт" % (AD, 87*AD))
    print("\nТочки росы / допустимая T_in (запас 2 К)")
    for k, s in SC.items():
        d = dew(s['ta'], s['rh'])
        print(k, "tdew=%.1f  bare>=%.1f  cloth>=%.1f  pa=%.2f" % (d, d+2, d+2-3, pa_kpa(s['ta'], s['rh'])))
    print("\nСкрининг, M=87, tsk=35")
    for k, s in SC.items():
        r = screening(**s)
        print(k, {kk: (round(vv,2) if vv != float('inf') else 'inf') for kk, vv in r.items()})
    print("\nЧувствительность: M=105")
    for k, s in SC.items():
        r = screening(**s, M=105.0)
        print(k, "def=%.0f Ereq=%.0f Emax=%.0f t38=%s" % (r['deficit'], r['Ereq'], r['Emax'], r['t38']))
    print("\nЧувствительность: tsk=34 / 36, M=87")
    for tk in (34.0, 36.0):
        for k, s in SC.items():
            r = screening(**s, tsk=tk)
            print(tk, k, "def=%.0f Ereq=%.0f Emax=%.0f" % (r['deficit'], r['Ereq'], r['Emax']))
    print("\nЧувствительность: clo=1.0, M=87")
    for k, s in SC.items():
        r = screening(**s, clo=1.0)
        print(k, "def=%.0f Ereq=%.0f Emax=%.0f CR=%.0f" % (r['deficit'], r['Ereq'], r['Emax'], r['CR']))
    print("\nЖилет Q(U, Tin), tsk=33, A=0.5, 1 л/мин")
    md = 1/60.0
    for U in (5, 10, 15):
        print(U, [round(vest_Q(U, 0.5, T, 33.0, md), 1) for T in (12, 16, 18, 20, 24, 28)])
    for tsk in (30, 32, 34):
        print("tsk", tsk, "U=10:", [round(vest_Q(10, 0.5, T, tsk, md), 1) for T in (12, 16, 18, 20, 24, 28)])
    print("\nТребуемое UA")
    for (n, T, Q) in (("B", 20, 30), ("C", 20, 60), ("D", 29, 35)):
        ua = UA_required(Q, T, 33.0, md)
        print(n, T, Q, round(ua, 2), round(ua/0.5, 1))
    print("\nСвязанная модель (Gs)")
    for Gs in (10, 20, 40):
        print("Gs", Gs, [round(vest_Q_coupled(5.0, T, 34.0, md, Gs), 1) for T in (12,16,20,24)])
    print("\nТрубка d=8 мм, k=0.035, h=4.5")
    r1 = 0.004
    for t2 in (0, 0.002, 0.004, 0.006, 0.010, 0.020):
        print("тол-на %.0f мм: q=%.3f Вт/(м·К)" % (t2*1000, q_per_m(r1, r1+t2, 0.035, 4.5, 1.0)))
    bare = q_per_m(r1, r1, 0.035, 4.5, 1.0)
    f = lambda r2: q_per_m(r1, r2, 0.035, 4.5, 1.0) - bare
    lo, hi = 0.0079, 0.5
    flo = f(lo)
    for _ in range(100):  # бисекция: не требует SciPy
        mid = 0.5*(lo+hi)
        fm = f(mid)
        if flo*fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    req = 0.5*(lo+hi)
    print("r_кр=%.2f мм; r_экв=%.1f мм (толщина %.1f мм)" % (0.035/4.5*1000, req*1000, (req-r1)*1000))
    print("\nЛёд: ", [round(P*8*3600/376e3, 2) for P in (50,100,200)])
