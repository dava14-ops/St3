#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""МТ-ОП-01 v2.2: построение рис. 1–3 статьи (matplotlib).
Запуск: python3 mt_op_01_make_figures.py  (рядом должен лежать mt_op_01_raschet_v2_2.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import mt_op_01_raschet_v2_2 as m

for f in font_manager.findSystemFonts():
    if "LiberationSerif" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})
MD = m.WATER_MASS_FLOW_1_L_MIN
BW = ("#222222", "#666666", "#999999")


def fig1(path):
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    tin = [12 + 0.25 * i for i in range(0, 65)]
    for u, ls in zip((5, 10, 15), ("--", "-", ":")):
        ax.plot(tin, [m.vest_heat_removal_w(u, 0.5, t, 33.0, MD) for t in tin],
                ls, color="black", label=f"U = {u} Вт/(м²·К)")
    for name, col in zip("ABCD", ("#1b7837", "#2166ac", "#b2182b", "#762a83")):
        sc = m.SCENARIOS[name]
        t = m.minimum_coolant_temperature_c(sc["ta"], sc["rh"], surface_offset_k=3.0)
        ax.axvline(t, color=col, lw=1.2, alpha=0.9)
        ax.text(t + 0.15, 148, name, color=col, fontsize=11, fontweight="bold")
    ax.set_xlabel("Температура теплоносителя на входе T_in, °C")
    ax.set_ylabel("Мощность жилета Q, Вт")
    ax.set_xlim(12, 28); ax.set_ylim(0, 155)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig2(path):
    base = dict(u=10.0, a=0.5, tin=18.0, tsk=33.0, flow=1.0)
    def q(**kw):
        p = {**base, **kw}
        return m.vest_heat_removal_w(p["u"], p["a"], p["tin"], p["tsk"], MD * p["flow"])
    rows = [("U, Вт/(м²·К):  5 | 15", q(u=5.0), q(u=15.0)),
            ("A_eff, м²:  0,4 | 0,6", q(a=0.4), q(a=0.6)),
            ("T_in, °C:  20 | 16", q(tin=20.0), q(tin=16.0)),
            ("t_sk,loc, °C:  31 | 35", q(tsk=31.0), q(tsk=35.0)),
            ("Расход, л/мин:  0,5 | 2", q(flow=0.5), q(flow=2.0))]
    q0 = q()
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    for i, (lab, lo, hi) in enumerate(rows[::-1]):
        ax.barh(i, lo - q0, left=q0, color="#9e9e9e", height=0.55,
                label="первое значение пары" if i == 0 else None)
        ax.barh(i, hi - q0, left=q0, color="#2166ac", height=0.55,
                label="второе значение пары" if i == 0 else None)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.axvline(q0, color="black", lw=1)
    ax.text(q0 + 0.5, len(rows) - 0.45, f"база {q0:.0f} Вт", fontsize=9)
    ax.set_xlabel("Мощность жилета Q, Вт"); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig3(path):
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    rh = [20 + i for i in range(0, 81)]
    for ta, ls in zip((25, 31, 33, 36), ("-", "--", "-.", ":")):
        ax.plot(rh, [m.minimum_coolant_temperature_c(ta, r, surface_offset_k=3.0) for r in rh],
                ls, color="black", label=f"t_a = {ta} °C")
    for name, sc in m.SCENARIOS.items():
        t = m.minimum_coolant_temperature_c(sc["ta"], sc["rh"], surface_offset_k=3.0)
        ax.plot(sc["rh"], t, "o", color="#b2182b")
        ax.annotate(name, (sc["rh"], t), textcoords="offset points", xytext=(5, -12), fontsize=11)
    ax.set_xlabel("Относительная влажность RH, %")
    ax.set_ylabel("Минимальная T_in, °C")
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


if __name__ == "__main__":
    fig1("fig1.png"); fig2("fig2.png"); fig3("fig3.png")
    print("fig1.png, fig2.png, fig3.png")
