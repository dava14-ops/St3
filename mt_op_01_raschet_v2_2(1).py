#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""МТ-ОП-01, версия 2.2.

Воспроизводимый инженерный скрининг исходного теплового баланса,
отдельная оценка теплоотвода жидкостного жилета методом ε-NTU,
расчёт точки росы и упрощённая оценка теплопритоков через изоляцию трубки.

Ограничения:
* скрипт НЕ реализует PHS по ISO 7933:2023 и не рассчитывает Dlim;
* не прогнозирует потоотделение, водопотерю, температуру ядра тела
  или безопасную продолжительность воздействия;
* мощность жилета рассчитывается отдельно и не вычитается из скрининга;
* сценарии и параметры теплообмена являются проектными допущениями.

Для смены теплоносителя в функции vest_Q следует задавать его фактическую
удельную теплоёмкость cp [Дж/(кг·К)]. Зависимости: стандартная библиотека Python.

Изменения v2.0 относительно исходного скрипта:
* исправлена зависимость конвективных дыхательных потерь C_res;
* согласованы коэффициенты h_c и h_r при определении I_a;
* использовано соотношение Льюиса LR = 16.7 К/кПа;
* давление насыщенного пара определяется одной формулой Магнуса;
* удалены неподтверждённые оценки потоотделения и времени до +1 К;
* удалена несвязанная с измерениями одноконтурная поправка G_s.

Изменения v2.2 относительно v2.1 (базовые результаты табл. 2, 4, 5 не менялись):
* вариант tsk=33 °C включён в анализ чувствительности наравне с остальными
  (в статье — строка табл. 3); добавлено U, требуемое по (7) для дефицитов
  D при tsk=33 °C (70,4 и 76,6 Вт);
* добавлена корреляция Черчилля–Чу для горизонтального цилиндра и расчёт
  суммарного h (конвекция + излучение) как функции ΔT;
* outer_radius_equal_to_bare_loss_m при h >= k/r1 возвращает None
  (решения нет: изоляция снижает теплоприток при любой толщине), а не падает;
* защита от pa >= p_sk,s (испарение невозможно, Emax = 0, флаг condensation_on_skin);
* контрольные проверки охватывают tsk=33 °C (в т.ч. сценарий B), смещения
  Tsurf - Tin, расход, h(ΔT);
* удалён неиспользуемый импорт; inner_radius_m переименован в tube_outer_radius_m.

Изменения v2.1 относительно v2.0 (базовые результаты не менялись):
* добавлены варианты чувствительности по скорости воздуха (0,5 и 1,0 м/с)
  и справочный вариант tsk=33 °C (вне табл. 3 статьи);
* добавлен расчёт требуемого U по (7) для сценария D и варианты
  Tsurf - Tin = 0 / 3 / 5 К;
* добавлена чувствительность мощности жилета к расходу (0,5 / 1 / 2 л/мин);
* теплоприток через трубку считается для h = 4,5 и 10 Вт/(м²·К);
* уточнена терминология (соотношение Льюиса) и расширены контрольные проверки.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. Расчётный оператор и константы
# ---------------------------------------------------------------------------
MASS_KG = 75.0
HEIGHT_M = 1.80
A_D = 0.202 * MASS_KG**0.425 * HEIGHT_M**0.725  # Du Bois, м²

EPSILON_CLOTHING = 0.97
SIGMA = 5.67e-8                 # Вт/(м²·К⁴)
RADIATING_AREA_FACTOR = 0.70    # принято для положения сидя
LEWIS_K_PER_KPA = 16.7         # соотношение Льюиса LR, К/кПа
CP_WATER = 4180.0              # Дж/(кг·К)
WATER_MASS_FLOW_1_L_MIN = 1.0 / 60.0  # кг/с, rho воды принято 1000 кг/м³

MAGNUS_A = 17.62
MAGNUS_B_C = 243.12
MAGNUS_C_KPA = 0.6112

SCENARIOS: Dict[str, Dict[str, float]] = {
    "A": {"ta": 25.0, "tr": 27.0, "rh": 50.0},
    "B": {"ta": 31.0, "tr": 38.0, "rh": 45.0},
    "C": {"ta": 36.0, "tr": 45.0, "rh": 35.0},
    "D": {"ta": 33.0, "tr": 33.0, "rh": 70.0},
}


# ---------------------------------------------------------------------------
# 2. Психрометрия
# ---------------------------------------------------------------------------
def saturation_vapor_pressure_kpa(temp_c: float) -> float:
    """Давление насыщенного водяного пара по формуле Магнуса, кПа."""
    if temp_c <= -MAGNUS_B_C:
        raise ValueError("Температура вне области применимости формулы Магнуса")
    return MAGNUS_C_KPA * math.exp(
        MAGNUS_A * temp_c / (MAGNUS_B_C + temp_c)
    )


def vapor_pressure_kpa(temp_c: float, rh_percent: float) -> float:
    """Парциальное давление водяного пара в воздухе, кПа."""
    if not 0.0 < rh_percent <= 100.0:
        raise ValueError("Относительная влажность должна быть в диапазоне (0; 100] %")
    return rh_percent / 100.0 * saturation_vapor_pressure_kpa(temp_c)


def dew_point_c(temp_c: float, rh_percent: float) -> float:
    """Температура точки росы по формуле Магнуса, °C."""
    if not 0.0 < rh_percent <= 100.0:
        raise ValueError("Относительная влажность должна быть в диапазоне (0; 100] %")
    gamma = math.log(rh_percent / 100.0) + (
        MAGNUS_A * temp_c / (MAGNUS_B_C + temp_c)
    )
    denominator = MAGNUS_A - gamma
    if abs(denominator) < 1e-12:
        raise ValueError("Не удалось определить точку росы для заданных условий")
    return MAGNUS_B_C * gamma / denominator


# ---------------------------------------------------------------------------
# 3. Стационарный инженерный скрининг (все первичные потоки — Вт/м²)
# ---------------------------------------------------------------------------
def _heat_transfer_coefficients(
    tcl_c: float, ta_c: float, tr_c: float, v_ms: float
) -> Tuple[float, float]:
    """Коэффициенты конвекции и линеаризованного излучения, Вт/(м²·К)."""
    if v_ms < 0.0:
        raise ValueError("Скорость воздуха не может быть отрицательной")
    hc = max(
        2.38 * abs(tcl_c - ta_c) ** 0.25,
        3.5 + 5.2 * v_ms,
        8.7 * v_ms**0.6,
    )
    mean_radiant_kelvin = 273.15 + 0.5 * (tcl_c + tr_c)
    if mean_radiant_kelvin <= 0:
        raise ValueError("Некорректная абсолютная температура для расчёта излучения")
    hr = (
        4.0
        * EPSILON_CLOTHING
        * SIGMA
        * RADIATING_AREA_FACTOR
        * mean_radiant_kelvin**3
    )
    return hc, hr


def screening(
    ta: float,
    tr: float,
    rh: float,
    *,
    metabolic_w_m2: float = 87.0,
    clothing_clo: float = 0.6,
    air_speed_ms: float = 0.2,
    moisture_permeability_index: float = 0.38,
    max_skin_wettedness: float = 0.85,
    skin_temperature_c: float = 35.0,
    external_work_w_m2: float = 0.0,
) -> Dict[str, float]:
    """Выполняет стационарный скрининг.

    Значения с суффиксом ``_w_m2`` являются удельными потоками.
    Значения с суффиксом ``_w`` — полные мощности для площади A_D.
    E_max_lim учитывает принятое ограничение max_skin_wettedness.
    """
    if metabolic_w_m2 <= 0.0:
        raise ValueError("Метаболическая мощность должна быть положительной")
    if clothing_clo <= 0.0:
        raise ValueError("Теплоизоляция одежды должна быть положительной")
    if moisture_permeability_index <= 0.0:
        raise ValueError("Индекс паропроницаемости должен быть положительным")
    if not 0.0 < max_skin_wettedness <= 1.0:
        raise ValueError("max_skin_wettedness должен быть в диапазоне (0; 1]")
    if external_work_w_m2 < 0.0:
        raise ValueError("Внешняя механическая работа не может быть отрицательной")

    pa = vapor_pressure_kpa(ta, rh)
    i_cl = clothing_clo * 0.155  # м²·К/Вт
    f_cl = 1.0 + 0.3 * clothing_clo

    # Итерационное определение температуры наружной поверхности одежды.
    t_cl = 0.5 * (skin_temperature_c + ta)
    for _ in range(500):
        hc, hr = _heat_transfer_coefficients(t_cl, ta, tr, air_speed_ms)
        t_cl_new = (
            skin_temperature_c / i_cl + f_cl * (hc * ta + hr * tr)
        ) / (1.0 / i_cl + f_cl * (hc + hr))
        if abs(t_cl_new - t_cl) < 1e-9:
            t_cl = t_cl_new
            break
        t_cl = 0.5 * (t_cl + t_cl_new)
    else:
        raise RuntimeError("Итерация температуры одежды не сошлась")

    # Повторно вычисляем hc и hr именно для сошедшейся температуры t_cl.
    hc, hr = _heat_transfer_coefficients(t_cl, ta, tr, air_speed_ms)

    # Сухой теплообмен через одежду, Вт/м².
    dry_heat_w_m2 = (skin_temperature_c - t_cl) / i_cl

    # Дыхательные потери: сначала температура выдыхаемого воздуха,
    # затем конвективная и испарительная составляющие, Вт/м².
    expired_air_temp_c = 28.56 + 0.115 * ta + 0.641 * pa
    c_res_w_m2 = 0.001516 * metabolic_w_m2 * (expired_air_temp_c - ta)
    e_res_w_m2 = 0.00127 * metabolic_w_m2 * (
        59.34 + 0.53 * ta - 11.63 * pa
    )

    # Требуемое испарение при условном стационарном балансе S = 0, Вт/м².
    e_req_w_m2 = (
        metabolic_w_m2
        - external_work_w_m2
        - dry_heat_w_m2
        - c_res_w_m2
        - e_res_w_m2
    )

    # Сопротивление воздушного пограничного слоя и суммарное
    # испарительное сопротивление. Ia использует те же hc и hr,
    # которые рассчитаны по температуре наружной поверхности одежды.
    i_a = 1.0 / (hc + hr)  # м²·К/Вт
    r_e_t = (i_cl + i_a / f_cl) / (
        moisture_permeability_index * LEWIS_K_PER_KPA
    )  # м²·кПа/Вт

    p_skin_sat = saturation_vapor_pressure_kpa(skin_temperature_c)
    # Если pa >= p_sk,s, испарение с кожи невозможно: Emax = 0.
    vapor_gradient_ok = p_skin_sat > pa
    e_max_lim_w_m2 = (
        max_skin_wettedness * (p_skin_sat - pa) / r_e_t if vapor_gradient_ok else 0.0
    )

    ratio_req_to_max = (
        e_req_w_m2 / e_max_lim_w_m2 if e_max_lim_w_m2 > 0.0 else math.inf
    )
    deficit_w_m2 = max(0.0, e_req_w_m2 - e_max_lim_w_m2)

    return {
        "pa_kpa": pa,
        "dew_point_c": dew_point_c(ta, rh),
        "t_cl_c": t_cl,
        "dry_heat_w_m2": dry_heat_w_m2,
        "c_res_w_m2": c_res_w_m2,
        "e_res_w_m2": e_res_w_m2,
        "e_req_w_m2": e_req_w_m2,
        "e_max_lim_w_m2": e_max_lim_w_m2,
        "ratio_req_to_max": ratio_req_to_max,
        "dry_heat_w": dry_heat_w_m2 * A_D,
        "e_req_w": e_req_w_m2 * A_D,
        "e_max_lim_w": e_max_lim_w_m2 * A_D,
        "deficit_w": deficit_w_m2 * A_D,
        "c_res_w": c_res_w_m2 * A_D,
        "e_res_w": e_res_w_m2 * A_D,
        "vapor_gradient_ok": float(vapor_gradient_ok),
        "i_a_m2_k_w": i_a,
        "r_e_t_m2_kpa_w": r_e_t,
    }


# ---------------------------------------------------------------------------
# 4. Жилет: одноконтурная ε-NTU оценка
# ---------------------------------------------------------------------------
def vest_heat_removal_w(
    overall_u_w_m2_k: float,
    effective_area_m2: float,
    coolant_inlet_c: float,
    local_skin_c: float,
    mass_flow_kg_s: float,
    cp_j_kg_k: float = CP_WATER,
) -> float:
    """Идеализированная мощность передачи тепла теплоносителю, Вт."""
    if overall_u_w_m2_k < 0.0 or effective_area_m2 < 0.0:
        raise ValueError("U и эффективная площадь не могут быть отрицательными")
    if mass_flow_kg_s <= 0.0 or cp_j_kg_k <= 0.0:
        raise ValueError("Расход и теплоёмкость должны быть положительными")
    if local_skin_c <= coolant_inlet_c:
        return 0.0
    capacity_rate = mass_flow_kg_s * cp_j_kg_k
    ua = overall_u_w_m2_k * effective_area_m2
    ntu = ua / capacity_rate
    effectiveness = 1.0 - math.exp(-ntu)
    return effectiveness * capacity_rate * (local_skin_c - coolant_inlet_c)


def ua_required_w_k(
    target_heat_w: float,
    coolant_inlet_c: float,
    local_skin_c: float,
    mass_flow_kg_s: float,
    cp_j_kg_k: float = CP_WATER,
) -> float:
    """Требуемое UA для заданной мощности при модели с постоянной t_skin, Вт/К."""
    if target_heat_w < 0.0:
        raise ValueError("Целевая мощность не может быть отрицательной")
    if mass_flow_kg_s <= 0.0 or cp_j_kg_k <= 0.0:
        raise ValueError("Расход и теплоёмкость должны быть положительными")
    if local_skin_c <= coolant_inlet_c:
        raise ValueError("Локальная температура кожи должна превышать Tin")
    capacity_rate = mass_flow_kg_s * cp_j_kg_k
    maximum = capacity_rate * (local_skin_c - coolant_inlet_c)
    if target_heat_w == 0.0:
        return 0.0
    effectiveness = target_heat_w / maximum
    if not 0.0 <= effectiveness < 1.0:
        return math.inf
    return -capacity_rate * math.log1p(-effectiveness)


# ---------------------------------------------------------------------------
# 5. Конденсационные ограничения
# ---------------------------------------------------------------------------
def minimum_coolant_temperature_c(
    ta: float,
    rh: float,
    *,
    safety_margin_k: float = 2.0,
    surface_offset_k: float = 0.0,
) -> float:
    """Tin,min для заданной модели поверхности Tsurf = Tin + offset."""
    if safety_margin_k < 0.0:
        raise ValueError("Запас по точке росы не может быть отрицательным")
    return dew_point_c(ta, rh) + safety_margin_k - surface_offset_k


# ---------------------------------------------------------------------------
# 6. Теплопритоки через цилиндрическую изоляцию трубки
# ---------------------------------------------------------------------------
def air_properties(temp_k: float) -> Tuple[float, float, float, float]:
    """Плотность, динамическая вязкость, теплопроводность, cp воздуха (101,3 кПа)."""
    rho = 101325.0 / (287.05 * temp_k)
    mu = 1.458e-6 * temp_k**1.5 / (temp_k + 110.4)
    k = 2.64638e-3 * temp_k**1.5 / (temp_k + 245.4 * 10.0 ** (-12.0 / temp_k))
    return rho, mu, k, 1007.0


def churchill_chu_h_w_m2_k(diameter_m: float, air_c: float, delta_t_k: float) -> float:
    """Конвективный h для горизонтального цилиндра (Черчилль–Чу), Вт/(м²·К).

    Свойства воздуха берутся при плёночной температуре air_c - ΔT/2
    (поверхность трубки холоднее воздуха на ΔT)."""
    if diameter_m <= 0.0 or delta_t_k <= 0.0:
        raise ValueError("Диаметр и ΔT должны быть положительными")
    t_film = air_c - 0.5 * delta_t_k + 273.15
    rho, mu, k, cp = air_properties(t_film)
    pr, nu = mu * cp / k, mu / rho
    ra = 9.81 / t_film * delta_t_k * diameter_m**3 * pr / nu**2
    nu_d = (0.6 + 0.387 * ra ** (1.0 / 6.0)
            / (1.0 + (0.559 / pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)) ** 2
    return nu_d * k / diameter_m


def total_outside_h_w_m2_k(
    diameter_m: float, air_c: float, delta_t_k: float, emissivity: float = 0.9
) -> Tuple[float, float, float]:
    """(h_конв, h_изл, h_сумм), Вт/(м²·К); излучение линеаризовано при T_film."""
    h_c = churchill_chu_h_w_m2_k(diameter_m, air_c, delta_t_k)
    t_film = air_c - 0.5 * delta_t_k + 273.15
    h_r = 4.0 * emissivity * SIGMA * t_film**3
    return h_c, h_r, h_c + h_r


def heat_gain_per_length_w_m(
    tube_outer_radius_m: float,
    outer_radius_m: float,
    insulation_k_w_m_k: float,
    outside_h_w_m2_k: float,
    delta_t_k: float,
) -> float:
    """Стационарный теплоприток на длину, Вт/м.

    tube_outer_radius_m — наружный радиус трубки до изоляции;
    outer_radius_m — наружный радиус изоляции (равен радиусу трубки при её отсутствии).
    Сопротивление стенки трубки и внутренней конвекции не учитывается.
    """
    if tube_outer_radius_m <= 0.0 or outer_radius_m < tube_outer_radius_m:
        raise ValueError("Некорректные радиусы трубки/изоляции")
    if insulation_k_w_m_k <= 0.0 or outside_h_w_m2_k <= 0.0:
        raise ValueError("Теплопроводность и коэффициент теплоотдачи должны быть > 0")
    if delta_t_k < 0.0:
        raise ValueError("Разность температур должна быть неотрицательной")
    if math.isclose(outer_radius_m, tube_outer_radius_m, rel_tol=0.0, abs_tol=1e-15):
        resistance = 1.0 / (
            outside_h_w_m2_k * 2.0 * math.pi * tube_outer_radius_m
        )
    else:
        r_cond = math.log(outer_radius_m / tube_outer_radius_m) / (
            2.0 * math.pi * insulation_k_w_m_k
        )
        r_conv = 1.0 / (
            outside_h_w_m2_k * 2.0 * math.pi * outer_radius_m
        )
        resistance = r_cond + r_conv
    return delta_t_k / resistance


def outer_radius_equal_to_bare_loss_m(
    tube_outer_radius_m: float,
    insulation_k_w_m_k: float,
    outside_h_w_m2_k: float,
) -> Optional[float]:
    """Радиус изолированной трубки, при котором теплоприток равен голой трубке.

    Если h >= k/r1 (критический радиус k/h не больше радиуса трубки), изоляция
    снижает теплоприток при любой толщине и такого радиуса нет: возвращается None."""
    if outside_h_w_m2_k >= insulation_k_w_m_k / tube_outer_radius_m:
        return None
    bare = heat_gain_per_length_w_m(
        tube_outer_radius_m, tube_outer_radius_m,
        insulation_k_w_m_k, outside_h_w_m2_k, 1.0
    )
    lo = tube_outer_radius_m * (1.0 + 1e-8)
    hi = max(tube_outer_radius_m * 1000.0, 0.1)

    def difference(radius: float) -> float:
        return heat_gain_per_length_w_m(
            tube_outer_radius_m, radius,
            insulation_k_w_m_k, outside_h_w_m2_k, 1.0
        ) - bare

    f_lo, f_hi = difference(lo), difference(hi)
    if f_lo * f_hi > 0.0:
        raise RuntimeError("Не удалось заключить корень в интервал бисекции")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = difference(mid)
        if abs(f_mid) < 1e-12 or (hi - lo) < 1e-12:
            return mid
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 7. Вывод результатов и контроль воспроизводимости
# ---------------------------------------------------------------------------
def _print_screening() -> None:
    print("\nСтационарный инженерный скрининг (M=87 Вт/м², tsk=35 °C, clo=0,6)")
    print("Сцен. | pa, кПа | tdew, °C | C+R, Вт | Ereq, Вт | Emax_lim, Вт | Ereq/Emax_lim | Dev, Вт")
    for name, scenario in SCENARIOS.items():
        r = screening(**scenario)
        print(
            f"{name:>4} | {r['pa_kpa']:7.2f} | {r['dew_point_c']:8.1f} | "
            f"{r['dry_heat_w']:8.0f} | {r['e_req_w']:8.0f} | "
            f"{r['e_max_lim_w']:12.0f} | {r['ratio_req_to_max']:13.2f} | "
            f"{r['deficit_w']:7.0f}"
        )


def _print_sensitivity() -> None:
    print("\nАнализ чувствительности (табл. 3 статьи): дефицит Dev, Вт")
    print("Вариант                  |  A |   B |   C |   D")
    cases = [
        ("Базовый", {}),
        ("tsk=33 °C", {"skin_temperature_c": 33.0}),
        ("tsk=34 °C", {"skin_temperature_c": 34.0}),
        ("tsk=36 °C", {"skin_temperature_c": 36.0}),
        ("M=105 Вт/м²", {"metabolic_w_m2": 105.0}),
        ("clo=1,0", {"clothing_clo": 1.0}),
        ("v=0,5 м/с", {"air_speed_ms": 0.5}),
        ("v=1,0 м/с", {"air_speed_ms": 1.0}),
    ]
    for label, kwargs in cases:
        values = [screening(**s, **kwargs)["deficit_w"] for s in SCENARIOS.values()]
        print(f"{label:24s} | " + " | ".join(f"{v:3.1f}" for v in values))


def _print_vest() -> None:
    mdot = WATER_MASS_FLOW_1_L_MIN
    print("\nТеплоотвод жилета Q, Вт (Aeff=0,5 м², 1 л/мин, tsk,loc=33 °C)")
    tin_values = (12.0, 16.0, 18.0, 20.0, 24.0, 28.0)
    for u in (5.0, 10.0, 15.0):
        values = [vest_heat_removal_w(u, 0.5, tin, 33.0, mdot) for tin in tin_values]
        print(f"U={u:g}: " + ", ".join(f"{v:.1f}" for v in values))
    print("Расход, л/мин -> Q, Вт (U=10, Tin=18 °C, tsk,loc=33 °C):")
    for flow in (0.5, 1.0, 2.0):
        q = vest_heat_removal_w(10.0, 0.5, 18.0, 33.0, mdot * flow)
        print(f"  {flow:g}: {q:.1f}")


def _print_table6_extra() -> None:
    mdot = WATER_MASS_FLOW_1_L_MIN
    print("\nТребуемое U по (7) в сценарии D (Tin,min под тканью, Aeff=0,5 м², 1 л/мин)")
    tin_d = minimum_coolant_temperature_c(33.0, 70.0, surface_offset_k=3.0)
    for q_target in (22.7, 55.9, 70.4, 76.6):
        u_req = ua_required_w_k(q_target, tin_d, 33.0, mdot) / 0.5
        print(f"  Q*={q_target:g} Вт: U={u_req:.1f} Вт/(м²·К)")
    print("Мощность в D при разных Tsurf - Tin (U=5, 10, 15):")
    for offset in (0.0, 3.0, 5.0):
        tin = minimum_coolant_temperature_c(33.0, 70.0, surface_offset_k=offset)
        values = [vest_heat_removal_w(u, 0.5, tin, 33.0, mdot) for u in (5.0, 10.0, 15.0)]
        print(f"  offset={offset:g} К, Tin,min={tin:.1f}: " + ", ".join(f"{v:.1f}" for v in values))
    print("Критерий (11): Qnc (U=10) против Dev базового и при tsk=33 °C")
    for name, sc in SCENARIOS.items():
        tin = minimum_coolant_temperature_c(sc["ta"], sc["rh"], surface_offset_k=3.0)
        q = vest_heat_removal_w(10.0, 0.5, tin, 33.0, mdot)
        d0 = screening(**sc)["deficit_w"]
        d33 = screening(**sc, skin_temperature_c=33.0)["deficit_w"]
        print(f"  {name}: Qnc={q:.1f}; Dev={d0:.1f} -> {'да' if q >= d0 else 'нет'}; "
              f"Dev(tsk=33)={d33:.1f} -> {'да' if q >= d33 else 'нет'}")


def _print_dew_limits() -> None:
    print("\nТочка росы и минимальная температура теплоносителя, °C (запас 2 К)")
    print("Сцен. | tdew | Tin,min открытая | Tin,min под тканью")
    for name, s in SCENARIOS.items():
        dp = dew_point_c(s["ta"], s["rh"])
        bare = minimum_coolant_temperature_c(s["ta"], s["rh"])
        fabric = minimum_coolant_temperature_c(
            s["ta"], s["rh"], surface_offset_k=3.0
        )
        print(f"{name:>4} | {dp:4.1f} | {bare:15.1f} | {fabric:18.1f}")


def _print_pipe() -> None:
    r1, k = 0.004, 0.035
    print("\nТеплоприток через трубку d=8 мм, k=0,035 Вт/(м·К), ΔT=1 К")
    for h in (4.5, 10.0):
        print(f"h={h:g} Вт/(м²·К): k/r1={k/r1:.2f}, rcrit=k/h={k/h*1000:.1f} мм")
        for thickness in (0.0, 0.002, 0.004, 0.006, 0.010, 0.020):
            q = heat_gain_per_length_w_m(r1, r1 + thickness, k, h, 1.0)
            print(f"  изоляция {thickness*1000:4.0f} мм: {q:.3f} Вт/м")
        r_equal = outer_radius_equal_to_bare_loss_m(r1, k, h)
        if r_equal is None:
            print("  уровень голой трубки: изоляция снижает теплоприток при любой толщине")
        else:
            print(f"  r_equal={r_equal*1000:.1f} мм; толщина для уровня голой трубки={(r_equal-r1)*1000:.1f} мм")
    print("\nh по Черчиллю–Чу + излучение (eps=0,9), воздух 33 °C:")
    for dt in (1.0, 5.0, 10.0, 18.0):
        hc, hr, ht = total_outside_h_w_m2_k(2 * r1, 33.0, dt)
        print(f"  ΔT={dt:4.0f} К: h_конв={hc:.2f}, h_изл={hr:.2f}, h_сумм={ht:.2f}")
    print("Магистраль 3 м, ΔT=18 К, h по корреляции для ΔT=18 К (воздух 33 °C):")
    _, _, h18 = total_outside_h_w_m2_k(2 * r1, 33.0, 18.0)
    for thickness in (0.0, 0.006):
        q3 = 3.0 * heat_gain_per_length_w_m(r1, r1 + thickness, k, h18, 18.0)
        print(f"  изоляция {thickness*1000:.0f} мм: {q3:.1f} Вт (h={h18:.1f})")
    print("Магистраль 3 м, ΔT=18 К, h=10 const (расчёт v2.1):")
    for thickness in (0.0, 0.006):
        q3 = 3.0 * heat_gain_per_length_w_m(r1, r1 + thickness, k, 10.0, 18.0)
        print(f"  изоляция {thickness*1000:.0f} мм: {q3:.1f} Вт")
    print("Вызов при h=10 (раньше RuntimeError):", outer_radius_equal_to_bare_loss_m(r1, k, 10.0))


def run_checks() -> None:
    """Проверки контрольных значений, чтобы ловить случайные изменения модели."""
    if not math.isclose(A_D, 1.9378714863, rel_tol=0.0, abs_tol=1e-8):
        raise AssertionError("Изменилась расчётная площадь поверхности тела")
    base = {k: screening(**v) for k, v in SCENARIOS.items()}
    expected_deficits = {"A": 0.0, "B": 0.0, "C": 15.60, "D": 22.72}
    for key, expected in expected_deficits.items():
        if not math.isclose(base[key]["deficit_w"], expected, abs_tol=0.08):
            raise AssertionError(f"Контрольный дефицит сценария {key} не совпал")
    # Проверки порядка величины ε-NTU и конденсационных ограничений.
    q18 = vest_heat_removal_w(10.0, 0.5, 18.0, 33.0, WATER_MASS_FLOW_1_L_MIN)
    if not math.isclose(q18, 72.3, abs_tol=0.3):
        raise AssertionError("Контрольная мощность жилета не совпала")
    if not math.isclose(dew_point_c(33.0, 70.0), 26.8, abs_tol=0.1):
        raise AssertionError("Контрольная точка росы не совпала")
    # v2.1: чувствительность к скорости воздуха и требуемое U в D.
    for key in ("C", "D"):
        if screening(**SCENARIOS[key], air_speed_ms=1.0)["deficit_w"] > 1e-9:
            raise AssertionError("При v=1 м/с дефицит должен отсутствовать")
    tin_d = minimum_coolant_temperature_c(33.0, 70.0, surface_offset_k=3.0)
    u_56 = ua_required_w_k(56.0, tin_d, 33.0, WATER_MASS_FLOW_1_L_MIN) / 0.5
    if not math.isclose(u_56, 16.5, abs_tol=0.1):
        raise AssertionError("Требуемое U для 56 Вт в D не совпало")
    q15 = vest_heat_removal_w(15.0, 0.5, tin_d, 33.0, WATER_MASS_FLOW_1_L_MIN)
    if not math.isclose(q15, 51.2, abs_tol=0.3):
        raise AssertionError("Мощность в D при U=15 не совпала")
    if not (heat_gain_per_length_w_m(0.004, 0.010, 0.035, 10.0, 1.0)
            < heat_gain_per_length_w_m(0.004, 0.004, 0.035, 10.0, 1.0)):
        raise AssertionError("При h=10 изоляция должна снижать теплоприток")
    # v2.2: tsk=33 °C (вариант чувствительности, включая сценарий B).
    d33 = {k: screening(**v, skin_temperature_c=33.0)["deficit_w"] for k, v in SCENARIOS.items()}
    for key, expected in {"A": 0.0, "B": 3.9, "C": 70.4, "D": 76.6}.items():
        if not math.isclose(d33[key], expected, abs_tol=0.1):
            raise AssertionError(f"Дефицит при tsk=33 °C, сценарий {key}, не совпал")
    for target, expected in ((70.4, 21.1), (76.6, 23.1)):
        u_req = ua_required_w_k(target, tin_d, 33.0, WATER_MASS_FLOW_1_L_MIN) / 0.5
        if not math.isclose(u_req, expected, abs_tol=0.1):
            raise AssertionError(f"Требуемое U для {target} Вт в D не совпало")
    for offset, expected in ((0.0, (10.3, 20.3, 29.9)), (5.0, (22.6, 44.4, 65.5))):
        tin = minimum_coolant_temperature_c(33.0, 70.0, surface_offset_k=offset)
        for u, e in zip((5.0, 10.0, 15.0), expected):
            q = vest_heat_removal_w(u, 0.5, tin, 33.0, WATER_MASS_FLOW_1_L_MIN)
            if not math.isclose(q, e, abs_tol=0.15):
                raise AssertionError("Мощность в D при смещении Tsurf не совпала")
    for flow, expected in ((0.5, 69.9), (2.0, 73.7)):
        q = vest_heat_removal_w(10.0, 0.5, 18.0, 33.0, WATER_MASS_FLOW_1_L_MIN * flow)
        if not math.isclose(q, expected, abs_tol=0.15):
            raise AssertionError("Чувствительность к расходу не совпала")
    # v2.2: h(ΔT) по Черчиллю–Чу; при h=10 бисекция не вызывает исключения.
    h1 = total_outside_h_w_m2_k(0.008, 33.0, 1.0)[2]
    h18 = total_outside_h_w_m2_k(0.008, 33.0, 18.0)[2]
    if not (10.0 < h1 < 11.5 and 13.0 < h18 < 14.5):
        raise AssertionError("h(ΔT) по Черчиллю–Чу вне контрольного диапазона")
    if outer_radius_equal_to_bare_loss_m(0.004, 0.035, 10.0) is not None:
        raise AssertionError("При h > k/r1 решения быть не должно")
    r_eq = outer_radius_equal_to_bare_loss_m(0.004, 0.035, 4.5)
    if r_eq is None or not math.isclose(r_eq * 1000.0, 18.3, abs_tol=0.1):
        raise AssertionError("Радиус равного теплопритока при h=4,5 не совпал")
    # v2.2: защита от pa >= p_sk,s.
    humid = screening(ta=40.0, tr=40.0, rh=100.0, skin_temperature_c=35.0)
    if humid["vapor_gradient_ok"] != 0.0 or humid["e_max_lim_w"] != 0.0:
        raise AssertionError("При pa >= p_sk,s Emax должно быть нулевым")


if __name__ == "__main__":
    run_checks()
    print(f"A_D = {A_D:.3f} м²; M_total = {87.0*A_D:.1f} Вт")
    _print_dew_limits()
    _print_screening()
    _print_sensitivity()
    _print_vest()
    _print_table6_extra()
    _print_pipe()
    print("\nКонтрольные проверки: OK")
