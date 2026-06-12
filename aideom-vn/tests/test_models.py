# -*- coding: utf-8 -*-
"""Unit test cho các hàm tính toán thuần của app.py (rubric F2.1: 'có kiểm tra')."""
import importlib.util, sys, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("app", Path(__file__).parent.parent / "app.py")
app = importlib.util.module_from_spec(spec)
sys.modules["app"] = app
spec.loader.exec_module(app)


def test_tfp_inverse():
    """Giai nguoc TFP roi nhan lai phai ra dung Y."""
    m = app.MACRO_DEFAULT
    Y, K, L = m["GDP_trillion_VND"].values, m["K_trillion_VND"].values, m["L_million"].values
    D, AI, H = m["D_digital_pct_GDP"].values, m["AI_thousand_firms"].values, m["H_trained_pct"].values
    A = app.tfp_series(Y, K, L, D, AI, H, .33, .42, .10, .08, .07)
    Y2 = A * K**.33 * L**.42 * D**.10 * AI**.08 * H**.07
    assert np.allclose(Y, Y2)


def test_minmax_bounds():
    x = np.array([3.0, 7.0, 11.0])
    g = app.minmax_norm(x)
    b = app.minmax_norm(x, benefit=False)
    assert g.min() == 0 and g.max() == 1 and b[0] == 1 and b[-1] == 0


def test_topsis_dominant_alternative_wins():
    X = np.array([[10, 10], [5, 5], [1, 1]], float)
    sc = app.topsis(X, [0.5, 0.5], [True, True])
    assert np.argmax(sc) == 0 and np.argmin(sc) == 2


def test_entropy_weights_sum_to_one():
    w = app.entropy_weights(app.REGIONS_DEFAULT[["digital_index_0_100", "gini_coef"]].values)
    assert abs(w.sum() - 1) < 1e-9


def test_b4_fairness_costs_gdp():
    D0 = app.REGIONS_DEFAULT["digital_index_0_100"].values
    _, Zf, sf = app.solve_b4(app.BETA_B4_DEFAULT.values, D0, 50000, 5000, 12000, 12000, .002, .65, True)
    _, Zn, sn = app.solve_b4(app.BETA_B4_DEFAULT.values, D0, 50000, 5000, 12000, 12000, .002, .65, False)
    assert sf == sn == "Optimal" and Zn >= Zf > 0


def test_b4_lambda_070_infeasible():
    """Phat hien quan trong: lambda=0.70 nguyen van de la bat kha thi."""
    D0 = app.REGIONS_DEFAULT["digital_index_0_100"].values
    _, _, stt = app.solve_b4(app.BETA_B4_DEFAULT.values, D0, 50000, 5000, 12000, 12000, .002, .70, True)
    assert stt != "Optimal"


def test_b5_constraints_hold():
    P = app.PROJECTS_DEFAULT
    sel, Z, stt = app.solve_b5(P["Chi phí"].values, P["Lợi ích"].values,
                               P["Chi năm 1-2"].values, 80000, 40000, 7, 11)
    assert stt == "Optimal" and 7 <= len(sel) <= 11
    assert not (0 in sel and 1 in sel)              # C3 loai tru
    assert (7 not in sel) or (11 in sel)            # C4 precedence P8<=P12
    assert (12 not in sel) or (11 in sel)           # C5 precedence P13<=P12
    assert 13 in sel                                # C6 an ninh mang bat buoc
    assert P["Chi phí"].values[sel].sum() <= 80000 + 1e-6


def test_b9_netjob_nonnegative_and_capacity():
    s = app.solve_b9(app.JOBS_DEFAULT, 30000)
    assert s is not None and (s["net"] >= -1e-4).all()
    d1 = app.JOBS_DEFAULT["d1"].values
    assert (s["displaced"] <= d1 * s["xH"] + 1e-4).all()
    assert s["xAI"].sum() + s["xH"].sum() <= 30000 + 1e-4


def test_b10_evpi_nonnegative():
    prob = np.array([0.30, 0.45, 0.20, 0.05])
    bs = app.SCEN10_DEFAULT[["I", "D", "AI", "H"]].values
    _, Z = app.solve_b10_sp(app.BETA10_BASE, bs, prob, stage1_stochastic=True)
    ws = [app.solve_b10_one_scenario(app.BETA10_BASE, bs[s], stage1_stochastic=True)[1]
          for s in range(4)]
    assert float(np.dot(prob, ws)) - Z >= -1e-6     # EVPI >= 0 (tinh chat ly thuyet)


def test_b11_env_episode_terminates():
    env = app.VNEconEnv(budget=1500, seed=1)
    s = env.reset()
    for _ in range(10):
        s, r, done = env.step(1)
    assert done and isinstance(r, float)
