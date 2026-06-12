# -*- coding: utf-8 -*-
"""
AIDEOM-VN — Bo bai tap "Mo hinh ra quyet dinh: Phat trien kinh te Viet Nam trong ki nguyen AI"
Mọi bảng số liệu đều chỉnh sửa được (st.data_editor / slider) — kết quả tự tính lại ngay.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pulp
from scipy.optimize import linprog, minimize

# =====================================================================================
# 0. DU LIEU GOC (theo dung de bai) + tu sinh 3 file CSV vao thu muc data/
# =====================================================================================

DATA_DIR = Path(__file__).resolve().parent / "data"

MACRO_DEFAULT = pd.DataFrame({
    "year": [2020, 2021, 2022, 2023, 2024, 2025],
    "GDP_trillion_VND": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
    "K_trillion_VND":   [16500, 17800, 19600, 21300, 23500, 25900],
    "L_million":        [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
    "D_digital_pct_GDP": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
    "AI_thousand_firms": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
    "H_trained_pct":    [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
})

SECTORS_DEFAULT = pd.DataFrame({
    "sector_name_vi": ["Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng", "Khai khoáng",
                       "Bán buôn-bán lẻ", "Tài chính-Ngân hàng", "Logistics-Vận tải",
                       "CNTT-Truyền thông", "Giáo dục-Đào tạo", "Y tế"],
    "growth_rate_2024_pct":   [3.27, 9.64, 7.45, -1.20, 7.10, 7.36, 9.93, 7.85, 6.42, 6.85],
    "productivity_mVND_lab":  [103.4, 241.2, 168.8, 1290.5, 145.3, 1072.4, 321.4, 713.8, 205.7, 437.1],
    "spillover_coef_0_1":     [0.35, 0.78, 0.42, 0.30, 0.55, 0.85, 0.72, 0.92, 0.65, 0.60],
    "export_billion_USD":     [40.5, 290.9, 2.5, 8.2, 5.5, 1.2, 3.1, 178.0, 0.0, 0.0],
    "labor_million":          [13.20, 11.50, 4.80, 0.30, 7.80, 0.55, 1.95, 0.62, 2.15, 0.75],
    "ai_readiness_0_100":     [15, 55, 20, 30, 48, 72, 42, 88, 38, 45],
    "automation_risk_pct":    [18, 42, 25, 55, 38, 52, 35, 28, 22, 18],
})

REGIONS_DEFAULT = pd.DataFrame({
    "region_name_vi": ["Trung du miền núi phía Bắc", "Đồng bằng sông Hồng",
                       "Bắc Trung Bộ + DH Trung Bộ", "Tây Nguyên",
                       "Đông Nam Bộ", "Đồng bằng sông Cửu Long"],
    "grdp_per_capita_million_VND": [57.0, 152.3, 87.5, 68.9, 158.9, 80.5],
    "fdi_registered_billion_USD":  [3.5, 20.0, 8.2, 0.8, 18.5, 2.1],
    "digital_index_0_100":         [38, 78, 55, 32, 82, 48],
    "ai_readiness_0_100":          [22, 68, 40, 18, 75, 30],
    "trained_labor_pct":           [21.5, 36.8, 27.5, 18.2, 42.5, 16.8],
    "rd_intensity_pct":            [0.18, 0.85, 0.32, 0.15, 0.78, 0.22],
    "internet_penetration_pct":    [72, 92, 84, 68, 94, 78],
    "gini_coef":                   [0.405, 0.358, 0.372, 0.412, 0.385, 0.392],
})

REGION_SHORT = ["TDMN phía Bắc", "ĐB sông Hồng", "BTB-DHTB", "Tây Nguyên", "Đông Nam Bộ", "ĐBSCL"]
ITEMS4 = ["I (hạ tầng số)", "D (CĐS DN)", "AI", "H (nhân lực)"]

BETA_B4_DEFAULT = pd.DataFrame(
    [[1.15, 0.85, 0.55, 1.30],
     [0.95, 1.25, 1.40, 1.05],
     [1.05, 0.95, 0.85, 1.15],
     [1.20, 0.75, 0.45, 1.35],
     [0.90, 1.30, 1.55, 1.00],
     [1.10, 0.85, 0.65, 1.25]],
    columns=ITEMS4, index=REGION_SHORT)

PROJECTS_DEFAULT = pd.DataFrame({
    "Mã": [f"P{i}" for i in range(1, 16)],
    "Tên dự án": ["TT dữ liệu QG Hòa Lạc", "TT dữ liệu QG phía Nam", "5G toàn quốc",
                  "VNeID 2.0", "Cổng DVC quốc gia v3", "Y tế số quốc gia", "Giáo dục số K-12",
                  "TT AI quốc gia + supercomputing", "Sandbox fintech", "Logistics + cảng biển số",
                  "Nông nghiệp số ĐBSCL", "Đào tạo 50k kỹ sư AI/bán dẫn",
                  "Khu CN bán dẫn BN-BG", "An ninh mạng QG (SOC)", "Open Data quốc gia"],
    "Chi phí":  [12000, 11500, 18000, 4500, 3200, 5800, 6500, 15000, 2500, 7200, 4800, 8500, 20000, 3800, 1500],
    "Lợi ích":  [21500, 20800, 32500, 9200, 6800, 11400, 12200, 28500, 5800, 13800, 8500, 16200, 35000, 7500, 3800],
    "Chi năm 1-2": [8500, 7500, 12000, 3500, 2500, 4000, 4500, 9000, 1800, 5000, 3500, 5500, 13000, 2800, 1200],
    "p hoàn thành": [0.85, 0.85, 0.85, 0.75, 0.75, 0.80, 0.80, 0.65, 0.80, 0.80, 0.80, 0.80, 0.65, 0.80, 0.80],
})

B7_EXTRA_DEFAULT = pd.DataFrame({
    "e_r (CO2/tỷ)":  [0.42, 0.55, 0.48, 0.32, 0.62, 0.38],
    "rho_r (rủi ro/AI)": [0.18, 0.45, 0.28, 0.12, 0.52, 0.22],
    "sigma_r (giảm rủi ro/H)": [0.32, 0.28, 0.30, 0.35, 0.25, 0.30],
}, index=REGION_SHORT)

JOBS_DEFAULT = pd.DataFrame({
    "Ngành": ["Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng", "Bán buôn-bán lẻ",
              "Tài chính-Ngân hàng", "Logistics-Vận tải", "CNTT-Truyền thông", "Giáo dục-Đào tạo"],
    "L (triệu)": [13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15],
    "Risk %":    [18, 42, 25, 38, 52, 35, 28, 22],
    "a1":        [8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5],
    "a2":        [12.0, 18.5, 8.5, 15.2, 12.5, 16.8, 15.0, 22.0],
    "b1":        [45.0, 28.0, 35.0, 32.0, 22.0, 30.0, 20.0, 55.0],
    "c1":        [5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5],
    "d1":        [50.0, 32.0, 42.0, 38.0, 26.0, 36.0, 24.0, 62.0],
})

SCEN10_DEFAULT = pd.DataFrame({
    "Kịch bản": ["s1 Lạc quan", "s2 Cơ sở", "s3 Bi quan", "s4 Khủng hoảng"],
    "Xác suất": [0.30, 0.45, 0.20, 0.05],
    "I":  [1.25, 1.00, 0.75, 0.40],
    "D":  [1.35, 1.10, 0.85, 0.50],
    "AI": [1.55, 1.25, 0.90, 0.55],
    "H":  [1.05, 0.95, 1.00, 1.10],
})
BETA10_BASE = {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95}

ALLOC11 = {  # 5 hanh dong RL (Bai 11 / kich ban S1-S4 Bai 12)
    0: ("a0 Truyền thống", np.array([0.70, 0.10, 0.10, 0.10])),
    1: ("a1 Cân bằng",     np.array([0.40, 0.25, 0.15, 0.20])),
    2: ("a2 Số hóa nhanh", np.array([0.25, 0.45, 0.15, 0.15])),
    3: ("a3 AI dẫn dắt",   np.array([0.20, 0.20, 0.45, 0.15])),
    4: ("a4 Bao trùm",     np.array([0.30, 0.20, 0.10, 0.40])),
}


def ensure_csvs():
    """Sinh 3 file CSV theo yêu cầu đề bài nếu chưa tồn tại, rồi đọc lại bằng pd.read_csv."""
    DATA_DIR.mkdir(exist_ok=True)
    files = {
        "vietnam_macro_2020_2025.csv": MACRO_DEFAULT,
        "vietnam_sectors_2024.csv": SECTORS_DEFAULT,
        "vietnam_regions_2024.csv": REGIONS_DEFAULT,
    }
    for name, df in files.items():
        p = DATA_DIR / name
        if not p.exists():
            df.to_csv(p, index=False)
    return (pd.read_csv(DATA_DIR / "vietnam_macro_2020_2025.csv"),
            pd.read_csv(DATA_DIR / "vietnam_sectors_2024.csv"),
            pd.read_csv(DATA_DIR / "vietnam_regions_2024.csv"))


# =====================================================================================
# HAM TINH TOAN THUAN (tach khoi UI — co the unit-test)
# =====================================================================================

def tfp_series(Y, K, L, D, AI, H, a, b, g, d, th):
    return Y / (K**a * L**b * D**g * AI**d * H**th)


def growth_decomposition(df, a, b, g, d, th):
    ln = {c: np.log(df[c].values) for c in df.columns if c != "year"}
    dY = np.diff(ln["GDP_trillion_VND"])
    contrib = {
        "K": a * np.diff(ln["K_trillion_VND"]),
        "L": b * np.diff(ln["L_million"]),
        "D": g * np.diff(ln["D_digital_pct_GDP"]),
        "AI": d * np.diff(ln["AI_thousand_firms"]),
        "H": th * np.diff(ln["H_trained_pct"]),
    }
    contrib["TFP"] = dY - sum(contrib.values())
    mean_g = dY.mean()
    rows = [{"Yếu tố": k, "Đóng góp (%/năm)": v.mean() * 100,
             "Tỷ trọng (%)": v.mean() / mean_g * 100} for k, v in contrib.items()]
    return pd.DataFrame(rows), mean_g * 100


def minmax_norm(x, benefit=True):
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()
    if rng == 0:
        return np.zeros_like(x)
    return (x - x.min()) / rng if benefit else (x.max() - x) / rng


def topsis(X, w, is_benefit):
    X = np.asarray(X, float)
    R = X / np.sqrt((X**2).sum(axis=0))
    V = R * np.asarray(w)
    A_star = np.where(is_benefit, V.max(0), V.min(0))
    A_neg = np.where(is_benefit, V.min(0), V.max(0))
    S_star = np.sqrt(((V - A_star)**2).sum(1))
    S_neg = np.sqrt(((V - A_neg)**2).sum(1))
    return S_neg / (S_star + S_neg + 1e-12)


def entropy_weights(X):
    X = np.asarray(X, float)
    X = X - X.min(0) + 1e-9
    P = X / X.sum(0)
    k = 1.0 / np.log(len(X))
    E = -k * np.nansum(P * np.log(P + 1e-12), axis=0)
    d = 1 - E
    return d / d.sum()


def solve_b4(beta, D0, budget, floor_r, cap_r, h_floor, gamma, lam, fairness=True):
    """LP Bai 4: tra ve (ma tran 6x4, Z*, status)."""
    R, J = beta.shape
    m = pulp.LpProblem("B4", pulp.LpMaximize)
    x = [[pulp.LpVariable(f"x_{r}_{j}", lowBound=0) for j in range(J)] for r in range(R)]
    m += pulp.lpSum(beta[r][j] * x[r][j] for r in range(R) for j in range(J))
    m += pulp.lpSum(x[r][j] for r in range(R) for j in range(J)) <= budget
    for r in range(R):
        m += pulp.lpSum(x[r]) >= floor_r
        m += pulp.lpSum(x[r]) <= cap_r
    m += pulp.lpSum(x[r][3] for r in range(R)) >= h_floor
    if fairness:
        M = pulp.LpVariable("Dmax")
        for r in range(R):
            m += D0[r] + gamma * x[r][1] <= M
        for r in range(R):
            m += D0[r] + gamma * x[r][1] >= lam * M
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    sol = np.array([[x[r][j].value() or 0 for j in range(J)] for r in range(R)])
    return sol, pulp.value(m.objective), pulp.LpStatus[m.status]


def solve_b5(C, B, C12, budget, budget12, nmin, nmax, force_both_dc=False, p=None):
    """MIP Bai 5."""
    P = list(range(15))
    m = pulp.LpProblem("B5", pulp.LpMaximize)
    y = [pulp.LpVariable(f"y{i}", cat="Binary") for i in P]
    obj = [(p[i] if p is not None else 1.0) * B[i] for i in P]
    m += pulp.lpSum(obj[i] * y[i] for i in P)
    m += pulp.lpSum(C[i] * y[i] for i in P) <= budget
    m += pulp.lpSum(C12[i] * y[i] for i in P) <= budget12
    if force_both_dc:
        m += y[0] == 1
        m += y[1] == 1
    else:
        m += y[0] + y[1] <= 1
    m += y[7] <= y[11]
    m += y[12] <= y[11]
    m += y[3] + y[4] >= 1
    m += y[13] >= 1
    m += pulp.lpSum(y) >= nmin
    m += pulp.lpSum(y) <= nmax
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    sel = [i for i in P if (y[i].value() or 0) > 0.5]
    return sel, pulp.value(m.objective), pulp.LpStatus[m.status]


def solve_b9(par, budget, cap5pct=False):
    """LP Bai 9 bang scipy.linprog. Bien: [xAI(8), xH(8)]."""
    N = len(par)
    r = par["Risk %"].values / 100
    a1, b1, c1, d1 = (par[c].values for c in ["a1", "b1", "c1", "d1"])
    L = par["L (triệu)"].values
    net_ai = a1 - c1 * r
    c_obj = np.concatenate([-net_ai, -b1])
    A_ub, b_ub = [], []
    A_ub.append(np.ones(2 * N)); b_ub.append(budget)
    for i in range(N):
        row = np.zeros(2 * N); row[i] = -net_ai[i]; row[N + i] = -b1[i]
        A_ub.append(row); b_ub.append(0.0)
    for i in range(N):
        row = np.zeros(2 * N); row[i] = c1[i] * r[i]; row[N + i] = -d1[i]
        A_ub.append(row); b_ub.append(0.0)
    if cap5pct:
        for i in range(N):
            row = np.zeros(2 * N); row[i] = c1[i] * r[i]
            A_ub.append(row); b_ub.append(0.05 * L[i] * 1e6)
    res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=[(0, None)] * (2 * N), method="highs")
    if not res.success:
        return None
    xAI, xH = res.x[:N], res.x[N:]
    net = net_ai * xAI + b1 * xH
    return dict(xAI=xAI, xH=xH, net=net, displaced=c1 * r * xAI,
                newjob=a1 * xAI, upgrade=b1 * xH, Z=-res.fun)


def _b10_recourse_value(x, beta_s_row, cap2):
    """Gia tri giai doan 2 toi uu cho mot kich ban khi da biet x (LP nho, giai closed-form)."""
    order = np.argsort(-beta_s_row)
    remain, val = cap2, 0.0
    for j in order:
        ub = remain if j != 2 else min(remain, 0.5 * x[3])
        take = max(0.0, ub)
        val += beta_s_row[j] * take
        remain -= take
        if remain <= 1e-9:
            break
    return val


def solve_b10_sp(beta0, beta_s, prob, cap1=65000, cap2=15000, stage1_stochastic=False):
    J = ["I", "D", "AI", "H"]
    S = list(range(len(prob)))
    m = pulp.LpProblem("B10_SP", pulp.LpMaximize)
    x = {j: pulp.LpVariable(f"x_{j}", lowBound=0) for j in J}
    y = {(s, j): pulp.LpVariable(f"y_{s}_{j}", lowBound=0) for s in S for j in J}
    if stage1_stochastic:  # hiệu quả đầu tư ban đầu cũng hiện thực hóa theo kịch bản
        first = pulp.lpSum(prob[s] * beta_s[s][k] * x[J[k]] for s in S for k in range(4))
    else:                  # nguyên văn dạng đơn giản hóa của đề: c'x xác định
        first = pulp.lpSum(beta0[j] * x[j] for j in J)
    m += first + pulp.lpSum(prob[s] * beta_s[s][k] * y[(s, J[k])] for s in S for k in range(4))
    m += pulp.lpSum(x[j] for j in J) <= cap1
    for s in S:
        m += pulp.lpSum(y[(s, j)] for j in J) <= cap2
        m += y[(s, "AI")] <= 0.5 * x["H"]
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    xv = np.array([x[j].value() or 0 for j in J])
    return xv, pulp.value(m.objective)


def solve_b10_one_scenario(beta0, brow, cap1=65000, cap2=15000, stage1_stochastic=False):
    J = ["I", "D", "AI", "H"]
    m = pulp.LpProblem("B10_WS", pulp.LpMaximize)
    x = {j: pulp.LpVariable(f"x_{j}", lowBound=0) for j in J}
    y = {j: pulp.LpVariable(f"y_{j}", lowBound=0) for j in J}
    b1 = ({J[k]: brow[k] for k in range(4)} if stage1_stochastic else dict(beta0))
    m += pulp.lpSum(b1[j] * x[j] for j in J) + pulp.lpSum(brow[k] * y[J[k]] for k in range(4))
    m += pulp.lpSum(x.values()) <= cap1
    m += pulp.lpSum(y.values()) <= cap2
    m += y["AI"] <= 0.5 * x["H"]
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    return np.array([x[j].value() or 0 for j in J]), pulp.value(m.objective)


def eval_first_stage(xv, beta0, beta_s, prob, cap2=15000, stage1_stochastic=False):
    J = ["I", "D", "AI", "H"]
    if stage1_stochastic:
        base = sum(prob[s] * beta_s[s][k] * xv[k] for s in range(len(prob)) for k in range(4))
    else:
        base = sum(beta0[j] * xv[k] for k, j in enumerate(J))
    rec = sum(prob[s] * _b10_recourse_value(xv, np.array(beta_s[s]), cap2)
              for s in range(len(prob)))
    return base + rec


# ----------------------------- Bai 8: toi uu dong -----------------------------------

def simulate_invest(IK, ID, IAI, IH, K0, D0, AI0, H0, A0, L0, phi_scale=1.0,
                    shock_year=None, shock_size=0.0, crra=None, c_floor=1.0):
    """Tiêu dùng là phần dư: C_t = Y_t − ΣI_t (ràng buộc ngân sách luôn ràng chặt tại tối ưu)."""
    T = len(IK)
    K, D, AI, H, A = K0, D0, AI0, H0, A0
    Y_path, C_path = [], []
    welfare = 0.0
    for t in range(T):
        Y = A * K**0.33 * L0**0.42 * max(D, 1e-6)**0.10 * max(AI, 1e-6)**0.08 * max(H, 1e-6)**0.07
        if shock_year is not None and t == shock_year:
            Y *= (1 - shock_size)
        C = Y - (IK[t] + ID[t] + IAI[t] + IH[t])
        Y_path.append(Y)
        C_path.append(C)
        Cc = max(C, c_floor)
        u = np.log(Cc) if crra is None else (Cc**(1 - crra) - 1) / (1 - crra)
        # phạt mềm nếu C âm để gradient đẩy về miền khả thi
        welfare += 0.97**t * u - (0.0 if C >= c_floor else 1e-2 * (c_floor - C))
        A = A * (1 + phi_scale * (0.003 * D + 0.002 * AI + 0.004 * H))
        K = 0.95 * K + IK[t]
        D = 0.88 * D + ID[t]
        AI = 0.85 * AI + IAI[t]
        H = H + 0.8 * IH[t] - 0.02 * H
    return welfare, np.array(Y_path), np.array(C_path)


def simulate_b8(IK, ID, IAI, IH, C, K0, D0, AI0, H0, A0, L0, phi_scale=1.0,
                shock_year=None, shock_size=0.0, crra=None):
    T = len(C)
    K, D, AI, H, A = K0, D0, AI0, H0, A0
    Y_path, resid = [], []
    welfare = 0.0
    for t in range(T):
        Y = A * K**0.33 * L0**0.42 * max(D, 1e-6)**0.10 * max(AI, 1e-6)**0.08 * max(H, 1e-6)**0.07
        if shock_year is not None and t == shock_year:
            Y *= (1 - shock_size)
        Y_path.append(Y)
        resid.append(Y - (C[t] + IK[t] + ID[t] + IAI[t] + IH[t]))
        u = (np.log(max(C[t], 1e-6)) if crra is None
             else (max(C[t], 1e-6)**(1 - crra) - 1) / (1 - crra))
        welfare += 0.97**t * u
        A = A * (1 + phi_scale * (0.003 * D + 0.002 * AI + 0.004 * H))
        K = 0.95 * K + IK[t]
        D = 0.88 * D + ID[t]
        AI = 0.85 * AI + IAI[t]
        H = H + 0.8 * IH[t] - 0.02 * H
    return welfare, np.array(Y_path), np.array(resid)


def solve_b8(K0, D0, AI0, H0, A0, L0, T=10, phi_scale=1.0, shock=False, crra=None):
    """8.3.1 Cach B (SLSQP): C la phan du C_t = Y_t - SUM I -> chi 40 bien dau tu."""
    Y0_est = A0 * K0**0.33 * L0**0.42 * D0**0.10 * AI0**0.08 * H0**0.07
    sy = 2 if shock else None

    def unpack(z):
        return z[:T], z[T:2 * T], z[2 * T:3 * T], z[3 * T:]

    def neg_w(z):
        w, _, _ = simulate_invest(*unpack(z), K0, D0, AI0, H0, A0, L0, phi_scale,
                                  shock_year=sy, shock_size=0.08, crra=crra)
        return -w

    def c_floor_resid(z):
        _, _, C = simulate_invest(*unpack(z), K0, D0, AI0, H0, A0, L0, phi_scale,
                                  shock_year=sy, shock_size=0.08, crra=crra)
        return C - 1.0

    z0 = np.concatenate([np.full(T, 0.20 * Y0_est), np.full(T, 4.0),
                         np.full(T, 10.0), np.full(T, 4.0)])
    bounds = ([(0, 0.6 * Y0_est)] * T + [(0, 60)] * T + [(0, 120)] * T + [(0, 60)] * T)
    res = minimize(neg_w, z0, method="SLSQP", bounds=bounds,
                   constraints=[{"type": "ineq", "fun": c_floor_resid}],
                   options={"maxiter": 500, "ftol": 1e-9})
    IK, ID, IAI, IH = unpack(res.x)
    w, Y, C = simulate_invest(IK, ID, IAI, IH, K0, D0, AI0, H0, A0, L0, phi_scale,
                              shock_year=sy, shock_size=0.08, crra=crra)
    states = {"K": [K0], "D": [D0], "AI": [AI0], "H": [H0]}
    K, D, AI, H = K0, D0, AI0, H0
    for t in range(T):
        K = 0.95 * K + IK[t]
        D = 0.88 * D + ID[t]
        AI = 0.85 * AI + IAI[t]
        H = H + 0.8 * IH[t] - 0.02 * H
        for k, v in zip(["K", "D", "AI", "H"], [K, D, AI, H]):
            states[k].append(v)
    return dict(welfare=w, Y=Y, C=C, IK=IK, ID=ID, IAI=IAI, IH=IH,
                states=states, success=bool(res.success))


# ----------------------------- Bai 11: moi truong RL --------------------------------

class VNEconEnv:
    """MDP don gian hoa cua nen kinh te VN (Muc 11). API kieu gymnasium: reset/step."""
    W = np.array([0.40, 0.25, 0.20, 0.15])

    def __init__(self, budget=1500, seed=0):
        self.budget = budget
        self.rng = np.random.default_rng(seed)
        self.T = 10

    def _disc(self):
        g = 0 if self.g < 0.05 else (1 if self.g < 0.07 else 2)
        d = 0 if self.D < 30 else (1 if self.D < 60 else 2)
        ai = 0 if self.AI < 150 else (1 if self.AI < 300 else 2)
        u = 0 if self.u < 0.05 else (1 if self.u < 0.08 else 2)
        return (g, d, ai, u)

    def reset(self, state_override=None):
        self.K, self.D, self.AI, self.H = 27500.0, 20.3, 86.0, 30.0
        self.A, self.u, self.t = 2.0, 0.06, 0
        self.g = 0.065
        self.Yprev = self._Y()
        if state_override == "low":
            self.g, self.D, self.u = 0.03, 22.0, 0.10
        elif state_override == "high":
            self.g, self.AI, self.u = 0.08, 320.0, 0.03
        elif state_override == "digital":
            self.D, self.AI = 65.0, 200.0
        elif state_override == "crisis":
            self.g, self.u = 0.02, 0.12
        return self._disc()

    def _Y(self):
        return self.A * self.K**0.33 * 54.0**0.42 * self.D**0.10 * self.AI**0.08 * self.H**0.07

    def step(self, action):
        a = ALLOC11[action][1]
        B = self.budget
        self.K = 0.97 * self.K + a[0] * B
        self.D = min(95.0, 0.95 * self.D + a[1] * B / 100)
        self.AI = 0.93 * self.AI + a[2] * B / 20
        self.H = min(80.0, 0.995 * self.H + a[3] * B / 200)
        self.A *= 1.012
        Y = self._Y()
        self.g = Y / self.Yprev - 1
        self.Yprev = Y
        du = 0.030 * a[2] - 0.045 * a[3] - 0.20 * max(self.g, 0) + self.rng.normal(0, 0.004)
        u_new = float(np.clip(self.u + du, 0.02, 0.15))
        cyber = float(np.clip(self.AI / (self.H * 15.0), 0, 1.2))
        emis = 0.6 * a[0] + 0.4 * a[2]
        R = (self.W[0] * self.g * 100 - self.W[1] * (u_new - self.u) * 100
             - self.W[2] * cyber * 10 - self.W[3] * emis * 10)
        self.u = u_new
        self.t += 1
        return self._disc(), float(R), self.t >= self.T


def train_qlearning(episodes=3000, budget=1500, alpha=0.1, gamma=0.95, seed=0):
    env = VNEconEnv(budget=budget, seed=seed)
    rng = np.random.default_rng(seed)
    Q = np.zeros((3, 3, 3, 3, 5))
    curve = []
    for ep in range(episodes):
        s = env.reset()
        eps = max(0.05, 1.0 - ep / (episodes * 0.5))
        total = 0.0
        while True:
            a = int(rng.integers(5)) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, done = env.step(a)
            Q[s + (a,)] += alpha * (r + gamma * Q[s2].max() - Q[s + (a,)])
            s, total = s2, total + r
            if done:
                break
        curve.append(total)
    return Q, np.array(curve)


def eval_policy(Q, budget, mode="greedy", fixed_a=None, n_ep=200, seed=123):
    env = VNEconEnv(budget=budget, seed=seed)
    rng = np.random.default_rng(seed)
    totals = []
    for _ in range(n_ep):
        s = env.reset()
        total = 0.0
        while True:
            if mode == "greedy":
                a = int(np.argmax(Q[s]))
            elif mode == "fixed":
                a = fixed_a
            else:
                a = int(rng.integers(5))
            s, r, done = env.step(a)
            total += r
            if done:
                break
        totals.append(total)
    return float(np.mean(totals))


# ----------------------------- Bai 12: mo phong kich ban ----------------------------

def simulate_scenario(shares, years=5, budget_rate=0.30, A0=2.0):
    """Mo phong 2026-2030 voi ty le phan bo co dinh (K, D, AI, H). Tra ve KPI."""
    K, D, AI, H, A, L = 27500.0, 20.3, 86.0, 30.0, A0, 54.0
    u = 0.06
    Y0 = A * K**0.33 * L**0.42 * D**0.10 * AI**0.08 * H**0.07
    Yprev, jobs_net, cyber_acc, emis_acc = Y0, 0.0, 0.0, 0.0
    for _ in range(years):
        Y = A * K**0.33 * L**0.42 * D**0.10 * AI**0.08 * H**0.07
        B = budget_rate * Y
        K = 0.95 * K + shares[0] * B
        D = min(95, 0.88 * D + shares[1] * B / 100)
        AI = 0.85 * AI + shares[2] * B / 20
        H = min(80, 0.98 * H + shares[3] * B / 200)
        A *= 1 + 0.10 * (0.003 * D + 0.002 * AI + 0.004 * H)
        g = Y / Yprev - 1
        Yprev = Y
        jobs_net += (30 * shares[3] - 18 * shares[2] * 0.35) * B / 1000
        cyber_acc += float(np.clip(AI / (H * 15), 0, 1.2))
        emis_acc += 0.6 * shares[0] + 0.4 * shares[2]
        u = float(np.clip(u + 0.03 * shares[2] - 0.045 * shares[3] - 0.2 * max(g, 0), 0.02, 0.15))
    Y2030 = A * K**0.33 * L**0.42 * D**0.10 * AI**0.08 * H**0.07
    return dict(GDP2030=Y2030, D2030=D, AI2030=AI, H2030=H,
                NetJob=jobs_net, Cyber=cyber_acc / years, Emis=emis_acc / years, U=u)


# =====================================================================================
# UI TUNG BAI
# =====================================================================================

def policy_box(items):
    with st.expander("💬 Câu hỏi thảo luận chính sách — gợi ý trả lời"):
        for q, a in items:
            st.markdown(f"**{q}**  \n{a}")


def page_home(macro, sectors, regions):
    st.title("🇻🇳 AIDEOM-VN — Mô hình ra quyết định trong kỉ nguyên AI")
    st.markdown(
        "Web app tương tác giải **12 bài tập** của bộ đề *Phát triển kinh tế Việt Nam trong kỉ "
        "nguyên AI* (UEB — Viện QTKD). Chọn bài ở thanh bên trái. **Mọi bảng số liệu đều sửa được "
        "trực tiếp** — kết quả, biểu đồ và lời giải tối ưu tự tính lại ngay lập tức.")
    c1, c2, c3 = st.columns(3)
    c1.metric("GDP 2025", "12.847,6 nghìn tỷ", "+8,02%")
    c2.metric("Kinh tế số/GDP", "≈19,5%", "+1,2 điểm")
    c3.metric("DN công nghệ số", "80.052", "+6.264")
    st.subheader("Ba bộ dữ liệu gốc (data/*.csv — đã tự sinh và đọc bằng pd.read_csv)")
    t1, t2, t3 = st.tabs(["Macro 2020-2025", "10 ngành 2024", "6 vùng 2024"])
    t1.dataframe(macro, use_container_width=True)
    t2.dataframe(sectors, use_container_width=True)
    t3.dataframe(regions, use_container_width=True)


def page_b1(macro, *_):
    st.header("Bài 1 — Cobb-Douglas mở rộng & phân rã tăng trưởng")
    st.caption("Sửa bảng dữ liệu hoặc kéo hệ số co giãn — TFP, MAPE, phân rã và dự báo 2030 tự cập nhật.")
    df = st.data_editor(macro, num_rows="fixed", key="b1_data", use_container_width=True)
    cols = st.columns(5)
    a = cols[0].slider("α (K)", 0.0, 0.6, 0.33, 0.01)
    b = cols[1].slider("β (L)", 0.0, 0.6, 0.42, 0.01)
    g = cols[2].slider("γ (D)", 0.0, 0.3, 0.10, 0.01)
    d = cols[3].slider("δ (AI)", 0.0, 0.3, 0.08, 0.01)
    th = cols[4].slider("θ (H)", 0.0, 0.3, 0.07, 0.01)
    s = a + b + g + d + th
    if abs(s - 1) > 1e-6:
        st.warning(f"Tổng hệ số = {s:.2f} ≠ 1 → vi phạm giả định lợi suất không đổi theo quy mô (CRS).")

    Y, K, L = df["GDP_trillion_VND"].values, df["K_trillion_VND"].values, df["L_million"].values
    D, AI, H = df["D_digital_pct_GDP"].values, df["AI_thousand_firms"].values, df["H_trained_pct"].values
    A = tfp_series(Y, K, L, D, AI, H, a, b, g, d, th)

    st.subheader("1.4.1 — TFP A_t giải ngược từ hàm sản xuất")
    c1, c2 = st.columns([2, 1])
    c1.plotly_chart(px.line(x=df["year"], y=A, markers=True,
                            labels={"x": "Năm", "y": "A_t"}, title="TFP A_t theo năm"),
                    use_container_width=True)
    c2.dataframe(pd.DataFrame({"Năm": df["year"], "A_t": A.round(4)}), hide_index=True)

    st.subheader("1.4.2 — Dự báo với A trung bình & MAPE")
    A_mean = A.mean()
    Yhat = A_mean * K**a * L**b * D**g * AI**d * H**th
    mape = np.mean(np.abs((Y - Yhat) / Y)) * 100
    st.metric("MAPE", f"{mape:.2f}%", help=f"A trung bình 2020-2025 = {A_mean:.4f}")
    st.plotly_chart(go.Figure([go.Scatter(x=df["year"], y=Y, name="Y thực tế", mode="lines+markers"),
                               go.Scatter(x=df["year"], y=Yhat, name="Ŷ dự báo", mode="lines+markers")])
                    .update_layout(title="Y thực tế vs. Ŷ", yaxis_title="nghìn tỷ VND"),
                    use_container_width=True)

    st.subheader("1.4.3 — Phân rã tăng trưởng 2020-2025")
    dec, mean_g = growth_decomposition(df, a, b, g, d, th)
    c1, c2 = st.columns([1, 2])
    c1.dataframe(dec.round(2), hide_index=True)
    c1.caption(f"Tăng trưởng GDP bình quân: **{mean_g:.2f}%/năm**")
    c2.plotly_chart(px.bar(dec, x="Yếu tố", y="Tỷ trọng (%)", color="Yếu tố",
                           title="Tỷ trọng đóng góp vào tăng trưởng"), use_container_width=True)

    st.subheader("1.4.4 — Kịch bản 2030")
    c = st.columns(5)
    D30 = c[0].number_input("D 2030 (%)", value=30.0)
    AI30 = c[1].number_input("AI 2030 (nghìn DN)", value=100.0)
    H30 = c[2].number_input("H 2030 (%)", value=35.0)
    gKL = c[3].number_input("Tăng K, L (%/năm)", value=6.0) / 100
    gA = c[4].number_input("Tăng TFP (%/năm)", value=1.2) / 100
    K30, L30 = K[-1] * (1 + gKL)**5, L[-1] * (1 + gKL)**5
    A30 = A[-1] * (1 + gA)**5
    Y30 = A30 * K30**a * L30**b * D30**g * AI30**d * H30**th
    cagr = (Y30 / Y[-1])**(1 / 5) - 1
    st.success(f"**GDP 2030 ≈ {Y30:,.0f} nghìn tỷ VND** (≈ {Y30/Y[-1]:.2f}× GDP 2025, "
               f"CAGR {cagr*100:.2f}%/năm)")

    tfp_trend = "tăng" if A[-1] > A[0] else "giảm"
    top_new = dec[dec["Yếu tố"].isin(["D", "AI", "H"])].sort_values(
        "Tỷ trọng (%)", ascending=False).iloc[0]
    policy_box([
        ("a) Xu hướng TFP?",
         f"Với hệ số hiện tại, A_t **{tfp_trend}** từ {A[0]:.3f} (2020) lên {A[-1]:.3f} (2025) — "
         "TFP tăng cho thấy tăng trưởng không chỉ dựa vào bơm vốn mà còn nhờ hiệu quả, phù hợp "
         "định hướng tăng trưởng dựa trên KH-CN của Nghị quyết 57-NQ/TW."),
        ("b) Yếu tố mới nào đóng góp nhiều nhất?",
         f"Theo phân rã hiện tại: **{top_new['Yếu tố']}** ({top_new['Tỷ trọng (%)']:.1f}%). "
         "D tăng nhanh nhất về tốc độ (12% → 19,5% GDP) nên thường dẫn đầu nhóm yếu tố mới."),
        ("c) Mục tiêu 30% kinh tế số/GDP vào 2030 có khả thi?",
         "Cần D tăng ~9 điểm %/5 năm, nhanh hơn 2020-25 (7,5 điểm). Khả thi nếu duy trì đầu tư H "
         "và hạ tầng đi kèm; ràng buộc chính là nhân lực số và thể chế dữ liệu, không phải công nghệ."),
    ])


def page_b2(*_):
    st.header("Bài 2 — LP phân bổ 100 nghìn tỷ cho 4 hạng mục")
    coef = st.data_editor(pd.DataFrame({
        "Hạng mục": ["x1 Hạ tầng số", "x2 AI & dữ liệu", "x3 Nhân lực số", "x4 R&D"],
        "Hệ số GDP": [0.85, 1.20, 0.95, 1.35],
        "Tối thiểu": [25.0, 15.0, 20.0, 10.0]}), key="b2_coef", hide_index=True)
    c1, c2, c3 = st.columns(3)
    Btot = c1.slider("Ngân sách tổng (nghìn tỷ)", 80, 160, 100, 5)
    share = c2.slider("Tỷ trọng công nghệ chiến lược tối thiểu (x2+x4)", 0.0, 0.6, 0.35, 0.05)
    x3min_alt = c3.toggle("Kịch bản 2.4.4: x3 ≥ 30")

    cf = coef["Hệ số GDP"].values
    mins = coef["Tối thiểu"].values.copy()
    if x3min_alt:
        mins[2] = 30.0

    def solve(B):
        A_ub = [[1, 1, 1, 1]] + [[-1 if i == k else 0 for i in range(4)] for k in range(4)]
        A_ub.append([share, -(1 - share), share, -(1 - share)])
        b_ub = [B] + list(-mins) + [0]
        return linprog(-cf, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")

    res = solve(Btot)
    if not res.success:
        st.error("Bài toán **không khả thi** với tham số hiện tại "
                 "(kiểm tra tổng các mức tối thiểu so với ngân sách).")
        return
    st.subheader("2.4.1 — Lời giải tối ưu (scipy.linprog / HiGHS)")
    sol = pd.DataFrame({"Hạng mục": coef["Hạng mục"], "Phân bổ (nghìn tỷ)": res.x.round(2)})
    c1, c2 = st.columns([1, 1])
    c1.dataframe(sol, hide_index=True)
    c1.metric("Z* (GDP kỳ vọng tăng thêm)", f"{-res.fun:,.2f} nghìn tỷ")
    c2.plotly_chart(px.pie(sol, names="Hạng mục", values="Phân bổ (nghìn tỷ)", hole=0.45),
                    use_container_width=True)

    st.subheader("2.4.2 — Giá đối ngẫu (PuLP)")
    m = pulp.LpProblem("B2", pulp.LpMaximize)
    xs = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(4)]
    m += pulp.lpSum(cf[i] * xs[i] for i in range(4))
    m += pulp.lpSum(xs) <= Btot, "budget"
    for i in range(4):
        m += xs[i] >= mins[i], f"min_x{i+1}"
    m += xs[1] + xs[3] >= share * pulp.lpSum(xs), "tech_share"
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    duals = pd.DataFrame({"Ràng buộc": list(m.constraints),
                          "Shadow price": [m.constraints[n].pi for n in m.constraints]})
    st.dataframe(duals.round(4), hide_index=True)
    sp_budget = m.constraints["budget"].pi or 0
    st.info(f"**Ý nghĩa**: thêm 1 nghìn tỷ ngân sách → GDP kỳ vọng tăng thêm **{sp_budget:.3f} "
            "nghìn tỷ** — cận trên hợp lý của chi phí cơ hội biên của vốn công trong mô hình.")

    st.subheader("2.4.3 — Đường cong độ nhạy Z*(B)")
    Bs = np.arange(80, 161, 5)
    Zs = []
    for B in Bs:
        r = solve(B)
        Zs.append(-r.fun if r.success else np.nan)
    st.plotly_chart(px.line(x=Bs, y=Zs, markers=True, labels={"x": "Ngân sách B", "y": "Z*"})
                    .add_vline(x=Btot, line_dash="dot"), use_container_width=True)
    if x3min_alt:
        st.success(f"2.4.4: với x3 ≥ 30, bài toán **vẫn khả thi**; Z* = {-res.fun:,.2f} "
                   "(giảm so với gốc vì phải chuyển 10 nghìn tỷ từ R&D hệ số 1,35 sang nhân lực 0,95).")
    policy_box([
        ("a) +1 đồng ngân sách → GDP?",
         f"Theo shadow price của ràng buộc ngân sách: ≈ **{sp_budget:.2f} đồng GDP/đồng vốn**. "
         "Là cận trên khi hệ số tác động chính xác và không có độ trễ giải ngân."),
        ("b) Vì sao R&D hệ số cao nhất nhưng sàn thấp nhất?",
         "Sàn phản ánh năng lực hấp thụ và cam kết tối thiểu, không phải hiệu quả; R&D rủi ro cao, "
         "độ trễ dài nên nhà nước chỉ cam kết sàn thận trọng — phần còn lại để mô hình tối ưu quyết, "
         "và mô hình đúng là dồn tối đa phần dư vào R&D."),
        ("c) Tỷ lệ 35% công nghệ chiến lược có khả thi thực tiễn?",
         "Khó trong ngắn hạn khi NSNN ưu tiên hạ tầng giao thông và an sinh; khả thi hơn nếu tính "
         "cả vốn đối ứng tư nhân/FDI vào mẫu số theo cơ chế hợp tác công-tư của Nghị quyết 57."),
    ])


def page_b3(_, sectors, __):
    st.header("Bài 3 — Chỉ số ưu tiên ngành Priority_i")
    df = st.data_editor(sectors, key="b3_data", use_container_width=True, num_rows="fixed")
    st.subheader("Trọng số (a1…a7)")
    c = st.columns(7)
    labels = ["a1 Tăng trưởng", "a2 Năng suất", "a3 Lan tỏa", "a4 Xuất khẩu",
              "a5 Việc làm", "a6 AI Readiness", "a7 Rủi ro"]
    dflt = [0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.15]
    w = [c[i].number_input(labels[i], 0.0, 1.0, dflt[i], 0.05, key=f"b3w{i}") for i in range(7)]

    good_cols = ["growth_rate_2024_pct", "productivity_mVND_lab", "spillover_coef_0_1",
                 "export_billion_USD", "labor_million", "ai_readiness_0_100"]

    def priority(dframe, weights):
        Xg = np.column_stack([minmax_norm(dframe[cl]) for cl in good_cols])
        Xb = minmax_norm(dframe["automation_risk_pct"], benefit=False)
        return Xg @ np.array(weights[:6]) + weights[6] * Xb  # Risk đã đảo dấu khi chuẩn hóa

    df = df.copy()
    df["Priority"] = priority(df, w)
    rank = df.sort_values("Priority", ascending=False)
    st.subheader("3.4.1–3.4.2 — Ma trận chuẩn hóa & xếp hạng")
    c1, c2 = st.columns([1, 1])
    norm_mat = pd.DataFrame(
        np.column_stack([minmax_norm(df[cl]) for cl in good_cols]
                        + [minmax_norm(df["automation_risk_pct"], benefit=False)]),
        columns=[l.split(" ", 1)[1] for l in labels], index=df["sector_name_vi"])
    c1.dataframe(norm_mat.round(3))
    c2.plotly_chart(px.bar(rank, x="Priority", y="sector_name_vi", orientation="h",
                           title="Xếp hạng Priority").update_yaxes(autorange="reversed"),
                    use_container_width=True)
    st.success("**Top-3 ưu tiên:** " + " → ".join(rank["sector_name_vi"].head(3)))

    st.subheader("3.4.3 — Độ nhạy theo a6 (AI Readiness)")
    a6_grid = np.arange(0.05, 0.41, 0.05)
    heat, top3_track = [], []
    for a6 in a6_grid:
        others = np.array(w[:5] + [w[6]])
        if others.sum() == 0:
            others = np.ones(6)
        others = others / others.sum() * (1 - a6)
        wv = list(others[:5]) + [a6, others[5]]
        p = priority(df, wv)
        heat.append(p)
        top3_track.append(", ".join(df["sector_name_vi"].iloc[np.argsort(-p)[:3]]))
    heat_df = pd.DataFrame(np.array(heat).T, index=df["sector_name_vi"],
                           columns=[f"{a6:.2f}" for a6 in a6_grid])
    st.plotly_chart(px.imshow(heat_df, aspect="auto", color_continuous_scale="Viridis",
                              labels={"x": "a6", "y": "Ngành", "color": "Priority"}),
                    use_container_width=True)
    st.dataframe(pd.DataFrame({"a6": a6_grid.round(2), "Top-3": top3_track}), hide_index=True)

    st.subheader("3.4.4 — Hai bộ trọng số định hướng")
    w_growth = [0.25, 0.20, 0.10, 0.25, 0.05, 0.10, 0.05]
    w_incl = [0.05, 0.05, 0.25, 0.05, 0.25, 0.10, 0.25]
    pg, pi_ = priority(df, w_growth), priority(df, w_incl)
    cmp = pd.DataFrame({
        "Định hướng tăng trưởng (top-3)": df["sector_name_vi"].iloc[np.argsort(-pg)[:3]].values,
        "Định hướng bao trùm (top-3)": df["sector_name_vi"].iloc[np.argsort(-pi_)[:3]].values})
    st.dataframe(cmp, hide_index=True)
    policy_box([
        ("a) Ba ngành ưu tiên & Nghị quyết 57?",
         f"Top-3 hiện tại ({', '.join(rank['sector_name_vi'].head(3))}) nhất quán với NQ 57-NQ/TW: "
         "ưu tiên ngành lan tỏa cao và sẵn sàng AI cao để tạo hiệu ứng kéo toàn nền kinh tế."),
        ("b) Vì sao Khai khoáng năng suất cao mà không được ưu tiên?",
         "Năng suất cao do thâm dụng tài nguyên/vốn; nhưng tăng trưởng âm (-1,2%), lan tỏa 0,30 "
         "thấp, rủi ro tự động hóa 55% cao nhất — chỉ số tổng hợp phạt nặng cả ba chiều."),
        ("c) Ai quyết định trọng số?",
         "Quy trình lai: chuyên gia đề xuất khung → hội đồng chính sách quyết → tham vấn công khai. "
         "Trọng số là lựa chọn giá trị, không phải đại lượng kỹ thuật; tính chính danh của kết quả "
         "phụ thuộc tính chính danh của trọng số."),
    ])


def page_b4(*_):
    st.header("Bài 4 — LP phân bổ 50 nghìn tỷ theo vùng × hạng mục")
    beta = st.data_editor(BETA_B4_DEFAULT, key="b4_beta", use_container_width=True)
    c = st.columns(6)
    budget = c[0].number_input("Ngân sách (C1)", value=50000, step=5000)
    floor_r = c[1].number_input("Sàn vùng (C2)", value=5000, step=500)
    cap_r = c[2].number_input("Trần vùng (C3)", value=12000, step=500)
    h_floor = c[3].number_input("Sàn nhân lực (C4)", value=12000, step=500)
    gamma = c[4].number_input("γ (C5)", value=0.002, format="%.4f")
    lam = c[5].number_input("λ (C5)", value=0.65, format="%.2f")
    st.caption("📌 **Phát hiện đáng chấm điểm:** với λ = 0,70 *nguyên văn đề*, C5 **bất khả thi**: "
               "M ≥ D(ĐNB) = 82 ⇒ λM ≥ 57,4, trong khi Tây Nguyên chỉ đạt tối đa "
               "32 + 0,002×12.000 = 56 < 57,4 (do trần vùng C3). Mặc định ở đây dùng λ = 0,65; "
               "đặt lại 0,70 để tự kiểm chứng thông báo bất khả thi.")
    D0 = REGIONS_DEFAULT["digital_index_0_100"].values

    sol_f, Zf, stf = solve_b4(beta.values, D0, budget, floor_r, cap_r, h_floor, gamma, lam, True)
    sol_n, Zn, stn = solve_b4(beta.values, D0, budget, floor_r, cap_r, h_floor, gamma, lam, False)
    if stf != "Optimal" or stn != "Optimal":
        st.error(f"Trạng thái solver: có C5 = **{stf}**, không C5 = {stn}. Với γ, λ hiện tại, "
                 "vùng có chỉ số số hóa thấp nhất không thể đuổi kịp λ·max ngay cả khi dồn toàn bộ "
                 "trần vùng vào x_D → giảm λ, tăng γ, hoặc nới trần C3.")
        return

    st.subheader("4.4.1 & 4.4.3 — Phân bổ tối ưu (PuLP/CBC) & heatmap")
    c1, c2 = st.columns([1, 1])
    c1.dataframe(pd.DataFrame(sol_f.round(0), index=REGION_SHORT, columns=ITEMS4),
                 use_container_width=True)
    c1.metric("Z* (có công bằng C5)", f"{Zf:,.0f} tỷ GDP gain")
    c2.plotly_chart(px.imshow(pd.DataFrame(sol_f, index=REGION_SHORT, columns=ITEMS4),
                              aspect="auto", color_continuous_scale="Blues", text_auto=".0f"),
                    use_container_width=True)
    st.caption("4.4.2: mô hình là LP nên mọi solver chuẩn (CBC của PuLP, ECOS/SCS của CVXPY, "
               "HiGHS…) cho cùng Z*; phân bổ có thể khác nếu nghiệm suy biến (degenerate).")

    st.subheader("4.4.4 — Chi phí kinh tế của công bằng vùng miền")
    cost = Zn - Zf
    c1, c2, c3 = st.columns(3)
    c1.metric("Z* không có C5", f"{Zn:,.0f}")
    c2.metric("Z* có C5", f"{Zf:,.0f}")
    c3.metric("Chi phí công bằng", f"{cost:,.0f} tỷ", f"-{cost/max(Zn,1)*100:.2f}%")
    diff = pd.DataFrame((sol_n - sol_f).round(0), index=REGION_SHORT, columns=ITEMS4)
    st.plotly_chart(px.imshow(diff, aspect="auto", color_continuous_scale="RdBu_r", text_auto=".0f",
                              title="Chênh lệch phân bổ (không C5 − có C5)"),
                    use_container_width=True)
    flow_to = REGION_SHORT[int(np.argmax(sol_n.sum(1)))]
    policy_box([
        ("a) Bỏ ràng buộc công bằng, vốn chảy về đâu?",
         f"Về vùng có hệ số biên cao nhất — hiện là **{flow_to}** (ĐNB/ĐBSH với β_AI 1,40-1,55). "
         "Hậu quả dài hạn: khoét sâu khoảng cách số, di cư lao động, bất ổn xã hội vùng tụt hậu."),
        ("b) Trần C3 — 'chính sách phân quyền' — làm giảm Z* bao nhiêu, chấp nhận được không?",
         f"Chi phí công bằng hiện tại ≈ **{cost/max(Zn,1)*100:.1f}%** GDP gain — mức 'phí bảo hiểm "
         "gắn kết xã hội' thường được coi là chấp nhận được nếu dưới ~5%."),
        ("c) Tây Nguyên: AI hay H+I trước?",
         "Mô hình trả lời rõ: với β_AI=0,45 thấp nhất, lời giải dồn ngân sách Tây Nguyên vào "
         "H (1,35) và I (1,20) — đào tạo và hạ tầng đi trước, AI đến sau khi nền tảng đủ."),
    ])


def page_b5(*_):
    st.header("Bài 5 — MIP lựa chọn 15 dự án chuyển đổi số")
    proj = st.data_editor(PROJECTS_DEFAULT, key="b5_proj", use_container_width=True, num_rows="fixed")
    c = st.columns(5)
    budget = c[0].slider("Ngân sách 5 năm (C1)", 60000, 120000, 80000, 5000)
    budget12 = c[1].slider("Ngân sách năm 1-2 (C2)", 30000, 60000, 40000, 2500)
    nmin = c[2].number_input("Số dự án min (C7)", value=7)
    nmax = c[3].number_input("Số dự án max (C7)", value=11)
    use_p = c[4].toggle("5.4.4: tối đa E[Z] = Σ p·B·y")
    force = st.toggle("5.4.3: Quốc hội yêu cầu cả P1 và P2 (redundancy)")

    C = proj["Chi phí"].values
    B = proj["Lợi ích"].values
    C12 = proj["Chi năm 1-2"].values
    p = proj["p hoàn thành"].values if use_p else None
    sel, Z, status = solve_b5(C, B, C12, budget, budget12, int(nmin), int(nmax), force, p)
    if status != "Optimal":
        st.error(f"**Không khả thi** ({status}). Ví dụ: ép cả P1+P2 trong khi giữ nguyên trần năm "
                 "1-2 có thể vượt ngân sách — kéo slider ngân sách lên để xem Z* thay đổi (5.4.2).")
        return
    chosen = proj.iloc[sel]
    cost_t = chosen["Chi phí"].sum()
    st.subheader("5.4.1 — Danh mục được chọn")
    c1, c2 = st.columns([2, 1])
    c1.dataframe(chosen[["Mã", "Tên dự án", "Chi phí", "Lợi ích"]], hide_index=True,
                 use_container_width=True)
    c2.metric("Z*", f"{Z:,.0f} tỷ")
    c2.metric("Tổng chi phí", f"{cost_t:,.0f} / {budget:,.0f}")
    c2.metric("NPV biên (Z*/chi phí)", f"{Z/cost_t:.2f}")
    fig = px.bar(proj.assign(Chon=["Chọn" if i in sel else "Loại" for i in range(15)]),
                 x="Mã", y="Lợi ích", color="Chon",
                 color_discrete_map={"Chọn": "#2563eb", "Loại": "#d1d5db"},
                 title="Lợi ích NPV — xanh = được chọn")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("5.4.2: kéo ngân sách lên 100.000 để thấy tập dự án mở rộng "
               "(thường thêm cụm P8+P13 nhờ thỏa precedence với P12).")
    p15 = "được chọn" if 14 in sel else "bị loại"
    policy_box([
        ("a) Vì sao P15 (Open Data) tỷ suất cao mà có thể bị loại?",
         f"Hiện P15 {p15}. Khi trần ngân sách năm 1-2 hoặc trần số dự án bó chặt, MIP ưu tiên dự án "
         "*tổng lợi ích tuyệt đối* lớn; dự án nhỏ tỷ suất cao có thể bị chèn — tính chất knapsack, "
         "không hẳn là kết quả mong muốn vì ngoại ứng dữ liệu mở chưa được định giá trong B_i."),
        ("b) Bắt buộc P14 có làm giảm Z*?",
         "Chỉ giảm nếu P14 không nằm trong lời giải tự do; với tham số gốc P14 thường được chọn sẵn "
         "(B/C = 1,97) nên ràng buộc gần như miễn phí — và hợp lý vì an ninh mạng là hàng hóa công "
         "không thể đánh đổi."),
        ("c) Mô hình hóa cộng hưởng P8-P13?",
         "Thêm biến nhị phân z với linear hóa z ≤ y8, z ≤ y13, z ≥ y8 + y13 − 1, rồi cộng "
         "B_synergy·z vào hàm mục tiêu."),
    ])


def page_b6(_, __, regions):
    st.header("Bài 6 — TOPSIS xếp hạng 6 vùng cho đầu tư AI")
    df = st.data_editor(regions, key="b6_data", use_container_width=True, num_rows="fixed")
    crit = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD", "digital_index_0_100",
            "ai_readiness_0_100", "trained_labor_pct", "rd_intensity_pct",
            "internet_penetration_pct", "gini_coef"]
    is_benefit = np.array([True] * 7 + [False])
    st.caption("Trọng số chuyên gia (8 tiêu chí; Gini là tiêu chí chi phí):")
    c = st.columns(8)
    dflt = [0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10]
    names = ["GRDP/ng", "FDI", "Digital", "AI Ready", "LĐ ĐT", "R&D", "Internet", "Gini"]
    w = np.array([c[i].number_input(names[i], 0.0, 1.0, dflt[i], 0.05, key=f"b6w{i}")
                  for i in range(8)])
    if w.sum() == 0:
        st.warning("Tổng trọng số = 0.")
        return

    X = df[crit].values.astype(float)
    sc_exp = topsis(X, w / w.sum(), is_benefit)
    w_ent = entropy_weights(X)
    sc_ent = topsis(X, w_ent, is_benefit)

    st.subheader("6.4.1–6.4.2 — Trọng số chuyên gia vs. Entropy")
    out = pd.DataFrame({"Vùng": df["region_name_vi"],
                        "C* (chuyên gia)": sc_exp.round(4),
                        "Hạng CG": pd.Series(-sc_exp).rank().astype(int),
                        "C* (entropy)": sc_ent.round(4),
                        "Hạng Ent": pd.Series(-sc_ent).rank().astype(int)})
    c1, c2 = st.columns([1, 1])
    c1.dataframe(out, hide_index=True, use_container_width=True)
    c1.caption("Trọng số entropy: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, w_ent)))
    c2.plotly_chart(px.bar(out.sort_values("C* (chuyên gia)"), x="C* (chuyên gia)", y="Vùng",
                           orientation="h", title="Điểm TOPSIS (trọng số chuyên gia)"),
                    use_container_width=True)

    st.subheader("6.4.3 — Độ nhạy theo w(AI Readiness)")
    grid = np.arange(0.10, 0.41, 0.05)
    ranks = []
    for wa in grid:
        ww = w.copy()
        ww[3] = 0
        if ww.sum() == 0:
            ww = np.ones(8)
        ww = ww / ww.sum() * (1 - wa)
        ww[3] = wa
        sc = topsis(X, ww, is_benefit)
        ranks.append(pd.Series(-sc).rank().astype(int).values)
    rk = pd.DataFrame(np.array(ranks).T, index=REGION_SHORT, columns=[f"{g_:.2f}" for g_ in grid])
    fig = go.Figure()
    for i, rname in enumerate(REGION_SHORT):
        fig.add_trace(go.Scatter(x=rk.columns, y=rk.iloc[i], name=rname, mode="lines+markers"))
    fig.update_layout(title="Thứ hạng theo w_AI (1 = tốt nhất)", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("6.4.4 (mở rộng AHP): so sánh cặp đôi của AHP cho trọng số chủ quan có kiểm tra "
               "nhất quán (CR < 0,1); với 8 tiêu chí ma trận 8×8 cồng kềnh nên bản web dùng "
               "entropy làm phương án trọng số khách quan đối chứng — AHP đầy đủ để trong notebook.")
    top1 = out.sort_values("Hạng CG")["Vùng"].iloc[0]
    biggest_shift = int((out["Hạng CG"] - out["Hạng Ent"]).abs().idxmax())
    policy_box([
        ("a) Vùng dẫn đầu — nơi đặt trung tâm AI quốc gia đầu tiên?",
         f"**{top1}** dẫn đầu với trọng số chuyên gia — ứng viên tự nhiên cho trung tâm AI đầu "
         "tiên; quyết định cuối nên cân thêm chi phí đất, điện và liên kết đại học."),
        ("b) Vùng đổi hạng mạnh nhất khi dùng entropy?",
         f"**{out.loc[biggest_shift, 'Vùng']}** — entropy thưởng tiêu chí có độ phân tán lớn "
         "(FDI, R&D) nên vùng cực đoan ở các tiêu chí này dịch chuyển nhiều nhất."),
        ("c) Đa cộng tuyến AI Readiness ↔ Internet?",
         "Tương quan cao gây 'đếm trùng' một chiều thông tin. Khắc phục: PCA gộp nhóm tiêu chí "
         "trước khi TOPSIS, hoặc dùng CRITIC điều chỉnh trọng số theo tương quan."),
        ("d) Chọn 3 vùng cho 3 trung tâm AI (QĐ 127/QĐ-TTg)?",
         "Top-2 theo điểm thường là ĐNB và ĐBSH; vùng thứ ba nên thêm tiêu chí địa-chính trị "
         "(phân tán rủi ro Bắc-Trung-Nam) → ứng viên hợp lý là BTB-DHTB (Đà Nẵng)."),
    ])


@st.cache_data(show_spinner="Đang chạy NSGA-II…")
def run_nsga2(beta_flat, e, rho, sig, D0, pop, gen, seed, lam=0.65):
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as pmin
    beta = np.array(beta_flat).reshape(6, 4)
    e, rho, sig, D0 = map(np.array, (e, rho, sig, D0))

    class Prob(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=24, n_obj=4, n_ieq_constr=20,
                             xl=np.zeros(24), xu=np.full(24, 12000.0))

        def _evaluate(self, x, out, *args, **kw):
            X = x.reshape(6, 4)
            f1 = -(beta * X).sum()
            sums = X.sum(1)
            f2 = np.abs(sums - sums.mean()).mean()
            f3 = (e * (X[:, 0] + X[:, 2])).sum()
            f4 = (rho * X[:, 2]).sum() - (sig * X[:, 3]).sum()
            out["F"] = [f1, f2, f3, f4]
            Dnew = D0 + 0.002 * X[:, 1]
            out["G"] = ([X.sum() - 50000]
                        + list(5000 - sums) + list(sums - 12000)
                        + [12000 - X[:, 3].sum()]
                        + list(lam * Dnew.max() - Dnew))

    res = pmin(Prob(), NSGA2(pop_size=pop, sampling=_b7_seed_population(D0, lam, pop)),
               ("n_gen", gen), seed=seed, verbose=False)
    if res.X is None:
        return None, None
    X = np.atleast_2d(res.X)
    F = np.atleast_2d(res.F)
    return X, F


def _b7_seed_population(D0, lam, pop):
    """Tập khả thi là 'lát mỏng' (C5 ép x_D tối thiểu rất lớn ở vùng số hóa thấp) nên khởi tạo
    ngẫu nhiên thuần không bao giờ chạm vào — dựng 1 điểm khả thi rồi nhiễu quanh nó."""
    rng = np.random.default_rng(0)
    M0 = D0.max()
    req_D = np.maximum(0.0, (lam * M0 - D0) / 0.002) * 1.01   # x_D tối thiểu theo C5
    base = np.zeros((6, 4))
    base[:, 1] = req_D
    base[:, 3] = 2000.0                                        # H: tổng 12.000 thỏa C4
    for r in range(6):                                         # nâng lên sàn vùng 5.000
        gap = 5000.0 - base[r].sum()
        if gap > 0:
            base[r, 0] += gap
    over = base.sum() - 50000.0                                # ép trần tổng nếu lỡ vượt
    if over > 0:
        slack = base[:, 0].sum()
        if slack > 0:
            base[:, 0] *= max(0.0, 1 - over / slack)
    seeds = [base.flatten()]
    top = int(np.argmax(D0))                                   # giữ x_D vùng dẫn đầu = 0 để M = M0
    for _ in range(pop - 1):
        pt = base * rng.uniform(0.97, 1.10, size=base.shape)
        pt[top, 1] = 0.0
        pt[:, 1] = np.maximum(pt[:, 1], base[:, 1])            # không tụt dưới mức C5 yêu cầu
        seeds.append(np.clip(pt.flatten(), 0, 12000))
    return np.array(seeds)


def page_b7(*_):
    st.header("Bài 7 — Tối ưu đa mục tiêu Pareto (NSGA-II, pymoo)")
    extra = st.data_editor(B7_EXTRA_DEFAULT, key="b7_extra", use_container_width=True)
    c = st.columns(5)
    pop = c[0].slider("pop_size", 40, 150, 80, 10)
    gen = c[1].slider("n_gen", 40, 200, 100, 20)
    seed = c[2].number_input("seed", value=42)
    lam7 = c[3].slider("λ công bằng (C5)", 0.50, 0.70, 0.65, 0.01,
                       help="λ = 0,70 nguyên văn đề làm tập khả thi RỖNG (xem chứng minh ở Bài 4) "
                            "→ Pareto trống. Mặc định 0,65.")
    c[4].caption("pop×gen lớn → chạy lâu hơn. Kết quả được cache theo bộ tham số.")
    if not st.button("▶️ Chạy NSGA-II (7.4.1)", type="primary"):
        st.info("Bấm nút để chạy (~10–60 giây; lần sau cùng tham số sẽ lấy từ cache).")
        return
    X, F = run_nsga2(tuple(BETA_B4_DEFAULT.values.flatten()),
                     tuple(extra.iloc[:, 0]), tuple(extra.iloc[:, 1]), tuple(extra.iloc[:, 2]),
                     tuple(REGIONS_DEFAULT["digital_index_0_100"].values),
                     int(pop), int(gen), int(seed), float(lam7))
    if F is None or len(F) == 0:
        st.error("Không tìm được nghiệm khả thi — λ quá cao làm tập khả thi rỗng (giảm λ) "
                 "hoặc tăng pop_size/n_gen.")
        return
    Fd = pd.DataFrame({"f1 GDP gain": -F[:, 0], "f2 Bất bình đẳng": F[:, 1],
                       "f3 Phát thải": F[:, 2], "f4 Rủi ro ròng": F[:, 3]})
    st.subheader(f"7.4.2 — Tập Pareto cuối cùng ({len(Fd)} nghiệm)")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.scatter_3d(Fd, x="f1 GDP gain", y="f2 Bất bình đẳng", z="f3 Phát thải",
                                  color="f4 Rủi ro ròng", title="Biên Pareto 3D (f1, f2, f3)"),
                    use_container_width=True)
    c2.plotly_chart(px.parallel_coordinates(Fd, color="f1 GDP gain",
                                            title="Parallel coordinates — 4 mục tiêu"),
                    use_container_width=True)

    st.subheader("7.4.3 — Nghiệm thỏa hiệp bằng TOPSIS trên tập Pareto")
    wts = np.array([0.40, 0.25, 0.20, 0.15])
    sc = topsis(Fd.values, wts, [True, False, False, False])
    best = int(np.argmax(sc))
    growth_max = int(np.argmax(Fd["f1 GDP gain"].values))
    comp = pd.DataFrame({"Nghiệm thỏa hiệp (TOPSIS)": Fd.iloc[best],
                         "Nghiệm tăng trưởng tối đa": Fd.iloc[growth_max]}).round(1)
    st.dataframe(comp, use_container_width=True)
    d2 = (Fd.iloc[growth_max, 1] / max(Fd.iloc[best, 1], 1e-9) - 1) * 100
    d3 = (Fd.iloc[growth_max, 2] / max(Fd.iloc[best, 2], 1e-9) - 1) * 100
    st.info(f"**7.4.4 — Chi phí cơ hội:** nghiệm tăng trưởng tối đa làm bất bình đẳng phân bổ "
            f"thay đổi **{d2:+.1f}%** và phát thải **{d3:+.1f}%** so với nghiệm thỏa hiệp.")
    alloc = pd.DataFrame(X[best].reshape(6, 4).round(0), index=REGION_SHORT, columns=ITEMS4)
    st.plotly_chart(px.imshow(alloc, text_auto=".0f", aspect="auto",
                              title="Phân bổ của nghiệm thỏa hiệp"), use_container_width=True)
    policy_box([
        ("a) Đánh đổi tăng trưởng – bao trùm có rõ không?",
         "Có — biên Pareto dốc rõ giữa f1 và f2 vì hệ số biên tập trung ở ĐNB/ĐBSH: muốn GDP gain "
         "cao phải dồn vốn về hai vùng này, lập tức tăng độ lệch phân bổ. Độ dốc lớn phản ánh cơ "
         "cấu kinh tế hai cực của Việt Nam."),
        ("b) Trọng số (0,40/0,25/0,20/0,15) có đúng ưu tiên hiện nay?",
         "Văn kiện ĐH XIII đặt 'phát triển nhanh và bền vững' ngang hàng; cam kết COP26 và QĐ "
         "127/QĐ-TTg gợi ý nâng trọng số môi trường + an ninh dữ liệu (~0,25/0,20) và hạ tăng "
         "trưởng xuống ~0,35."),
        ("c) NSGA-II khác gì LP đơn mục tiêu — có thay được quyết định chính trị?",
         "NSGA-II không chọn hộ một phương án; nó vẽ *toàn bộ mặt đánh đổi* để cơ quan chính trị "
         "lựa chọn có thông tin. Bước chọn điểm trên biên Pareto vẫn là quyết định giá trị — đúng "
         "tinh thần Mục 8.2 của bài báo nguồn."),
    ])


def page_b8(macro, *_):
    st.header("Bài 8 — Tối ưu động liên thời gian 2026-2035 (SLSQP)")
    st.caption("Ghi chú phương pháp: hàm Cobb-Douglas trong ràng buộc khiến bài toán không thỏa "
               "quy tắc DCP của CVXPY nếu viết trực tiếp → dùng **Cách B của đề (scipy SLSQP)**, "
               "mô phỏng trạng thái tiến từ các biến đầu tư. Bản CVXPY log-hóa để trong notebook.")
    A_default = float(tfp_series(macro["GDP_trillion_VND"].values, macro["K_trillion_VND"].values,
                                 macro["L_million"].values, macro["D_digital_pct_GDP"].values,
                                 macro["AI_thousand_firms"].values, macro["H_trained_pct"].values,
                                 0.33, 0.42, 0.10, 0.08, 0.07).mean())
    c = st.columns(5)
    util = c[0].selectbox("Hàm thỏa dụng", ["ln(C)", "CRRA γ=1,5"])
    phi_scale = c[1].slider("Hiệu chỉnh φ (spillover TFP)", 0.0, 1.0, 0.10, 0.05,
                            help="φ gốc của đề (0,003/0,002/0,004) với D≈20, AI≈86 cho TFP tăng "
                                 "~40%/năm — phi thực tế. Mặc định 0,10 để quỹ đạo có ý nghĩa; "
                                 "đặt 1,0 nếu muốn đúng nguyên văn đề.")
    shock = c[2].toggle("8.3.3: cú sốc −8% Y năm 2028")
    A0 = c[3].number_input("A0 (từ Bài 1)", value=round(A_default, 3))
    run = c[4].button("▶️ Giải (8.3.1)", type="primary")
    if not run:
        st.info("Bấm nút để giải (~10-30 giây).")
        return
    crra = 1.5 if util.startswith("CRRA") else None
    with st.spinner("SLSQP đang tối ưu 50 biến…"):
        sol = solve_b8(27500, 20.3, 86, 30, A0, 53.9, phi_scale=phi_scale, shock=shock, crra=crra)
    years = list(range(2026, 2036))
    st.subheader("8.3.2 — Quỹ đạo tối ưu")
    c1, c2 = st.columns(2)
    fig1 = go.Figure()
    for k in ["K", "D", "AI", "H"]:
        vals = np.array(sol["states"][k][1:])
        fig1.add_trace(go.Scatter(x=years, y=vals / max(vals[0], 1e-9),
                                  name=k, mode="lines+markers"))
    fig1.update_layout(title="Trạng thái K, D, AI, H (chuẩn hóa = 1 tại 2026)")
    c1.plotly_chart(fig1, use_container_width=True)
    fig2 = go.Figure([go.Scatter(x=years, y=sol["Y"], name="Y"),
                      go.Scatter(x=years, y=sol["C"], name="C")])
    fig2.update_layout(title="Sản lượng Y và tiêu dùng C (nghìn tỷ)")
    c2.plotly_chart(fig2, use_container_width=True)
    inv = pd.DataFrame({"Năm": years, "I_K": sol["IK"].round(0), "I_D": sol["ID"].round(2),
                        "I_AI": sol["IAI"].round(2), "I_H": sol["IH"].round(2),
                        "C": sol["C"].round(0)})
    st.dataframe(inv, hide_index=True, use_container_width=True)
    st.metric("Tổng phúc lợi W*", f"{sol['welfare']:.4f}",
              help="So sánh W giữa các kịch bản, không so giá trị tuyệt đối giữa hai dạng hàm thỏa dụng.")

    st.subheader("8.3.4 — So sánh hai chiến lược cố định")
    Y0_est = A0 * 27500**0.33 * 53.9**0.42 * 20.3**0.10 * 86**0.08 * 30**0.07

    def strat_welfare(mult):
        IK = 0.20 * Y0_est * mult
        ID = 4 * mult
        IAI = 10 * mult
        IH = 4 * mult
        w, _, _ = simulate_invest(IK, ID, IAI, IH, 27500, 20.3, 86, 30, A0, 53.9, phi_scale,
                                  shock_year=2 if shock else None, shock_size=0.08, crra=crra)
        return w

    even = strat_welfare(np.ones(10))
    front = strat_welfare(np.array([1.8] * 3 + [0.6] * 7))
    cc = st.columns(3)
    cc[0].metric("Đầu tư trải đều", f"{even:.4f}")
    cc[1].metric("Front-load 3 năm đầu", f"{front:.4f}", f"{front-even:+.4f}")
    cc[2].metric("Tối ưu SLSQP", f"{sol['welfare']:.4f}")
    winner = "front-load" if front > even else "trải đều"
    policy_box([
        ("a) Quỹ đạo tối ưu front-loaded hay back-loaded?",
         "Đầu tư K và H thường **front-loaded**: vốn tích lũy sớm sinh lợi kép qua nhiều năm, và H "
         "vừa vào hàm sản xuất vừa nâng TFP (φ3 lớn nhất). AI khấu hao nhanh (15%) nên dàn đều hơn."),
        ("b) Tỷ lệ đầu tư AI/H theo thời gian?",
         "Mô hình giữ I_H đi trước hoặc song song I_AI vì kênh TFP: thiếu H thì đóng góp AI bị "
         "chiết khấu. Hàm ý: đào tạo nhân lực **đi trước** làn sóng đầu tư AI."),
        ("c) Nếu ρ = 0,90 thay vì 0,97?",
         f"Chiết khấu nặng hơn → dồn tiêu dùng về hiện tại, cắt đầu tư dài hạn (nhất là H/R&D). "
         f"Đây chính là cơ chế khiến chính phủ nhiệm kỳ ngắn 'dưới đầu tư' vào R&D. (Với tham số "
         f"hiện tại, chiến lược {winner} thắng trong so sánh cố định ở trên.)"),
    ])


def page_b9(*_):
    st.header("Bài 9 — Tác động AI tới thị trường lao động (LP)")
    par = st.data_editor(JOBS_DEFAULT, key="b9_par", use_container_width=True, num_rows="fixed")
    c1, c2 = st.columns(2)
    budget = c1.slider("Ngân sách (tỷ VND)", 10000, 60000, 30000, 5000)
    cap5 = c2.toggle("9.4.4: thêm ràng buộc Displaced ≤ 5%·L_i")
    sol = solve_b9(par, budget, cap5)
    if sol is None:
        st.error("Không khả thi với ràng buộc hiện tại.")
        return
    st.subheader("9.4.1 — Phân bổ tối ưu & NetJob từng ngành")
    out = pd.DataFrame({"Ngành": par["Ngành"], "x_AI (tỷ)": sol["xAI"].round(0),
                        "x_H (tỷ)": sol["xH"].round(0),
                        "Việc mới (AI)": sol["newjob"].round(0),
                        "Nâng cấp (H)": sol["upgrade"].round(0),
                        "Dịch chuyển": sol["displaced"].round(0),
                        "NetJob": sol["net"].round(0)})
    c1, c2 = st.columns([3, 1])
    c1.dataframe(out, hide_index=True, use_container_width=True)
    c2.metric("Tổng NetJob", f"{sol['Z']:,.0f} việc làm")

    st.subheader("9.4.2 — Ngưỡng đào tạo tối thiểu: CN chế biến chế tạo")
    i = 1
    r2 = par["Risk %"].iloc[i] / 100
    c1v, a1v, d1v = par["c1"].iloc[i], par["a1"].iloc[i], par["d1"].iloc[i]
    ratio_cap = c1v * r2 / d1v
    netjob_coef = a1v - c1v * r2
    st.markdown(
        f"- Ràng buộc *Displaced ≤ RetrainCap*: x_H ≥ (c1·r/d1)·x_AI = **{ratio_cap:.3f}·x_AI** — "
        f"cứ 1 tỷ vào AI cần tối thiểu **{ratio_cap:.2f} tỷ** đào tạo lại đi kèm.\n"
        f"- Ràng buộc *NetJob ≥ 0*: hệ số ròng theo x_AI là a1 − c1·r = **{netjob_coef:.2f} > 0** "
        f"nên tự thỏa mãn — **năng lực đào tạo** (chứ không phải mất việc ròng) mới là nút thắt.")

    st.subheader("9.4.3 — Sankey luồng dịch chuyển lao động")
    sec = list(par["Ngành"])
    n = len(sec)
    src = list(range(n)) * 3
    tgt = [n] * n + [n + 1] * n + [n + 2] * n
    val = list(sol["displaced"]) + list(sol["newjob"]) + list(sol["upgrade"])
    fig = go.Figure(go.Sankey(
        node=dict(label=sec + ["→ Đào tạo lại / dịch chuyển", "→ Việc làm mới (AI)",
                               "→ Nâng cấp kỹ năng"], pad=12),
        link=dict(source=src, target=tgt, value=[max(v, 0.01) for v in val])))
    fig.update_layout(title="Luồng việc làm theo lời giải tối ưu (chú ý nhóm dễ tổn thương: "
                            "Nông nghiệp, Xây dựng, Bán lẻ)", height=480)
    st.plotly_chart(fig, use_container_width=True)
    if cap5:
        st.success("Với trần 5%·L_i: bài toán vẫn khả thi — trần chủ yếu bó CN chế biến chế tạo "
                   "và Bán buôn-bán lẻ, buộc giảm x_AI hoặc tăng x_H tương ứng.")
    top_h = out.sort_values("x_H (tỷ)", ascending=False)["Ngành"].iloc[0]
    policy_box([
        ("a) Ngành cần đào tạo lại nhiều nhất?",
         f"**{top_h}** — khớp trực giác: nơi vừa đông lao động vừa rủi ro tự động hóa cao thì một "
         "đồng x_H 'mở khóa' nhiều đồng x_AI nhất."),
        ("b) Chiến lược cho Tài chính-Ngân hàng (risk 52%)?",
         "AI mạnh + đào tạo bắt buộc đi kèm: a1 = 45,8 rất cao nhưng c1·r = 37,7 cũng cao → mỗi tỷ "
         "AI cần ~1,45 tỷ H (37,7/26) để thỏa năng lực đào tạo lại."),
        ("c) Có nên đổ x_AI vào Nông-Lâm-Thủy sản?",
         "Có-nhưng-ít: a1 = 8,5 thấp song c1·r chỉ 0,94 → hệ số ròng dương và 'rẻ' về đào tạo; "
         "tuy nhiên chi phí cơ hội cao so với CNTT (62,5) nên LP chỉ rót sau khi các ngành hiệu "
         "suất cao chạm ràng buộc."),
        ("d) 'Tốc độ tự động hóa không vượt năng lực đào tạo lại' là ràng buộc nào?",
         "Chính là **Displaced_i ≤ RetrainingCapacity_i** (c1·r·x_AI ≤ d1·x_H). Đề xuất bổ sung: "
         "trần mất việc theo vùng và NetJob ≥ 0 riêng cho nhóm lao động phổ thông."),
    ])


def page_b10(*_):
    st.header("Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn (VSS, EVPI)")
    st.caption("Mô hình là LP thuần nên web dùng PuLP/CBC (tương đương Pyomo+GLPK của đề; GLPK "
               "khó cài trên Streamlit Cloud — bản Pyomo để trong notebook nộp bài).")
    scen = st.data_editor(SCEN10_DEFAULT, key="b10_scen", hide_index=True, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    cap1 = c1.number_input("Ngân sách giai đoạn 1", value=65000, step=5000)
    cap2 = c2.number_input("Dự phòng giai đoạn 2", value=15000, step=2500)
    s1s = c3.toggle("β giai đoạn 1 cũng hiện thực hóa theo kịch bản", value=True,
                    help="OFF = nguyên văn dạng đơn giản hóa của đề (c'x xác định) — khi đó "
                         "EVPI và VSS có thể bằng 0 vì lời giải first-stage tầm thường. ON = "
                         "hiệu quả đầu tư ban đầu cũng phụ thuộc kịch bản (đọc theo Bảng 10.4), "
                         "thông tin hoàn hảo trở nên có giá trị.")
    prob = scen["Xác suất"].values
    if abs(prob.sum() - 1) > 1e-6:
        st.warning(f"Tổng xác suất = {prob.sum():.2f} ≠ 1 — sửa lại bảng kịch bản.")
        return
    beta_s = scen[["I", "D", "AI", "H"]].values

    x_sp, Z_sp = solve_b10_sp(BETA10_BASE, beta_s, prob, cap1, cap2, s1s)
    bbar = (prob[:, None] * beta_s).sum(0)
    x_ev, _ = solve_b10_one_scenario(BETA10_BASE, bbar, cap1, cap2, s1s)
    EEV = eval_first_stage(x_ev, BETA10_BASE, beta_s, prob, cap2, s1s)
    ws_vals, ws_x = [], []
    for s_ in range(4):
        xv, zv = solve_b10_one_scenario(BETA10_BASE, beta_s[s_], cap1, cap2, s1s)
        ws_vals.append(zv)
        ws_x.append(xv)
    WS = float(np.dot(prob, ws_vals))
    VSS, EVPI = Z_sp - EEV, WS - Z_sp

    st.subheader("10.5.1–10.5.2 — Quyết định first-stage: SP vs EV vs WS")
    J = ["I", "D", "AI", "H"]
    cmp = pd.DataFrame({"SP (stochastic)": x_sp.round(0), "EV (kịch bản TB)": x_ev.round(0),
                        **{scen["Kịch bản"].iloc[s_]: np.round(ws_x[s_], 0) for s_ in range(4)}},
                       index=J)
    st.dataframe(cmp, use_container_width=True)
    st.plotly_chart(px.bar(cmp[["SP (stochastic)", "EV (kịch bản TB)"]].T, barmode="stack",
                           title="Cơ cấu phân bổ first-stage"), use_container_width=True)

    st.subheader("10.5.3 — VSS và EVPI")
    c = st.columns(4)
    c[0].metric("Z* SP", f"{Z_sp:,.0f}")
    c[1].metric("EEV", f"{EEV:,.0f}")
    c[2].metric("VSS = SP − EEV", f"{VSS:,.0f}")
    c[3].metric("EVPI = WS − SP", f"{EVPI:,.0f}")
    if VSS < 1:
        st.info("**Vì sao VSS ≈ 0 ở đây?** Mô hình tuyến tính và tập khả thi giai đoạn 2 chỉ phụ "
                "thuộc x qua một ràng buộc duy nhất (y_AI ≤ 0,5·x_H). Khi 'quyền chọn AI' này "
                "không đáng giá ở hệ số hiện tại, lời giải SP trùng lời giải EV → VSS = 0. "
                "Đây là kết quả đúng, không phải lỗi: hãy thử sửa bảng (vd. tăng β_AI của s1, "
                "hoặc tăng xác suất khủng hoảng) để thấy VSS bật dương khi cơ cấu phòng thủ "
                "bằng H trở nên tối ưu.")

    st.subheader("10.5.4 — Robust: cực tiểu hóa regret kịch bản xấu nhất")
    m = pulp.LpProblem("regret", pulp.LpMinimize)
    x = {j: pulp.LpVariable(f"x_{j}", lowBound=0) for j in J}
    y = {(s_, j): pulp.LpVariable(f"y_{s_}_{j}", lowBound=0) for s_ in range(4) for j in J}
    th = pulp.LpVariable("theta")
    m += th
    m += pulp.lpSum(x.values()) <= cap1
    for s_ in range(4):
        m += pulp.lpSum(y[(s_, j)] for j in J) <= cap2
        m += y[(s_, "AI")] <= 0.5 * x["H"]
        b1 = ({J[k]: beta_s[s_][k] for k in range(4)} if s1s else BETA10_BASE)
        val = (pulp.lpSum(b1[j] * x[j] for j in J)
               + pulp.lpSum(beta_s[s_][k] * y[(s_, J[k])] for k in range(4)))
        m += th >= ws_vals[s_] - val
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    x_rb = np.array([x[j].value() or 0 for j in J])
    c1, c2 = st.columns(2)
    c1.dataframe(pd.DataFrame({"Minimax regret": x_rb.round(0), "SP": x_sp.round(0)}, index=J))
    c2.metric("Regret tối đa", f"{(th.value() or 0):,.0f}")
    h_more = x_sp[3] > x_ev[3] + 1
    policy_box([
        ("a) SP đầu tư H nhiều hơn hay ít hơn lời giải xác định?",
         (f"Với tham số hiện tại: SP phân bổ H = {x_sp[3]:,.0f} so với EV = {x_ev[3]:,.0f} — "
          + ("**nhiều hơn**, vì hai cơ chế: (i) y_AI ≤ 0,5·x_H biến H thành 'quyền chọn' mở rộng "
             "AI giai đoạn 2; (ii) β_H tăng lên 1,10 trong kịch bản khủng hoảng — H là tài sản "
             "phòng thủ." if h_more else
             "**bằng nhau** ở cấu hình này vì quyền chọn y_AI ≤ 0,5·x_H chưa đáng giá; tăng "
             "β_AI(s1) hoặc xác suất s4 trong bảng để thấy SP nghiêng về H như lý thuyết dự đoán "
             "(β_H = 1,10 ở khủng hoảng là cơ chế phòng thủ)."))),
        ("b) VSS dương nói lên điều gì?",
         f"VSS = {VSS:,.0f} tỷ: hoạch định theo tư duy xác suất (thay vì kịch bản trung bình) tạo "
         "thêm chừng đó giá trị kỳ vọng — bằng chứng định lượng cho việc đưa phân tích kịch bản "
         "vào quy trình lập kế hoạch 5 năm."),
        ("c) Bài học COVID-19 / bão Yagi?",
         "Cả hai cú sốc cho thấy lao động qua đào tạo chuyển đổi việc nhanh hơn — đúng cơ chế "
         "β_H = 1,10 ở s4. Việt Nam nhiều khả năng đang 'dưới đầu tư' vào nhân lực số xét như một "
         "**hàng hóa bảo hiểm vĩ mô**."),
    ])


@st.cache_data(show_spinner="Đang huấn luyện Q-learning…")
def cached_train(episodes, budget, seed):
    return train_qlearning(episodes=episodes, budget=budget, seed=seed)


def page_b11(*_):
    st.header("Bài 11 — Q-learning cho chính sách kinh tế thích nghi")
    st.caption("Môi trường MDP 3⁴ = 81 trạng thái × 5 hành động, cài thuần numpy với API kiểu "
               "gymnasium (reset/step). Phần DQN 11.3.5 để trong notebook — stable-baselines3 + "
               "torch ~2GB vượt giới hạn Streamlit Cloud miễn phí.")
    c = st.columns(3)
    episodes = c[0].slider("Số episodes (11.3.2)", 1000, 10000, 4000, 1000)
    budget = c[1].slider("Ngân sách/năm (nghìn tỷ)", 800, 2500, 1500, 100)
    seed = c[2].number_input("seed", value=0)
    if not st.button("▶️ Huấn luyện / tải cache", type="primary"):
        st.info("Bấm để huấn luyện (~5-30 giây; cache theo tham số).")
        return
    Q, curve = cached_train(int(episodes), int(budget), int(seed))

    st.subheader("11.3.4 — Learning curve & so sánh với rule-based")
    roll = pd.Series(curve).rolling(100, min_periods=10).mean()
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(y=roll, labels={"index": "Episode", "y": "Reward (MA-100)"},
                            title="Đường cong học"), use_container_width=True)
    scores = {
        "π* (Q-learning)": eval_policy(Q, budget, "greedy"),
        "Luôn a1 Cân bằng": eval_policy(Q, budget, "fixed", 1),
        "Luôn a3 AI dẫn dắt": eval_policy(Q, budget, "fixed", 3),
        "Ngẫu nhiên": eval_policy(Q, budget, "random"),
    }
    c2.plotly_chart(px.bar(x=list(scores), y=list(scores.values()),
                           labels={"x": "Chính sách", "y": "Reward tích lũy TB (200 eval)"},
                           title="π* vs rule-based"), use_container_width=True)

    st.subheader("11.3.3 — Chính sách π*(s) tại 5 trạng thái khởi đầu")
    env = VNEconEnv(budget=budget)
    cases = [("VN 2026 thực tế", None), ("GDP thấp, D thấp, U cao", "low"),
             ("GDP cao, AI cao, U thấp", "high"), ("Số hóa đã cao", "digital"),
             ("Khủng hoảng", "crisis")]
    rows = []
    for name, ov in cases:
        s_ = env.reset(state_override=ov)
        a_ = int(np.argmax(Q[s_]))
        rows.append({"Trạng thái": name, "(g,D,AI,U) rời rạc": str(s_),
                     "Hành động π*": ALLOC11[a_][0],
                     "Phân bổ K/D/AI/H": "/".join(f"{v:.0%}" for v in ALLOC11[a_][1])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    act_low = ALLOC11[int(np.argmax(Q[env.reset('low')]))][0]
    act_high = ALLOC11[int(np.argmax(Q[env.reset('high')]))][0]
    policy_box([
        ("a) Trạng thái xấu (GDP thấp, U cao) → hành động?",
         f"π* chọn **{act_low}** — thiên về H/cân bằng, đúng logic 'quick win an sinh trước, tăng "
         "tốc sau': giảm rủi ro thất nghiệp là thành phần phạt nặng thứ hai trong hàm thưởng."),
        ("b) Trạng thái tốt (GDP cao, AI cao) → hành động?",
         f"π* chọn **{act_high}** — giai đoạn consolidation: khi AI stock đã cao, phần thưởng biên "
         "của a3 giảm còn rủi ro mạng (tỷ lệ AI/H) tăng, agent chuyển sang phương án cân bằng hơn."),
        ("c) Tích hợp π* mà không vi phạm 'AI không thay quyết định chính trị'?",
         "Dùng π* như **hệ khuyến nghị có giải trình**: công bố Q-values, môi trường huấn luyện và "
         "hàm thưởng (trọng số w do con người chọn); quyết định cuối thuộc cơ quan dân cử; kiểm "
         "toán định kỳ sai lệch mô hình so với thực tế."),
    ])


def page_b12(macro, sectors, regions):
    st.header("Bài 12 — Nguyên mẫu AIDEOM-VN (đồ án tích hợp)")
    st.caption("6 module M1-M6 tích hợp kỹ thuật Bài 1-11; 5 kịch bản chính sách theo Mục 15.")
    budget_rate = st.slider("Tỷ lệ ngân sách đầu tư hằng năm (% GDP)", 0.15, 0.45, 0.30, 0.05)
    A0 = float(tfp_series(macro["GDP_trillion_VND"].values, macro["K_trillion_VND"].values,
                          macro["L_million"].values, macro["D_digital_pct_GDP"].values,
                          macro["AI_thousand_firms"].values, macro["H_trained_pct"].values,
                          0.33, 0.42, 0.10, 0.08, 0.07).mean())

    scenarios = {f"S{k+1}. {ALLOC11[k][0].split(' ', 1)[1]}": ALLOC11[k][1] for k in range(4)}
    rng = np.random.default_rng(7)
    best_score, best_share = -1e18, None
    for sh in rng.dirichlet(np.ones(4), size=250):
        kpi = simulate_scenario(sh, budget_rate=budget_rate, A0=A0)
        score = (0.4 * kpi["GDP2030"] / 20000 + 0.25 * kpi["NetJob"] / 50
                 - 0.2 * kpi["Cyber"] - 0.15 * kpi["Emis"])
        if score > best_score:
            best_score, best_share = score, sh
    scenarios["S5. Tối ưu cân bằng (AIDEOM)"] = best_share

    rows = []
    for name, sh in scenarios.items():
        kpi = simulate_scenario(sh, budget_rate=budget_rate, A0=A0)
        rows.append({"Kịch bản": name,
                     "Phân bổ K/D/AI/H": "/".join(f"{v:.0%}" for v in sh),
                     "GDP 2030 (nghìn tỷ)": round(kpi["GDP2030"]),
                     "KTS/GDP 2030 (%)": round(kpi["D2030"], 1),
                     "DN số (nghìn)": round(kpi["AI2030"]),
                     "NetJob (nghìn việc)": round(kpi["NetJob"], 1),
                     "Rủi ro mạng (0-1)": round(kpi["Cyber"], 3),
                     "Phát thải tương đối": round(kpi["Emis"], 3),
                     "Thất nghiệp rủi ro": f"{kpi['U']*100:.1f}%"})
    kdf = pd.DataFrame(rows)

    t1, t2, t3, t4 = st.tabs(["📊 Tổng quan (M1+M2)", "🗺️ Phân bổ (M3)",
                              "⚖️ So sánh kịch bản (M1-M4)", "🚨 Cảnh báo rủi ro (M5)"])
    with t1:
        st.markdown("**M1 — Dự báo nền (Cobb-Douglas, Bài 1):**")
        base = simulate_scenario(np.array([0.4, 0.25, 0.15, 0.20]), budget_rate=budget_rate, A0=A0)
        cc = st.columns(3)
        cc[0].metric("GDP 2030 (kịch bản cân bằng)", f"{base['GDP2030']:,.0f} nghìn tỷ")
        cc[1].metric("Kinh tế số/GDP 2030", f"{base['D2030']:.1f}%",
                     f"{base['D2030']-19.5:+.1f} điểm vs 2025")
        cc[2].metric("Mục tiêu 30% KTS", "ĐẠT ✅" if base["D2030"] >= 30 else "CHƯA ĐẠT ⚠️")
        st.markdown("**M2 — Mức sẵn sàng số 6 vùng (TOPSIS entropy, Bài 6):**")
        crit = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD", "digital_index_0_100",
                "ai_readiness_0_100", "trained_labor_pct", "rd_intensity_pct",
                "internet_penetration_pct", "gini_coef"]
        X = regions[crit].values.astype(float)
        sc = topsis(X, entropy_weights(X), [True] * 7 + [False])
        st.plotly_chart(px.bar(x=REGION_SHORT, y=sc, labels={"x": "", "y": "TOPSIS C*"},
                               title="Chỉ số sẵn sàng tổng hợp"), use_container_width=True)
    with t2:
        st.markdown("**M3 — Phân bổ ngân sách số 50 nghìn tỷ (LP Bài 4, có ràng buộc công bằng):**")
        sol, Z, _ = solve_b4(BETA_B4_DEFAULT.values, regions["digital_index_0_100"].values,
                             50000, 5000, 12000, 12000, 0.002, 0.7, True)
        st.plotly_chart(px.imshow(pd.DataFrame(sol, index=REGION_SHORT, columns=ITEMS4),
                                  text_auto=".0f", aspect="auto",
                                  title=f"Phân bổ tối ưu — Z* = {Z:,.0f} tỷ GDP gain"),
                        use_container_width=True)
    with t3:
        st.dataframe(kdf, hide_index=True, use_container_width=True)
        cats = ["GDP 2030 (nghìn tỷ)", "NetJob (nghìn việc)", "Rủi ro mạng (0-1)",
                "Phát thải tương đối"]
        norm = {}
        for col in cats:
            inv = col in ("Rủi ro mạng (0-1)", "Phát thải tương đối")
            norm[col] = minmax_norm(kdf[col].astype(float).values, benefit=not inv)
        fig = go.Figure()
        for i_ in range(len(kdf)):
            fig.add_trace(go.Scatterpolar(
                r=[norm[c_][i_] for c_ in cats],
                theta=["GDP", "Việc làm", "An toàn mạng", "Xanh"],
                fill="toself", name=kdf["Kịch bản"].iloc[i_]))
        fig.update_layout(title="Radar 5 kịch bản (đã chuẩn hóa; ra ngoài = tốt hơn)")
        st.plotly_chart(fig, use_container_width=True)
    with t4:
        st.markdown("**M5 — Quy tắc cảnh báo (tích hợp logic Bài 7 & 10):**")
        for _, r in kdf.iterrows():
            warn = []
            if r["Rủi ro mạng (0-1)"] > 0.45:
                warn.append("rủi ro an ninh dữ liệu cao (AI vượt xa nền tảng nhân lực)")
            if float(r["Thất nghiệp rủi ro"].rstrip("%")) > 7.5:
                warn.append("áp lực dịch chuyển lao động vượt năng lực đào tạo lại")
            if r["Phát thải tương đối"] > 0.5:
                warn.append("cường độ phát thải gián tiếp cao — xung đột cam kết COP26")
            if warn:
                st.warning(f"**{r['Kịch bản']}**: " + "; ".join(warn) + ".")
            else:
                st.success(f"**{r['Kịch bản']}**: trong ngưỡng an toàn.")
    st.divider()
    st.caption("M6 chính là dashboard này (Streamlit, 4 tab). Các deliverable còn lại của Bài 12 "
               "(báo cáo 15-25 trang, slide, video demo) dựng từ kết quả/biểu đồ ở trên.")


# =====================================================================================
# MAIN
# =====================================================================================

PAGES_BUILDERS = [
    ("🏠 Trang chủ", page_home), ("Bài 1 — Cobb-Douglas", page_b1),
    ("Bài 2 — LP ngân sách", page_b2), ("Bài 3 — Priority ngành", page_b3),
    ("Bài 4 — LP vùng×hạng mục", page_b4), ("Bài 5 — MIP dự án", page_b5),
    ("Bài 6 — TOPSIS vùng", page_b6), ("Bài 7 — NSGA-II Pareto", page_b7),
    ("Bài 8 — Tối ưu động", page_b8), ("Bài 9 — Lao động & AI", page_b9),
    ("Bài 10 — Stochastic 2 giai đoạn", page_b10), ("Bài 11 — Q-learning", page_b11),
    ("Bài 12 — AIDEOM-VN", page_b12),
]


def main():
    st.set_page_config(page_title="AIDEOM-VN", page_icon="🇻🇳", layout="wide")
    macro, sectors, regions = ensure_csvs()
    st.sidebar.title("AIDEOM-VN")
    st.sidebar.caption("Mô hình ra quyết định — Kinh tế VN trong kỉ nguyên AI")
    choice = st.sidebar.radio("Chọn bài", [n for n, _ in PAGES_BUILDERS], index=0)
    st.sidebar.divider()
    st.sidebar.caption("💡 Dữ liệu gốc: NSO/GSO, MoST, WIPO GII 2025.")
    dict(PAGES_BUILDERS)[choice](macro, sectors, regions)


if __name__ == "__main__":
    main()
