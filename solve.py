from pyscipopt import Model, quicksum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def compute_bounds_and_M(T, H0, depreciation, h_cap):
    dep = list(map(float, depreciation))  # [d1..dm]
    m = len(dep)
    H_min = {t: float(H0) for t in range(1, T+1)}
    H_max = {t: float(H0) for t in range(1, T+1)}
    for t in range(1, T+1):
        ub = 0.0
        for ell in range(1, m+1):
            tm = t - ell
            if tm >= 0:  # tm==0 means h0
                ub += dep[ell-1] * h_cap
        H_max[t] = float(H0) + ub
    M_t = {t: H_max[t] - H_min[t] for t in range(1, T+1)}  # per-period M
    return H_min, H_max, M_t

def build_scip_model(
    SCEN,              # list of scenario ids: [1..I]
    TIME,              # list of periods: [1..T]
    TIME_TR,           # trade periods: [1..T-1]
    risk_assets,       # list of asset names (risky only)
    rf_asset,          # name of risk-free (for reporting)
    delta,             # tx cost
    initial_call_rate, # cash growth for t=1
    H0, thresholds, depreciation, m,      # HC parameters
    beta,                              # [β1..β8]
    W0, WE, WG,                        # wealth targets
    rho_df, r_df, rho_bar_T,           # market data (pandas)
    h_cap=None,                         # cap on h per period
    share_u_t1=True,                    # share regime u across scenarios at t=1
    link_yH_t1=False,                   # (optional) share y/H at t=1 across scenarios
    time_limit=None, threads=6, msg=True
):
    """Return (model, var dicts) ready to solve in SCIP."""

    K = len(beta)                      # number of segments (e.g., 8)
    I = len(SCEN)
    T = max(TIME)

    # Sensible cap for h if not given
    if h_cap is None:
        h_cap = W0

    # taus = [τ0..τK] (length K+1), include lower and upper
    # Tight H bounds and per-period Big-M
    H_min, H_max, M_t = compute_bounds_and_M(T, H0, depreciation, h_cap)

    # Build strictly increasing taus (do NOT inject H0 twice)
    EPS  = 1e-4
    taus_raw = [float(x) for x in thresholds if x > H_min[1] + EPS and x < H_max[T] - EPS]
    taus = [H_min[1]] + sorted(taus_raw) + [H_max[T]]
    for j in range(1, len(taus)):
        if taus[j] <= taus[j-1] + EPS:
            taus[j] = taus[j-1] + EPS

    K = len(taus) - 1
    beta = list(map(float, beta))[:K]


    # --- SCIP model
    mdl = Model("Dynamic_Portfolio_with_HumanCapital_StepMILP")
    mdl.setParam("display/verblevel", 4 if msg else 0)
    mdl.setParam("parallel/maxnthreads", threads)
    if time_limit is not None:
        mdl.setParam("limits/time", float(time_limit))

    # -----------------------------
    # Variables
    # -----------------------------
    # Risky inventory z[a,t] (units, t=0..T-1 in your PuLP; here we’ll store t in 0..T-1 for inventory)
    z = {}                 # z[a,t]  t in 0..T-1
    Pplus, Pminus = {}, {} # trades at t>=1
    for a in risk_assets:
        for t in range(0, T):  # inventory defined for 0..T-1
            z[a, t] = mdl.addVar(vtype="C", lb=0.0, name=f"z_{a}_{t}")
        for t in TIME_TR:      # trades at 1..T-1
            Pplus[a, t]  = mdl.addVar(vtype="C", lb=0.0, name=f"Pplus_{a}_{t}")
            Pminus[a, t] = mdl.addVar(vtype="C", lb=0.0, name=f"Pminus_{a}_{t}")

    # Cash v[i,t], human-cap invest h[i,t], q[i]
    v = {} ; h = {} ; q = {}
    for i in SCEN:
        for t in TIME:
            v[i, t] = mdl.addVar(vtype="C", lb=0.0, name=f"v_{i}_{t}")
            h[i, t] = mdl.addVar(vtype="C", lb=0.0, ub=h_cap, name=f"h_{i}_{t}")
        q[i] = mdl.addVar(vtype="C", lb=0.0, name=f"q_{i}")

    # Initial cash & h
    v0 = mdl.addVar(vtype="C", lb=0.0, name="v0")
    h0 = mdl.addVar(vtype="C", lb=0.0, ub=h_cap, name="h0")

    # H[i,t], y[i,t]
    H = {} ; y = {}
    for i in SCEN:
        for t in TIME:
            H[i, t] = mdl.addVar(vtype="C", name=f"H_{i}_{t}")
            y[i, t] = mdl.addVar(vtype="C", lb=0.0, name=f"y_{i}_{t}")

    # Regime binaries u[(i,t,k)], with shared t=1 option
    u = {}
    if share_u_t1:
        u_t1 = {k: mdl.addVar(vtype="B", name=f"u_t1_{k}") for k in range(1, K+1)}
        for i in SCEN:
            for k in range(1, K+1):
                u[i, 1, k] = u_t1[k]
    else:
        for i in SCEN:
            for k in range(1, K+1):
                u[i, 1, k] = mdl.addVar(vtype="B", name=f"u_{i}_1_{k}")
    for i in SCEN:
        for t in TIME:
            if t == 1: 
                continue
            for k in range(1, K+1):
                u[i, t, k] = mdl.addVar(vtype="B", name=f"u_{i}_{t}_{k}")

    # Optionally link y/H at t=1 (shared across scenarios)
    if link_yH_t1:
        # use i0 as reference
        i0 = SCEN[0]
        for i in SCEN[1:]:
            mdl.addCons(y[i,1] == y[i0,1], name=f"link_y_t1_i{i}")
            mdl.addCons(H[i,1] == H[i0,1], name=f"link_H_t1_i{i}")

    # -----------------------------
    # Objective
    # -----------------------------
    mdl.setObjective( (1.0/len(SCEN)) * quicksum(q[i] for i in SCEN), "minimize")

    # -----------------------------
    # Constraints
    # -----------------------------
    # Initial budget: sum_a (1+δ)*ρ_{a0} z[a,0] + v0 + h0 == W0
    # If you have per-asset initial price, use it; else use 1.0 as you did.
    initial_asset_price = 1.0
    mdl.addCons(
        quicksum((1+delta)*initial_asset_price * z[a,0] for a in risk_assets) + v0 + h0 == float(W0),
        name="InitialBudget"
    )

    # Inventory dynamics: z[a,t] = z[a,t-1] + Pplus[a,t] - Pminus[a,t]
    for a in risk_assets:
        for t in TIME_TR:
            mdl.addCons(z[a, t] == z[a, t-1] + Pplus[a, t] - Pminus[a, t], name=f"InvDyn_{a}_{t}")

    # Per-scenario budgets t=1..T-1
    for i in SCEN:
        for t in TIME_TR:
            buy_cost  = quicksum( (1+delta)* rho_df.loc[i, (a, t)].values[0] * Pplus[a, t]  for a in risk_assets )
            sell_cash = quicksum( (1-delta)* rho_df.loc[i, (a, t)].values[0] * Pminus[a, t] for a in risk_assets )
            if t == 1:
                rhs = sell_cash + (1+float(initial_call_rate))*v0 + y[i,1]
            else:
                rhs = sell_cash + (1+ r_df.loc[i, t-1].values[0])*v[i, t-1] + y[i, t]
            mdl.addCons(buy_cost + v[i, t] + h[i, t] == rhs, name=f"Budget_{i}_{t}")

    # Human Capital dynamics:
    # H[i,t] = H0 + sum_{ell=1..m} d_ell * h[i, t-ell], with t-ell==0 → h0, <0 → ignore
    dep = list(map(float, depreciation))
    for i in SCEN:
        for t in TIME:
            terms = []
            for ell, d_ell in enumerate(dep, start=1):
                tm = t - ell
                if tm > 0:
                    terms.append(d_ell * h[i, tm])
                elif tm == 0:
                    terms.append(d_ell * h0)
            mdl.addCons(H[i,t] == float(H0) + quicksum(terms) if terms else float(H0), name=f"Hdyn_{i}_{t}")

    # One-hot regime selection
    # t=1: once if shared, otherwise per scenario; t>=2: per (i,t)
    if share_u_t1:
        mdl.addCons(quicksum(u[SCEN[0], 1, k] for k in range(1, K+1)) == 1, name="OneSeg_t1_shared")
        # all u[i,1,k] point to same var objects, so no extra constraints needed
    else:
        for i in SCEN:
            mdl.addCons(quicksum(u[i, 1, k] for k in range(1, K+1)) == 1, name=f"OneSeg_{i}_1")
    for i in SCEN:
        for t in TIME:
            if t == 1: 
                continue
            mdl.addCons(quicksum(u[i, t, k] for k in range(1, K+1)) == 1, name=f"OneSeg_{i}_{t}")

    # Big-M bracketing: τ_{k-1} <= H[i,t] < τ_k when u[i,t,k]=1 (relaxed otherwise with M_t)
    for i in SCEN:
        for t in TIME:
            Mt = float(M_t[t])
            Hit = H[i, t]
            for k in range(1, K+1):
                tau_lo = float(taus[k-1]); tau_hi = float(taus[k])
                mdl.addCons(Hit >= tau_lo - Mt*(1 - u[i, t, k]), name=f"Hlo_{i}_{t}_{k}")
                mdl.addCons(Hit <= (tau_hi - EPS) + Mt*(1 - u[i, t, k]), name=f"Hhi_{i}_{t}_{k}")

    # Income selection: y[i,t] = sum_k beta[k-1]*u[i,t,k]
    beta = list(map(float, beta))  # [β1..βK]
    for i in SCEN:
        for t in TIME:
            mdl.addCons(y[i, t] == quicksum(beta[k-1] * u[i, t, k] for k in range(1, K+1)), name=f"Ymap_{i}_{t}")

    # Expected terminal wealth:
    #   sum_a (1-δ) * rho_bar_T[a] * z[a, T-1] + (1/I) * sum_i {(1+r[i,T-1])*v[i,T-1] + y[i,T]} >= WE
    term_left_risky = quicksum( (1-delta)*float(rho_bar_T[a]) * z[a, T-1] for a in risk_assets )
    term_left_cash  = (1.0/len(SCEN)) * quicksum( (1+r_df.loc[i, T-1].values[0])*v[i, T-1] + y[i, T] for i in SCEN )
    mdl.addCons(term_left_risky + term_left_cash >= float(WE), name="ExpTerminal")

    # Pathwise terminal goal per scenario
    for i in SCEN:
        risky_T = quicksum( (1-delta)* rho_df.loc[i, (a, T)].values[0]  * z[a, T-1] for a in risk_assets )
        cash_T  = (1+ r_df.loc[i, T-1].values[0])*v[i, T-1] + y[i, T]
        mdl.addCons(risky_T + cash_T + q[i] >= float(WG), name=f"PathGoal_{i}")

    return mdl, dict(z=z, Pplus=Pplus, Pminus=Pminus, v=v, h=h, v0=v0, h0=h0, H=H, y=y, q=q, u=u)

def solve_and_extract(mdl, vars, SCEN, TIME, risk_assets):
    mdl.optimize()

    status = mdl.getStatus()
    nsols  = mdl.getNSols()
    print(f"Status: {status}  #solutions: {nsols}")

    if nsols == 0:
        return dict(status=status, obj=None, z=None, v=None, h=None, H=None, y=None,
                    v0=None, h0=None, u=None, u_choice=None)

    sol = mdl.getBestSol()
    def val(x): return mdl.getSolVal(sol, x) if x is not None else np.nan

    # Unpack
    z = vars["z"]; v = vars["v"]; h = vars["h"]; H = vars["H"]; y = vars["y"]
    v0 = vars["v0"]; h0 = vars["h0"]; u = vars["u"]

    # z (a,t=0..T-1)
    Tmax = max(TIME)
    z_df = pd.DataFrame(index=["z_jt"],
                        columns=pd.MultiIndex.from_product([risk_assets, range(0, Tmax)]))
    for a in risk_assets:
        for t in range(0, Tmax):
            z_df.at["z_jt", (a, t)] = val(z[a, t])

    # v,h,H,y (i,t=1..T)
    idx  = pd.Index(SCEN, name="i")
    cols = pd.Index(TIME, name="t")
    v_df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    h_df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    H_df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    y_df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for i in SCEN:
        for t in TIME:
            v_df.loc[i, t] = val(v[i, t])
            h_df.loc[i, t] = val(h[i, t])
            H_df.loc[i, t] = val(H[i, t])
            y_df.loc[i, t] = val(y[i, t])

    v0_val = val(v0)
    h0_val = val(h0)

    # ===== NEW: extract binaries u_{i,t,k} =====
    # Find all k present
    Ks = sorted({k for (_, _, k) in u.keys()})
    # Build DataFrame indexed by (i,t), columns = k
    u_index = pd.MultiIndex.from_product([SCEN, TIME], names=["i", "t"])
    u_cols  = [f"k={k}" for k in Ks]
    u_df = pd.DataFrame(index=u_index, columns=u_cols, dtype=float)

    for i in SCEN:
        for t in TIME:
            for k in Ks:
                val_ = val(u.get((i, t, k)))  # shared t=1 binaries are same object across i
                u_df.loc[(i, t), f"k={k}"] = val_

    # Chosen segment per (i,t): argmax; robust to tiny float noise
    u_choice = u_df.astype(float).copy()
    # optional: clip to [0,1] then threshold small magnitudes
    u_choice = u_choice.clip(lower=0.0, upper=1.0)
    chosen_k = u_choice.idxmax(axis=1).str.replace("k=", "", regex=False).astype(int)

    return dict(status=status, obj=mdl.getSolObjVal(sol), z=z_df, v=v_df, h=h_df, H=H_df, y=y_df,
                v0=v0_val, h0=h0_val, u=u_df, u_choice=chosen_k)

def investment_ratio(rho_df, asset_names, sol, v0_val, h0_val, v_val, h_val, z_val, SCEN, T, W_0, W_E, W_G):
    # === Precompute average prices as a simple DataFrame ===
    # rho_df: rows = scenarios i, columns = MultiIndex (資産名, 期間)
    avg_price = rho_df.mean(axis=0)                 # Series indexed by (asset, period)
    price_df  = avg_price.unstack(level=-1)
             # rows: 資産名, cols: 期間
    risk_assets = asset_names[1:]
    rf_asset = asset_names[0]
    assets = list(risk_assets)
    all_cols = [rf_asset, "人的資本"] + assets

    # Detect whether price periods are 0..T-1 or 1..T and map accordingly
    price_cols = price_df.columns
    use_offset = 0 if 0 in price_cols else 1        # if periods are 1..T, use t+1
    TIME0 = list(range(T))                           # 0..T-1 for pre-trade snapshot

    # Check solve status once (PySCIPOpt getStatus() returns strings like "optimal", "infeasible", etc.)
    status_ok = (sol.get("status") == "optimal") and (sol.get("obj") is not None)

    if not status_ok:
        print("投資比率は計算できません")
    else:
        # Compute weights normally
        weights = pd.DataFrame(
            index=pd.MultiIndex.from_product([SCEN, TIME0], names=["i", "t"]),
            columns=all_cols,
            dtype=float,
        )

        for t in TIME0:
            tt = t + use_offset
            risky_val_t = sum(
                float(price_df.at[a, tt]) * float(z_val.loc["z_jt", (a, t)])
                for a in assets
            )

            if t == 0:
                cash_vec = pd.Series(v0_val, index=SCEN, dtype=float)
                h_vec    = pd.Series(h0_val, index=SCEN, dtype=float)
            else:
                cash_vec = v_val.loc[SCEN, t].astype(float)
                h_vec    = h_val.loc[SCEN, t].astype(float)

            denom = risky_val_t + cash_vec + h_vec
            denom_pos = denom.where(denom > 0, other=np.nan)

            weights.loc[(slice(None), t), rf_asset]   = (cash_vec / denom_pos).values
            weights.loc[(slice(None), t), "人的資本"] = (h_vec / denom_pos).values

            for a in assets:
                num = float(price_df.at[a, tt]) * float(z_val.loc["z_jt", (a, t)])
                weights.loc[(slice(None), t), a] = (num / denom_pos).values

            weights.loc[(slice(None), t), :] = (weights.loc[(slice(None), t), :] * 100).fillna(0.0)

        # Re-order columns just in case
        weights = weights[all_cols]

        # --- Display ---

        # Scenario-mean by period
        weights_mean_by_t = weights.groupby(level="t").mean().round(2)
        print(f"全シナリオ平均の投資比率％ (初期富：{W_0}, 目標富：{W_E}, 期待富：{W_G})")
        display(weights_mean_by_t)
        # Plot with integer x-axis labels
        ax = weights_mean_by_t.plot(
            ylabel="投資比率（％）", figsize=(12, 6), marker="o"
        )
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # <<< added line
        plt.xlabel("期間")
        plt.title("全シナリオ平均の投資比率")
        plt.tight_layout()
        plt.show()

def max_exp_wealth(exval_std_df, W_0, W_E):
    # exval_std_df : rows MultiIndex(["統計量"]), cols MultiIndex(["資産名","期間"])
    # W_0 : initial wealth

    # 1) Build a table: rows=資産名, cols=期間 of 期待値
    mu_panel = exval_std_df.loc["期待値"]             # columns: MultiIndex (資産名, 期間)
    periods  = mu_panel.columns.get_level_values("期間").unique().tolist()

    # columns per period: a small dict of Series (index=資産名)
    mu_by_t = pd.DataFrame(
        {t: mu_panel.xs(t, level="期間", axis=1).squeeze() for t in periods}
    )
    mu_by_t.index.name = "資産名"
    mu_by_t.columns.name = "期間"

    # 2) For each period, pick the asset with the max expected return
    best_asset_by_t = mu_by_t.idxmax(axis=0)   # Series (index=期間) -> 資産名
    best_mu_by_t    = mu_by_t.max(axis=0)      # Series (index=期間) -> 最大期待値

    # 3) Cumulative expected wealth path
    gross_by_t  = (1.0 + best_mu_by_t.astype(float))
    wealth_path = gross_by_t.cumprod() * float(W_0)

    # 4) Result table
    max_path_df = pd.DataFrame({
        "資産(最大期待値)": best_asset_by_t.values,
        "期待値": best_mu_by_t.values,
        "期待富": wealth_path.values
    }, index=pd.Index(periods, name="期間"))

    plt.figure(figsize=(12,6))
    plt.plot(max_path_df.index, max_path_df["期待富"], marker="o", label="最大期待富")
    plt.axhline(y=W_E, color="red", linestyle="--", label=f"目標期待富 W_E={W_E:,.0f}")
    plt.title("各期の最大期待富", fontsize=14)
    plt.xlabel("期間", fontsize=12)
    plt.ylabel("期待富", fontsize=12)
    plt.legend()
    plt.show()

    # ---- 結果出力 ----
    print(f"最終期の最大期待富: {wealth_path.iloc[-1]:,.4f}")
    if W_E > wealth_path.iloc[-1]:
        print("期待富 W_E が大きすぎます")
    display(max_path_df)


def asset_composition(sol, rho_df, r_df, asset_names, v0_val, v_val, z_val, y_val, initial_call_rate, SCEN, TIME, W_0, W_E, W_G):
    avg_price = rho_df.mean(axis=0)  
    risk_assets = asset_names[1:]
    rf_asset = asset_names[0]
    assets = list(risk_assets)
    all_cols_post = ["富"] + [rf_asset] + ["賃金"] + assets

    # Check solve status once (PySCIPOpt getStatus() returns strings like "optimal", "infeasible", etc.)
    status_ok = (sol.get("status") == "optimal") and (sol.get("obj") is not None)

    if not status_ok:
        print("資産構成は計算できません")
    else:

    

        # Build one tidy weights table: index = (scenario i, time t)
        post_weights = pd.DataFrame(
            index=pd.MultiIndex.from_product([SCEN, TIME], names=["i", "t"]),
            columns=all_cols_post,
            dtype=float,
        )

        for i in SCEN:
            for t in TIME:
                # --- compute risky & cash values ---
                risky_val = 0.0
                for a in assets:
                    # Use average asset price (no scenario dependence)
                    price = (avg_price[a][t])
                    units = (z_val.at["z_jt", (a, t-1)])
                    risky_val +=  price * units 
                
                if t == 1:
                    cash_pre = (1 + initial_call_rate) * v0_val
                else:
                    cash_pre =  ( 1 + r_df.loc[i,t-1].values) * float(v_val.loc[i, t-1])
                
                y_t = y_val.loc[i, t]

                denom  = risky_val + cash_pre + y_t

                if denom <= 0:
                    post_weights.loc[(i, t), :] = 0.0
                else:
                    # Risk-free (cash) weight first
                    post_weights.loc[(i, t), rf_asset] = np.round(cash_pre, decimals=2)
                    post_weights.loc[(i, t), "賃金"] = y_t 
                    # Risky asset weights
                    for a in assets:
                        price = (avg_price[a][t])
                        units = float(z_val.at["z_jt", (a, t-1)])
                        post_weights.loc[(i, t), a] = np.round(price * units, decimals=2) 
                    
                post_weights.loc[(i, t), "富"] = np.round(denom, decimals=2)

        # Re-order columns (just in case)
        post_weights = post_weights[all_cols_post] 
        # --- Display ---

        # Scenario-average portfolio composition per period
        post_weights_mean_by_t = post_weights.groupby(level="t").mean().round(2)
        print(f"全シナリオ平均の資産構成 (初期富：{W_0}, 目標富：{W_G}, 期待富：{W_E})")
        display(post_weights_mean_by_t)
        display((post_weights_mean_by_t[post_weights_mean_by_t.columns[1:]]).plot(kind="bar", figsize=(12,6), stacked=True))

def show_skill_chg(sol):
    # Check solve status once (PySCIPOpt getStatus() returns strings like "optimal", "infeasible", etc.)
    status_ok = (sol.get("status") == "optimal") and (sol.get("obj") is not None)

    if not status_ok:
        print("uは計算できません")
    else:
        # 2) Chosen segment per (i,t) as an integer k
        chosen = sol["u_choice"]                  # index: (i,t) → k
        chosen_df = chosen.unstack("t")           # rows=i, cols=t
        chosen_df.columns.name = "t"
        chosen_df.name = "chosen_k"

        # 3) How skill level changes over time (counts per t)
        counts_by_t = chosen.groupby(level="t").value_counts().unstack(fill_value=0)
        counts_by_t.index.name = "t"
        counts_by_t.columns.name = "k"
        print("Segment counts by period:")
        display(counts_by_t)

        # --- Average skill level by t ---
        avg_k_by_t = chosen.groupby(level="t").mean()

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(avg_k_by_t.index, avg_k_by_t.values, marker="o", color="C0")

        ax.set_title("平均スキル段階の推移 (k の平均)")
        ax.set_xlabel("期間 t")
        ax.set_ylabel("平均スキル段階 k")

        # Integer tick control
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_ylim(0, 8.9)

        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

def max_exp_wealth_with_hc_reinvest(
    exval_std_df, W_0, W_E,
    thresholds, beta, H0, depreciation, T,
    h_cap=None,                 # 人的資本に回す毎期の投資上限（既定: W0/2）
    reinvest_rate=1.0,          # 毎期の賃金をどの割合で金融資産に積立するか（0~1）
    title="最大期待富：金融（半分）+ 人的資本（半分、賃金は積立）"
):
    """
    - 金融パート: 期tの最大期待リターンmu*_tで入金つき複利
        F_0 = W0/2
        F_t = F_{t-1} * (1 + mu*_t) + reinvest_rate * y_t   （入金は期末、当期運用なし）
    - 人的資本パート: 毎期 h_cap 投資できると仮定し、H_max[t] 到達から段階賃金 y_t を決定
    """
    # ---- 1) 期ごとの最大期待リターン系列（金融） ----
    mu_panel = exval_std_df.loc["期待値"]                    # 列: MultiIndex(資産名, 期間)
    periods  = sorted(mu_panel.columns.get_level_values("期間").unique().tolist())
    mu_by_t  = pd.DataFrame({t: mu_panel.xs(t, level="期間", axis=1).squeeze() for t in periods})
    mu_by_t.index.name = "資産名"; mu_by_t.columns.name = "期間"

    best_mu_by_t = mu_by_t.max(axis=0).astype(float)        # 期ごとの最大期待値（Series, index=期間）

    # ---- 2) 人的資本：H_max[t] → 区間 → 賃金 y_t ----
    if h_cap is None:
        h_cap = float(W_0) / 2.0                            # 半々想定

    H_min, H_max, M_t = compute_bounds_and_M(T, H0, depreciation, h_cap)

    # 厳密単調増加な τ を構成（τ0..τK）
    EPS = 1e-4
    taus_raw = [float(x) for x in thresholds if x > H_min[1]+EPS and x < H_max[T]-EPS]
    taus = [H_min[1]] + sorted(taus_raw) + [H_max[T]]
    for j in range(1, len(taus)):
        if taus[j] <= taus[j-1] + EPS:
            taus[j] = taus[j-1] + EPS

    K = len(taus) - 1
    beta = list(map(float, beta))[:K]                       # 区間数に合わせる

    # 到達可能上限 H_max[t] に対応する最大賃金 y_t
    y_by_t = []
    for t in periods:
        Ht = H_max[t]
        chosen_beta = beta[-1]
        for k in range(1, K+1):
            if taus[k-1] <= Ht < taus[k] - EPS:
                chosen_beta = beta[k-1]
                break
        y_by_t.append(chosen_beta)
    y_by_t = pd.Series(y_by_t, index=periods, name="賃金（到達可能最大）")

    # ---- 3) 入金つき複利で金融資産を推移 ----
    F_path = []                     # 金融資産の期末残高
    F = float(W_0) / 2.0            # 初期の金融運用元本（半分）
    for t in periods:
        g_t = 1.0 + float(best_mu_by_t.loc[t])   # 当期の最大期待グロス
        F = F * g_t                              # まず当期を複利
        F = F + reinvest_rate * float(y_by_t.loc[t])  # 当期末に賃金の一部を入金
        F_path.append(F)
    F_path = pd.Series(F_path, index=periods, name="金融（賃金積立込み）")

    # 参考: 金融のみ（賃金積立なし、初期W0/2運用）
    F_only = (1.0 + best_mu_by_t).cumprod() * (float(W_0)/2.0)
    F_only.name = "金融のみ（W0/2運用）"

    # 総期待富 = 金融（賃金積立込み） + （賃金の非積立分を消費せず保有するなら足す）
    # ここでは「賃金の残り（1 - reinvest_rate）は消費」に解釈し、合計は F_path とする
    total_wealth = F_path.rename("合計期待富（金融+賃金積立）")

    # ---- 4) 描画 ----
    plt.figure(figsize=(12,6))
    plt.plot(F_only.index, F_only.values, marker="o", label=F_only.name)
    plt.plot(F_path.index, F_path.values, marker="o", label=F_path.name)
    plt.axhline(y=W_E, linestyle="--", label=f"目標期待富 W_E={W_E:,.0f}")
    plt.title(title, fontsize=14); plt.xlabel("期間", fontsize=12); plt.ylabel("期待富", fontsize=12)
    plt.legend(); plt.grid(True); plt.show()

    # ---- 5) テーブル返却 ----
    out = pd.DataFrame({
        "最大期待値（金融）": best_mu_by_t.values,
        "賃金（期ごと）": y_by_t.values,
        "金融のみ（W0/2運用）": F_only.values,
        "金融（賃金積立込み）": F_path.values,
        "合計期待富（金融+賃金積立）": total_wealth.values
    }, index=pd.Index(periods, name="期間"))

    print(f"最終期の 金融のみ（W0/2）: {F_only.iloc[-1]:,.4f}")
    print(f"最終期の 合計期待富:        {total_wealth.iloc[-1]:,.4f}")
    if W_E > total_wealth.iloc[-1]:
        print("⚠️ 期待富 W_E が大きすぎます（賃金の全額積立でも未達）")

    return out
