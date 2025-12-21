from pyscipopt import Model, quicksum, SCIP_PARAMEMPHASIS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


def wage_function(thresholds, Ht, wage_df, t):
    K = len(thresholds) + 1 
    for k in range(K - 1):
        if Ht < thresholds[k]:
            return wage_df.loc[f"wage{k+1}", t]

    return wage_df.loc[f"wage{K}", t]


def compute_bounds_and_M(I, T, H_0, W_0, dep_df, thresholds, shocks_H, asset_names, rho_df, wage_df, wage_n=None): 
    m = dep_df.shape[0]
    H_min = {t: H_0 + shocks_H[t-1] for t in range(1, T+1)}
    H_max = {t: H_0 for t in range(1, T+1)}
    yt_max = {t: 0.0 for t in range(1, T+1)}
    Wt_max = {
        "H":{t: 0.0 for t in range(0, T+1)},
        "50":{t: 0.0 for t in range(0, T+1)},
        "F":{t: 0.0 for t in range(0, T+1)}
               }
    Wt_5050 = {t:[] for t in range(0, T+1)}

    Wt_max["H"][0] = W_0
    Wt_max["50"][0] = np.array([W_0]*I)
    Wt_max["F"][0] = np.array([W_0]*I)
    Wt_5050[0] = [W_0*0.5, W_0*0.5]

    K = len(thresholds) + 1 
    
    if wage_n is None:
        wage_k = []
        for k in range(1,K+1):
            wage_k.append(wage_df.xs(f"wage{k}", level="wage level").mean(axis=0))
        ave_wage_df = pd.concat(wage_k, axis=1).T
        ave_wage_df.index = [f"wage{k}" for k in range(1,K+1)]
        print("Average Wage: Simulated")
        display(ave_wage_df)

    else:
        ave_wage_df=pd.DataFrame(np.tile(wage_n, (T,1)), index=pd.Index(range(1,T+1)),columns=[f"wage{k}" for k in range(1,K+1)]).T
        print("Average Wage: Normal Regime")
        display(ave_wage_df)


    # t == 1 
    H1 = H_0 + dep_df.loc["d_1", 1]*W_0
    yt_max[1] = wage_function(thresholds, Ht=H1, wage_df=ave_wage_df, t=1)

    for t in range(1, T+1):
        dep = list(map(float, dep_df[t].values))
        ub = 0.0
        ub_50 = 0.0
        for ell in range(1, m+1):
            tm = t - ell
            if tm >= 0:  # tm==0 means h0
                ub += dep[ell-1] * Wt_max["H"][t-1]
                ub_50 += dep[ell-1] * 0.5 * Wt_max["50"][t-1]

        H_max[t] = H_0 + ub + shocks_H[t-1]
        Ht_50 = H_0 + ub_50 + shocks_H[t-1]
        yt_max[t] = wage_function(thresholds, Ht=H_max[t], wage_df=ave_wage_df, t=t)
        Wt_max["H"][t] = (yt_max[t])
        if isinstance(Ht_50, list):
            yt_50 = []
            for Ht in Ht_50:       
                yt_50.append(wage_function(thresholds, Ht=Ht, wage_df=ave_wage_df, t=t))
        else:
            yt_50 = wage_function(thresholds, Ht=H1, wage_df=ave_wage_df, t=t)
        
        max_candidates = []
        for asset in asset_names:
            max_candidates.append(rho_df.loc[:, (asset)].pct_change(axis=1)[t].values)
        financial_assets_max_rtn = np.vstack([max_candidates]).max(axis=0) 
        financial_50 = (1+financial_assets_max_rtn) * 0.5 * Wt_max["50"][t-1]
        Wt_max["50"][t] = financial_50 + yt_50
        Wt_max["F"][t] = (1+financial_assets_max_rtn) * W_0
        Wt_5050[t] = [financial_50, yt_50]
        
    M_t = {t: max(H_max[t] - thresholds[0], thresholds[-1] - H_min[t]) for t in range(1, T+1)}  # per-period M
    return H_min, H_max, M_t, Wt_max, Wt_5050, yt_max

def build_scip_model(
    I,T,K,
    asset_names,
    delta,
    H0, thresholds, 
    dep_df, 
    m,      
    wage_df,
    W0, WE, WG,                        # wealth params
    M_t,
    shock_H,                           # exogenous shocks to H per period (dict t->value)                                                           
    rho_df, r_df,
    initial_call_rate = 0.001,
    h_cap=None,                         # cap on h per period    
    time_limit=None,
    feasibility=True,
    slack=True
):
    """Return (model, var dicts) ready to solve in SCIP."""

    SCEN = [i+1 for i in range(I)]       
    TIME = list(range(1, T+1))
    TIME_TR = list(range(1, T))

    # Select columns for the last period
    rho_last = rho_df.xs(T, level="t", axis=1)
    # Calculate the mean across all paths (rows) for each asset
    rho_bar_T = rho_last.mean(axis=0)

    # Sensible cap for h if not given
    if h_cap is None:
        h_cap = W0

    # taus = [τ0..τK] (length K+1), include lower and upper
    # Tight H bounds and per-period Big-M
    # --- SCIP model
    mdl = Model("Dynamic_Portfolio_with_HumanCapital_StepMILP")
    if time_limit is not None:
        mdl.setParam("limits/time", time_limit)
    if feasibility:
        mdl.setEmphasis(SCIP_PARAMEMPHASIS.HARDLP)

    # -----------------------------
    # Variables
    # -----------------------------
    # Risky inventory z[a,t] (units, t=0..T-1 in your PuLP; here we'll store t in 0..T-1 for inventory)
    z = {} ; h = {}                 # z[a,t]  t in 0..T-1
    
    for t in range(0, T):  # inventory defined for 0..T-1
        h[t] = mdl.addVar(vtype="C", lb=0.0, name=f"h_{t}")
        for a in asset_names:
            z[a, t] = mdl.addVar(vtype="C", lb=0.0, name=f"z_{a}_{t}")

    # Cash v[i,t], human-cap invest h[i,t], q[i]
    v = {} ; q = {} ; y = {}
    for i in SCEN:
        q[i] = mdl.addVar(vtype="C", lb=0.0, name=f"q_{i}")
        for t in TIME:
            v[i, t] = mdl.addVar(vtype="C", lb=0.0, name=f"v_{i}_{t}") 
            y[i, t] = mdl.addVar(vtype="C", lb=0.0, name=f"y_{i}_{t}")    

    # Initial cash
    v0 = mdl.addVar(vtype="C", lb=0.0, name="v0")

    # H[t], y[t]
    H = {} 
    for t in TIME:    
        H[t] = mdl.addVar(vtype="C", name=f"H_{t}")
        
    # Regime binaries u[(i,t,k)], with shared t=1 option
    u = {}
    for t in TIME:
        for k in range(1, K+1):
            u[t, k] = mdl.addVar(vtype="B", name=f"u_{t}_{k}")

    print("variables added")

    # -----------------------------
    # Constraints
    # -----------------------------
    # Initial budget: sum_a (1+δ)*ρ_{a0} z[a,0] + v0 + h0 == W0
    # If you have per-asset initial price, use it; else use 1.0 as you did.
    initial_asset_price = 1.0
    mdl.addCons(
        quicksum((1+delta)*initial_asset_price * z[a,0] for a in asset_names) + v0 + h[0] == W0,
        name="InitialBudget"
    )

    print("initial budget constraint added")

    # Per-scenario budgets t=1..T-1
    for i in SCEN:
        for t in TIME_TR:
            buy_cost  = quicksum( (1+delta)* rho_df.loc[i, (a, t)] * z[a, t] for a in asset_names )
            sell_cash = quicksum( (1-delta)* rho_df.loc[i, (a, t)] * z[a, t-1] for a in asset_names )
            if t == 1:
                rhs = sell_cash + (1+initial_call_rate)*v0 + y[i, 1]
            else:
                rhs = sell_cash + (1+ r_df.loc[i, t-1])*v[i, t-1] + y[i,t]
            mdl.addCons(buy_cost + v[i, t] + h[t] == rhs, name=f"Budget_{i}_{t}")
    print("per-scenario budget constraints added")
    # Human Capital dynamics:
    # H[i,t] = H0 + sum_{ell=1..m} d_ell * h[i, t-ell], with t-ell==0 → h0, <0 → ignore
    for t in TIME:
        dep = list(map(float, dep_df[t].values))
        terms = []
        for ell, d_ell in enumerate(dep, start=1):
            tm = t - ell
            if tm >= 0:
                terms.append(d_ell * h[tm])
            else:
                break
        mdl.addCons(H[t] == H0 + quicksum(terms) + shock_H[t-1] if terms else H0, name=f"Hdyn_{t}")
    print("HC dynamics constraints added")
    # One-hot regime selection
    # t=1: once if shared, otherwise per scenario; t>=2: per (i,t)
    for t in TIME:
        mdl.addCons(quicksum(u[t, k] for k in range(1, K+1)) == 1, name=f"OneSeg_{t}")
    print("one-hot regime selection constraints added")
    # Big-M bracketing: τ_{k-1} <= H[i,t] < τ_k when u[i,t,k]=1 (relaxed otherwise with M_t)
    EPS = 1e-6
    for t in TIME:
        Ht = H[t]
        for k in range(1, K+1):
            if k == 1:
                tau = thresholds[0]
                mdl.addCons(Ht <= (tau - EPS) + M_t[t]*(1 - u[t, k]), name=f"Hhi_{t}_{k}")
            elif k == K:
                tau = thresholds[-1]
                mdl.addCons(Ht >= tau - M_t[t]*(1 - u[t, k]), name=f"Hlo_{t}_{k}")
            else:
                tau_lo = thresholds[k-2]; tau_hi = thresholds[k-1]
                mdl.addCons(Ht >= tau_lo - M_t[t]*(1 - u[t, k]), name=f"Hlo_{t}_{k}")
                mdl.addCons(Ht <= (tau_hi - EPS) + M_t[t]*(1 - u[t, k]), name=f"Hhi_{t}_{k}")
    print("Big-M regime bracketing constraints added")
    for i in SCEN:
        for t in TIME:
            mdl.addCons(y[i,t] == quicksum(wage_df.loc[i,t][f"wage{k}"] * u[t, k] for k in range(1, K+1)), name=f"Ymap_{i}_{t}")
    print("wage mapping constraints added")
    # Expected terminal wealth:
    #   sum_a (1-δ) * rho_bar_T[a] * z[a, T-1] + (1/I) * sum_i {(1+r[i,T-1])*v[i,T-1] + y[i,T]} >= WE
    term_left_risky = quicksum( (1-delta)*float(rho_bar_T[a]) * z[a, T-1] for a in asset_names )
    term_left_cash  = (1.0/len(SCEN)) * quicksum( (1+r_df.loc[i, T-1])*v[i, T-1] + y[i, T] for i in SCEN )

    orig_obj = (1.0/len(SCEN)) * quicksum(q[i] for i in SCEN)
    # Pathwise terminal goal per scenario
    risky_T = {} ; cash_T = {}
    for i in SCEN:
        risky_T[i] = quicksum( (1-delta)* rho_df.loc[i, (a, T)]  * z[a, T-1] for a in asset_names )
        cash_T[i]  = (1+ r_df.loc[i, T-1])*v[i, T-1] + y[i, T]
    
    if not slack:
        mdl.addCons(term_left_risky + term_left_cash >= float(WE), name="ExpTerminal")
        print("expected terminal wealth constraint added")

        for i in SCEN:
            mdl.addCons(risky_T[i] + cash_T[i] + q[i] >= float(WG), name=f"PathGoal_{i}")
        print("pathwise terminal goal constraints added")

        mdl.setObjective( orig_obj, "minimize")
        print("objective set")

        vars = dict(z=z, v=v, h=h, v0=v0, H=H, y=y, q=q, u=u)

    else:
        s_exp = mdl.addVar(name="s_ExpTerminal", vtype="C", lb=0.0)  # 期待終端富用
        s_path = {i: mdl.addVar(name=f"s_PathGoal_{i}", vtype="C", lb=0.0) for i in SCEN}

        # --- ExpTerminal（平均的な終端富） ---
        mdl.addCons(term_left_risky + term_left_cash + s_exp >= float(WE), name="ExpTerminal_soft")
        print("expected terminal wealth constraint with slack added")

        # --- PathGoal（各シナリオの終端富） ---
        for i in SCEN:
            mdl.addCons(risky_T[i] + cash_T[i] + q[i] + s_path[i] >= float(WG), name=f"PathGoal_soft_{i}")
        print("pathwise terminal goal constraints with slack added")
        
        M = 1e6
        mdl.setObjective(orig_obj+ M * (s_exp + quicksum(s_path[i] for i in SCEN)), "minimize")
        print("Modified objective with slack penalties")

        vars = dict(z=z, v=v, h=h, v0=v0, H=H, y=y, q=q, u=u, s_exp=s_exp, s_path=s_path)

    return mdl, vars

def solve_and_extract(mdl, vars, I, T, asset_names, slack=True):
    
    SCEN = [i+1 for i in range(I)]       
    TIME = list(range(1, T+1))

    mdl.optimize()

    status = mdl.getStatus()
    nsols  = mdl.getNSols()
    print(f"Status: {status}  #solutions: {nsols}")

    if nsols == 0:
        return dict(status=status, obj=None, z=None, v=None, h=None, H=None, y=None,
                    v0=None, h0=None, u=None, u_choice=None)

    sol = mdl.getBestSol()
    def val(x): return mdl.getSolVal(sol, x) if x is not None else np.nan
    
    if slack:
        s_exp = vars["s_exp"]
        s_path = vars["s_path"]

        print("s_exp =", val(s_exp))
        for i in SCEN:
            print(f"scenario {i}: s_path =", val(s_path[i]))
    
    # Unpack
    z = vars["z"]; v = vars["v"]; h = vars["h"]; H = vars["H"]; y = vars["y"]
    v0 = vars["v0"]; u = vars["u"]

    # v,h,H,y (i,t=1..T)
    idx  = pd.Index(SCEN, name="i")
    cols = pd.Index(TIME, name="t")

    # z (a,t=0..T-1)
    Tmax = max(TIME)
    z_df = pd.DataFrame(index=["z_jt"],
                        columns=pd.MultiIndex.from_product([asset_names, range(0, Tmax)]))
    h_df = pd.DataFrame(index=["ht"], columns=[t for t in range(0, Tmax)], dtype=float)
   
    for t in range(0, Tmax):
        h_df.at["ht", t] = val(h[t])
        for a in asset_names:
            z_df.at["z_jt", (a, t)] = val(z[a, t])

    v_df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    y_df = pd.DataFrame(index=idx, columns=cols, dtype=float)    
    H_df = pd.DataFrame(index=["Ht"], columns=cols, dtype=float)
    

    for t in TIME:
        H_df.loc["Ht", t] = val(H[t])   
        for i in SCEN:
            v_df.loc[i, t] = val(v[i, t])
            y_df.loc[i, t] = val(y[i, t])

    v0_val = val(v0)

    # ===== NEW: extract binaries u_{i,t,k} =====
    # Find all k present
    Ks = sorted({k for (_, k) in u.keys()})
    # Build DataFrame indexed by (i,t), columns = k
    u_idx  = [f"k={k}" for k in Ks]
    u_df = pd.DataFrame(index=u_idx, columns=cols, dtype=float)

    for t in TIME:
        for k in Ks:
            u_df.loc[f"k={k}", t] = val(u[t, k])

    return dict(status=status, obj=mdl.getSolObjVal(sol), z=z_df, v=v_df, h=h_df, H=H_df, y=y_df,
                v0=v0_val, u=u_df)

def investment_ratio(rho_df, asset_names, sol, v0_val, v_val, h_val, z_val, I, T, W_0, W_E, W_G, param_type):
    # === Precompute average prices as a simple DataFrame ===
    SCEN = [i+1 for i in range(I)]       

    # rho_df: rows = scenarios i, columns = MultiIndex (資産名, 期間)
    avg_price = rho_df.mean(axis=0)                 # Series indexed by (asset, period)
    price_df  = avg_price.unstack(level=-1)
             # rows: 資産名, cols: 期間
    rf_asset = "Cash"
    all_cols = [rf_asset, "Human Cap"] + asset_names

    # Detect whether price periods are 0..T-1 or 1..T and map accordingly
    price_cols = price_df.columns
    use_offset = 0 if 0 in price_cols else 1        # if periods are 1..T, use t+1
    TIME0 = list(range(T))                           # 0..T-1 for pre-trade snapshot

    # Check solve status once (PySCIPOpt getStatus() returns strings like "optimal", "infeasible", etc.)
    status_ok = (sol.get("status") == "optimal") and (sol.get("obj") is not None)

    if not status_ok:
        print("cannnot calculate investment weights")
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
                for a in asset_names
            )

            if t == 0:
                cash_vec = v0_val
            else:
                cash_vec = v_val.loc[SCEN, t].astype(float)
            h_vec = h_val.loc["ht", t].astype(float)
            denom = risky_val_t + cash_vec + h_vec
            #denom_pos = denom.where(denom > 0, other=np.nan)
            #print((cash_vec / denom_pos))
            if t == 0:
                weights.loc[(slice(None), 0), rf_asset]   = (cash_vec / denom)
                weights.loc[(slice(None), 0), "Human Cap"] = (h_vec / denom)
                for a in asset_names:
                    num = price_df.at[a, tt] * z_val.loc["z_jt", (a, t)]
                    weights.loc[(slice(None), t), a] = (num / denom)
            else:
                weights.loc[(slice(None), t), rf_asset]   = (cash_vec / denom).values
                weights.loc[(slice(None), t), "Human Cap"] = (h_vec / denom).values
                for a in asset_names:
                    num = price_df.at[a, tt] * z_val.loc["z_jt", (a, t)]
                    weights.loc[(slice(None), t), a] = (num / denom).values

            weights.loc[(slice(None), t), :] = (weights.loc[(slice(None), t), :] * 100).fillna(0.0)

        # Re-order columns just in case
        weights = weights[all_cols]

        # --- Display ---

        # Scenario-mean by period
        weights_mean_by_t = weights.groupby(level="t").mean().round(2)
        print(f"Investment Weight: Senario Average (W_0:{W_0}, W_E:{W_E}, W_G:{W_G})")
        display(weights_mean_by_t)
        # Plot with integer x-axis labels
        ax = weights_mean_by_t.plot(
            ylabel="Weight (%)", figsize=(12, 6), marker="o"
        )
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # <<< added line
        plt.xlabel("Period")
        plt.title("Investment Weight: Senario Average")
        plt.tight_layout()

        p = Path(f"results/{param_type}/weights.png")
        if not p.exists():
            plt.savefig(f"results/{param_type}/weights.png")

        plt.show()



        return weights

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


def asset_composition(sol, rho_df, r_df, asset_names, v0_val, v_val, z_val, y_val, I, T, W_0, W_E, W_G, param_type, initial_call_rate=0.001):
    
    SCEN = [i+1 for i in range(I)]       
    TIME = list(range(1, T+1))

    avg_price = rho_df.mean(axis=0)  
    rf_asset = "Cash"
    all_cols_post = ["Wealth"] + [rf_asset] + ["Wage"] + asset_names

    # Check solve status once (PySCIPOpt getStatus() returns strings like "optimal", "infeasible", etc.)
    status_ok = (sol.get("status") == "optimal") and (sol.get("obj") is not None)

    if not status_ok:
        print("cannot calculate asset composition")
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
                for a in asset_names:
                    # Use average asset price (no scenario dependence)
                    price = (avg_price[a][t])
                    units = (z_val.at["z_jt", (a, t-1)])
                    risky_val +=  price * units 
                
                if t == 1:
                    cash_pre = (1 + initial_call_rate) * v0_val
                else:
                    cash_pre =  ( 1 + r_df.loc[i,t-1]) * v_val.loc[i, t-1]
                
                y_it = y_val.loc[i, t]

                denom  = risky_val + cash_pre + y_it

                if denom <= 0:
                    post_weights.loc[(i, t), :] = 0.0
                else:
                    # Risk-free (cash) weight first
                    post_weights.loc[(i, t), rf_asset] = np.round(cash_pre, decimals=2)
                    post_weights.loc[(i, t), "Wage"] = y_it 
                    # Risky asset weights
                    for a in asset_names:
                        price = (avg_price[a][t])
                        units = float(z_val.at["z_jt", (a, t-1)])
                        post_weights.loc[(i, t), a] = np.round(price * units, decimals=2) 
                    
                post_weights.loc[(i, t), "Wealth"] = np.round(denom, decimals=2)

        # Re-order columns (just in case)
        post_weights = post_weights[all_cols_post] 
        # --- Display ---

        # Scenario-average portfolio composition per period
        post_weights_mean_by_t = post_weights.groupby(level="t").mean().round(2)
        print(f"Asset Composition: Scenario Average (W_0:{W_0}, W_G:{W_G}, W_E:{W_E})")
        display(post_weights_mean_by_t)
        display((post_weights_mean_by_t[post_weights_mean_by_t.columns[1:]]).plot(kind="bar", figsize=(12,6), stacked=True))
        p = Path(f"results/{param_type}/asset_comp.png")
        if not p.exists():
            plt.savefig(f"results/{param_type}/asset_comp.png")
        
        return post_weights

def show_skill_chg(sol, param_type):
    # Check solve status once (PySCIPOpt getStatus() returns strings like "optimal", "infeasible", etc.)
    status_ok = (sol.get("status") == "optimal") and (sol.get("obj") is not None)

    if not status_ok:
        print("uは計算できません")
    else:
        u_df = (sol["u"].abs() > 0.5).astype(int)
        display(u_df)
        k_by_t = u_df.T.idxmax(axis=1).str.replace("k=", "").astype(int)
        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(k_by_t.index, k_by_t.values, marker="o", color="C0")

        ax.set_title("average skill level (average k)")
        ax.set_xlabel("Period")
        ax.set_ylabel("k")

        # Integer tick control
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_ylim(0, 8.9)

        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        p = Path(f"results/{param_type}/skill_level.png")
        if not p.exists():
            plt.savefig(f"results/{param_type}/skill_level.png")
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

    K = len(beta)                    

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


