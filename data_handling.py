import yfinance as yf
import pandas as pd
import numpy as np
import pulp as pl



def make_jitter(index: pd.Index, sigma: float = 1e-6, seed: int | None = 42) -> pd.Series:
    """
    平均0の微小ノイズを作成（標準偏差 sigma）。
    index の長さに合わせて返す。再現性のため seed を指定可能。
    """
    rng = np.random.default_rng(seed)
    eps = pd.Series(rng.normal(loc=0.0, scale=sigma, size=len(index)), index=index, dtype=float)
    # ごく微小でも平均ずれをさらに抑えたいなら、以下で平均0に再調整
    eps = eps - eps.mean()
    return eps

def data_download(mapping, T=10, start="2008-01-01", end=None,
                  include_crypto50 = True, 
                  include_dom_hy   = False, 
                  include_dom_income_re   = True,
                  include_jpy_cash  = True,
                  include_usd_cash  = False):
    
    # ========= ダウンロード =========
    tickers = [t for t in mapping.values() if t]
    extras  = []
    if include_crypto50:
        extras += ["BTC-USD","ETH-USD"]
    if include_dom_hy and (mapping.get("海外ジャンク債") is not None):
        pass  # HYGは既にtickersに含まれている
    if include_dom_income_re and mapping.get("国内REIT"):
        pass  # 1343.T は既にtickersに含まれている
    dl = yf.download(list(set(tickers + extras)), start=start, end=end, auto_adjust=True, progress=False)["Close"]

    assets = []
    # 既存（ダイレクト対応）
    for k, tkr in mapping.items():
        if tkr is not None:
            assets.append(k)
    # クリプト50/50
    if include_crypto50:
        crypto_name = "クリプトETF（BTC/ETH 50/50）"
        assets.append(crypto_name)
    # 国内ジャンク債（HYG 代理）
    if include_dom_hy:
        dom_hy_name = "国内ジャンク債（HYG代理）"
        assets.append(dom_hy_name)
    # 国内収益不動産（1343.T 1.2x）
    if include_dom_income_re:
        dom_inc_re_name = "国内収益不動産（J-REIT×1.2）"
        assets.append(dom_inc_re_name)
    # 円/ドル現金
    if include_jpy_cash:
        jpy_cash_name = "円現金/定期預金（年0.1%）"
        assets.append(jpy_cash_name)
    if include_usd_cash:
        usd_cash_name = "ドル現金/定期預金（年3%）"
        assets.append(usd_cash_name)
    # 価格→リターン変換の補助
    def to_mret(prices):
        return prices.resample("ME").last().pct_change().dropna(how="all")
    # 月次リターンテーブル作成
    ret_cols = {}
    # マッピング分
    for k, tkr in mapping.items():
        if tkr is None:
            continue
        s = dl[tkr].dropna()
        r = to_mret(s)
        ret_cols[k] = r
    # クリプト50/50（価格合成ではなく**リターン**の等ウェイト合成）
    if include_crypto50:
        btc = dl["BTC-USD"].dropna()
        eth = dl["ETH-USD"].dropna()
        rb = to_mret(btc)
        re = to_mret(eth)
        r_crypt = pd.concat([rb, re], axis=1).dropna()
        r_crypt["eq"] = r_crypt.mean(axis=1)
        ret_cols[crypto_name] = r_crypt["eq"]
    # 国内ジャンク債（HYG 代理）
    if include_dom_hy:
        hyg = dl["HYG"].dropna()
        ret_cols[dom_hy_name] = to_mret(hyg)
    # 国内収益不動産（J-REIT×1.2：リターン倍率）
    if include_dom_income_re:
        jreit = dl[mapping["国内REIT"]].dropna()
        r = to_mret(jreit).rename("r")
        r12 = (1 + r).pow(1.2) - 1  # リターンを1.2倍（単純×1.2 より歪みが少ない指数的拡大）
        ret_cols[dom_inc_re_name] = r12
    # 円/ドル現金：確定月次リターン
    def annual_to_monthly_rate(apr):
        return (1 + apr)**(1/12) - 1
    if include_jpy_cash:
        base_jpy = pd.Series(annual_to_monthly_rate(0.001), index=dl.index)  # 0.1%
        jitter_jpy = make_jitter(dl.index, sigma=1e-6, seed=100) 
        r_jpy = base_jpy.add(jitter_jpy)
        ret_cols[jpy_cash_name] = r_jpy
    if include_usd_cash:
        base_usd = pd.Series(annual_to_monthly_rate(0.03), index=dl.index)   # 3%
        jitter_usd = make_jitter(dl.index, sigma=1e-6, seed=200)
        r_usd = base_usd.add(jitter_usd)
        ret_cols[usd_cash_name] = r_usd
    # 連結（共通の月に合わせる）
    ret_df = pd.concat(ret_cols, axis=1).dropna(how="all")

    # --- ここを ret_df 作成直後に追加（concat の後すぐ） ---
    prefer_first = "円現金/定期預金（年0.1%）"

    # ret_df に実在する列だけを取って順序を作る
    cols_present = [c for c in ret_df.columns]

    # 先頭に置きたい資産（存在すれば先頭、なければ無視）
    front = [prefer_first] if prefer_first in cols_present else []

    # 残り（重複排除しつつ元の順序維持）
    rest = [c for c in cols_present if c != prefer_first]

    # 新しい列順に並べ替え
    new_order = front + rest
    ret_df = ret_df.reindex(columns=new_order)

    # 以後で使う資産名リストもこの順序に合わせる
    asset_names = new_order

    ret_df = ret_df.dropna(axis=0, how="any")  # 相関を取るため同一月だけに揃える
    # ========= 期間分割（等分割） =========
    months = ret_df.index
    n = len(months)
    cuts = np.linspace(0, n, T+1, dtype=int)  # 等分割の境界
    period_idxs = [(cuts[i], cuts[i+1]) for i in range(T)]
    # ==== 期待値・標準偏差を (資産名, 期間) 列に整形 ====
    T = len(period_idxs)

    # まずは (index=資産, columns=期間) の形で計算
    mu_mat  = pd.DataFrame(index=asset_names, columns=range(1, T+1), dtype=float)
    vol_mat = pd.DataFrame(index=asset_names, columns=range(1, T+1), dtype=float)

    for t_idx, (i0, i1) in enumerate(period_idxs, start=1):
        r_t = ret_df.iloc[i0:i1]  # その期間の月次リターン（期間ごとの実長でOK）
        # 資産の並びを統一しておく（念のため）
        r_t = r_t.reindex(columns=asset_names)
        mu_mat.loc[:, t_idx]  = r_t.mean().values
        vol_mat.loc[:, t_idx] = r_t.std(ddof=1).values

    # 列 MultiIndex（資産名→期間）を作成
    exval_std_columns = pd.MultiIndex.from_product(
        [asset_names, range(1, T+1)],
        names=["資産名", "期間"]
    )

    # 行インデックス（単一レベルでもOKだが、ご指定どおり MultiIndex を使用）
    exval_std_rows = pd.MultiIndex.from_product(
        [["期待値", "標準偏差"]],
        names=["統計量"]
    )

    # (資産, 期間) の順に並んだ 1次元 Series にしてから縦に結合
    mu_stacked  = mu_mat.stack()   # index=(資産名, 期間)
    vol_stacked = vol_mat.stack()  # index=(資産名, 期間)

    # ご指定の列順に合わせて並べ替え
    mu_stacked  = mu_stacked.reindex(exval_std_columns)
    vol_stacked = vol_stacked.reindex(exval_std_columns)

    # 2×(nA*T) の最終テーブル
    exval_std_df = pd.DataFrame(
        [mu_stacked, vol_stacked],
        index=["期待値", "標準偏差"],
        columns=exval_std_columns
    )
    # 行をご指定の MultiIndex に差し替え（見た目・互換性のため）
    exval_std_df.index = exval_std_rows

    # ========= 相関（全期間×全資産、(資産名, 期間)） =========
    nA = len(asset_names)
    T  = len(period_idxs)

    # 各期間の長さ -> 最小長 m で揃える
    lens = [i1 - i0 for (i0, i1) in period_idxs]
    m = min(lens)

    # 形状: X = (m, nA*T)
    blocks = []
    col_tuples = []
    for a in asset_names:
        for t_idx, (i0, i1) in enumerate(period_idxs, start=1):
            # 期間 t_idx の a のリターン（最小長 m に切揃え）
            s = ret_df.loc[ret_df.index[i0:i1], a].iloc[:m].to_numpy()
            blocks.append(s.reshape(-1, 1))
            col_tuples.append((a, t_idx))

    X = np.hstack(blocks)                   # (m, nA*T)
    C = np.corrcoef(X, rowvar=False)        # (nA*T, nA*T)

    # 行・列 MultiIndex（資産名→期間）
    mi = pd.MultiIndex.from_tuples(col_tuples, names=["資産名", "期間"])
    corr_df = pd.DataFrame(C, index=mi, columns=mi)

    return exval_std_df, corr_df, asset_names

def get_simulated_rets(exval_std_df, corr_df, n_paths=500):
    # 0) ensure index/column alignment and order
    corr_df_1 = corr_df.loc[exval_std_df.columns, exval_std_df.columns].copy()

    # 1) clean: clip to [-1,1], symmetrize, ensure diag=1
    A = corr_df_1.to_numpy().astype(float)

    # 4) simulate
    rng = np.random.default_rng(1)
    mu = exval_std_df.loc["期待値"].to_numpy().squeeze()
    sig = exval_std_df.loc["標準偏差"].to_numpy().squeeze()
    K = mu.size
    sig_matrix = np.zeros(shape=(K,K))
    np.fill_diagonal(sig_matrix, sig)
    cov = sig_matrix @ A @ sig_matrix


    z = rng.multivariate_normal(mean=mu, cov=cov, size=(n_paths,))

    sim_index = pd.MultiIndex.from_product([[I for I in range(1, n_paths+1)]], names=["パス"])
    simulated = pd.DataFrame(data=z,index=sim_index, columns=exval_std_df.columns)

    return simulated

def get_gross_rets(simulated, asset_names, initial_call_rate=0.000083, initial_asset_price=1.0, ):
    simulated_by_asset = {asset: simulated.xs(asset, axis=1, level="資産名") for asset in simulated.columns.get_level_values("資産名").unique()}
    T = simulated.columns.get_level_values("期間").max()
    variables_columns = pd.MultiIndex.from_product([asset_names, [t for t in range(T+1)]],names=["資産名", "期間"])
    variables_rows = simulated.index
    rf_asset = asset_names[0]

    # Build rho so that all assets at period 0 are 1, except ('金利', 0) is 0.0044 for all paths
    gross_rets = np.empty((len(variables_rows), len(variables_columns)))
    for col_idx, (asset, period) in enumerate(variables_columns):
        if period == 0:
            if asset == rf_asset and initial_call_rate is not None:
                gross_rets[:, col_idx] = initial_call_rate 
            else:
                gross_rets[:, col_idx] = initial_asset_price
        else:
            # simulated columns are MultiIndex (資産名, 期間) with period starting from 1
            if (asset, period) in simulated.columns:
                gross_rets[:, col_idx] = simulated[(asset, period)].values + 1
            else:
                gross_rets[:, col_idx] = np.nan  # or handle as needed

    gross_rets__df = pd.DataFrame(gross_rets, index=variables_rows, columns=variables_columns)
    return gross_rets__df

def get_r_rho_df(gross_rets__df, asset_names):
    rets_by_asset = {asset: gross_rets__df.xs(asset, axis=1, level="資産名") for asset in gross_rets__df.columns.get_level_values("資産名").unique()}

    # Build rho_df as the cumulative product for each asset across periods
    rho_df_parts = []
    for asset, df in rets_by_asset.items():
        # cumprod along periods (axis=1)
        cumprod_df = df.cumprod(axis=1)
        # assign MultiIndex columns for this asset
        cumprod_df.columns = pd.MultiIndex.from_product([[asset], cumprod_df.columns], names=["資産名", "期間"])
        rho_df_parts.append(cumprod_df)

    # Concatenate all assets along columns
    rho_df = pd.concat(rho_df_parts, axis=1)
    # Reorder columns to match variables_columns if needed
    rho_df = rho_df.reindex(columns=gross_rets__df.columns)

    rf_asset = asset_names[0]
    r_df = rho_df[rf_asset]
    r_df_mean=r_df.mean(axis=0)
    return r_df, rho_df

def generate_variables(T, I, asset_names):
    # Example: for variable z (no path index, just asset and period, only until T-1)
    # Also create P+_jt and P-_jt variables (except for asset == "金利")
    variables_columns = pd.MultiIndex.from_product([asset_names, [t for t in range(T+1)]],names=["資産名", "期間"])
    rf_asset = asset_names[0]

    z_vars = {}
    Pplus_vars = {}
    Pminus_vars = {}
    for col in variables_columns:
        asset, period = col
        if period < T:
            if asset == rf_asset:
                z_vars[col] = None
                Pplus_vars[col] = None
                Pminus_vars[col] = None
            else:
                z_vars[col] = pl.LpVariable(f"z_{asset}_{period}", lowBound=0)
                Pplus_vars[col] = pl.LpVariable(f"Pplus_{asset}_{period}", lowBound=0)
                Pminus_vars[col] = pl.LpVariable(f"Pminus_{asset}_{period}", lowBound=0)
        else:
            z_vars[col] = None  # No variable for period T
            Pplus_vars[col] = None
            Pminus_vars[col] = None

    # Create a DataFrame with the same columns as variables_columns, but a dummy single-row index
    dec_var_index = ["z_jt", "P+_jt", "P-_jt"]
    z_df = pd.DataFrame([z_vars], index=["z_jt"], columns=variables_columns)
    Pplus_df = pd.DataFrame([Pplus_vars], index=["P+_jt"], columns=variables_columns)
    Pminus_df = pd.DataFrame([Pminus_vars], index=["P-_jt"], columns=variables_columns)

    # Concatenate for a single view (optional)
    all_vars_df = pd.concat([z_df, Pplus_df, Pminus_df])

    v_0 = pl.LpVariable("v_0", lowBound=0)
    h_0 = pl.LpVariable("h_0", lowBound=0)

    # Build a MultiIndex for rows (scenarios/paths)
    v_index = pd.Index(range(1, I+1), name="パス")
    # Columns are periods 1..T-1
    v_columns = pd.Index(range(1, T), name="期間")

    # Create v^{(i)}_t and q^{i} variables for each scenario (path) and period t=1..T-1
    v_df = pd.DataFrame(index=v_index, columns=v_columns)
    q_df = pd.DataFrame(index=v_index, columns=["q^i"])
    h_df = pd.DataFrame(index=v_index, columns=v_columns)

    for i in range(1, I+1):
        q_df.loc[i,"q^i"] = pl.LpVariable(f"q^{i}", lowBound=0)
        for t in range(1, T):
            v_df.loc[i, t] = pl.LpVariable(f"v^{i}_{t}", lowBound=0)
            h_df.loc[i, t] = pl.LpVariable(f"h^{i}_{t}", lowBound=0)
    
    # Build a MultiIndex for rows (scenarios/paths)
    y_index = pd.Index(range(1, I+1), name="パス")
    # Columns are periods 1..T-1
    y_columns = pd.Index(range(1, T+1), name="期間")

    y_df = pd.DataFrame(index=range(1, I+1), columns=range(1, T+1), dtype=object)
    H_df = pd.DataFrame(index=range(1, I+1), columns=range(1, T+1), dtype=object)

    # t=1 用の共通変数を作成（全 i で同一オブジェクトを参照させる）
    y_t1_common = pl.LpVariable("y_t1_common")  # 下限制約があるなら指定
    H_t1_common = pl.LpVariable("H_t1_common", lowBound=0)              # 下限制約があるなら指定

    # t=1 には共通変数を代入
    for i in range(1, I+1):
        y_df.loc[i, 1] = y_t1_common
        H_df.loc[i, 1] = H_t1_common

    # t>=2 は通常どおり i ごとに別の変数を作る
    for t in range(2, T+1):
        for i in range(1, I+1):
            y_df.loc[i, t] = pl.LpVariable(f"y_{i}_{t}", lowBound=0)
            H_df.loc[i, t] = pl.LpVariable(f"H_{i}_{t}")
    
    return z_df, Pplus_df, Pminus_df, v_0, h_0, v_df, h_df, q_df, y_df, H_df
