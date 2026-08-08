import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def annualised_return(returns: pd.Series):
    total_growth = (1 + returns).prod()
    n = len(returns)
    return float(total_growth ** (TRADING_DAYS_PER_YEAR / n) - 1)


def annualised_volatility(returns: pd.Series):
    return returns.std(ddof=1) * np.sqrt(252)


def sharpe_ratio(returns, risk_free=0.0):
    returns = returns - (risk_free / 252)
    ann_return = annualised_return(returns)
    ann_vol = annualised_volatility(returns)

    if ann_vol < 1e-10:
        if ann_return > 0:
            return float("inf")
        else:
            return 0.0

    return float(ann_return / ann_vol)


def max_drawdown(returns: pd.Series):
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdowns = (peak - equity) / peak
    dd = float(drawdowns.max())

    if dd == 0.0:
        return 0.0, 0

    trough_pos = drawdowns.argmax()
    pre_trough = equity.iloc[:trough_pos]
    peak_pos = pre_trough[pre_trough == peak.iloc[:trough_pos]].index[-1]
    duration = trough_pos - equity.index.get_loc(peak_pos)

    return dd, duration


def turnover(weights: pd.DataFrame):
    return weights.diff().iloc[1:].abs().sum(axis=1).mean()
