import numpy as np

def get_error_stats(sim, obs, pcol="exc_prb", ycol="precip_avg_mm", to_inches=True):
    f = 25.4 if to_inches else 1.0
    cx = (1 / sim[pcol].to_numpy()).astype(float)
    cy = (sim[ycol].to_numpy() / f).astype(float)
    mx = (1 / obs[pcol].to_numpy()).astype(float)
    my = (obs[ycol].to_numpy() / f).astype(float)

    # clean & sort
    okc, okm = np.isfinite(cx*cy), np.isfinite(mx*my)
    cx, cy = cx[okc], cy[okc]; mx, my = mx[okm], my[okm]
    i, j = np.argsort(cx), np.argsort(mx)
    cx, cy, mx, my = cx[i], cy[i], mx[j], my[j]

    # integer grid 1..floor(max mx)
    xmax = float(mx.max())
    grid = np.arange(1, int(np.floor(xmax)) + 1, dtype=float)

    cyg = np.interp(grid, cx, cy)
    myg = np.interp(grid, mx, my)
    rmse = float(np.sqrt(np.mean((cyg - myg) ** 2)))

    # diff at exact max of monte
    diff_at_max = float(np.interp(xmax, cx, cy) - np.interp(xmax, mx, my))
    return {"rmse": rmse, "diff_at_max": diff_at_max}