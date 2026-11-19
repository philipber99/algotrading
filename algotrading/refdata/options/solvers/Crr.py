from typing import Dict, Optional, Union
import time
import numpy as np
import torch


ALGORITHM_VERSION = "local_crr_v1"


def crr_sigma_floor(
    T: torch.Tensor,
    r: torch.Tensor,
    q: torch.Tensor,
    *,
    steps: int,
    sigma_lo: float,
    p_epsilon: float = 1.0e-3,
    min_sigma_abs: float = 0.0,
) -> torch.Tensor:
    """Return the stable probability-feasibility floor for a CRR tree."""
    dt = torch.clamp(T / steps, min=1.0e-12).to(dtype=torch.float64)
    b = (r - q).to(dtype=torch.float64)
    m = torch.exp(b * dt)
    eps = float(p_epsilon)
    disc = torch.clamp(m * m - (4.0 * eps * (1.0 - eps)), min=0.0)
    root = torch.sqrt(disc)

    pos_x = (m + root) / (2.0 * (1.0 - eps))
    # For negative carry the admissible volatility boundary is the stable
    # small root.  The reciprocal large root grows like 1 / epsilon and can
    # incorrectly put the CRR floor above every configured volatility.
    neg_x = (2.0 * (1.0 - eps)) / (m + root)
    x = torch.where(b >= 0.0, pos_x, neg_x)
    a_min = torch.log(torch.clamp(x, min=1.0 + 1.0e-12))
    sigma_floor = a_min / torch.sqrt(dt)
    sigma_floor = torch.clamp(
        sigma_floor,
        min=max(float(sigma_lo), float(min_sigma_abs)),
    )
    return sigma_floor.to(dtype=T.dtype)


# Compatibility alias for callers that imported the former private helper.
_crr_sigma_floor = crr_sigma_floor


def _crr_intrinsic_value(stock_layer: torch.Tensor, strike: torch.Tensor, is_call: torch.Tensor) -> torch.Tensor:
    return torch.where(
        is_call.unsqueeze(1),
        torch.clamp(stock_layer - strike.unsqueeze(1), min=0.0),
        torch.clamp(strike.unsqueeze(1) - stock_layer, min=0.0),
    )


def _crr_j_full(cache: dict, steps_local: int, ref: torch.Tensor) -> torch.Tensor:
    j_full = cache.get("j_full")
    if (j_full is None) or (j_full.numel() != steps_local + 1) or (j_full.device != ref.device) or (j_full.dtype != ref.dtype):
        j_full = torch.arange(steps_local + 1, device=ref.device, dtype=ref.dtype)
        cache["j_full"] = j_full
    return j_full


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    nan = torch.full_like(num, float("nan"))
    return torch.where(torch.abs(den) > eps, num / den, nan)


def _crr_price_tensor(
    Sd: torch.Tensor,
    Kd: torch.Tensor,
    Td: torch.Tensor,
    rd: torch.Tensor,
    qd: torch.Tensor,
    sigmad: torch.Tensor,
    is_calld: torch.Tensor,
    *,
    steps_local: int,
    j_cache: dict,
    return_primary_greeks: bool = False,
):
    Td = torch.clamp(Td, min=1.0e-8)
    sigmad = torch.clamp(sigmad, min=1.0e-8)

    dt = Td / steps_local
    sqrt_dt = torch.sqrt(dt)
    u = torch.exp(sigmad * sqrt_dt)
    d = 1.0 / u
    disc = torch.exp(-rd * dt)
    b = rd - qd
    p_denom = u - d
    p_raw = (torch.exp(b * dt) - d) / p_denom
    valid_regime = (
        torch.isfinite(Sd)
        & torch.isfinite(Kd)
        & torch.isfinite(Td)
        & torch.isfinite(rd)
        & torch.isfinite(qd)
        & torch.isfinite(sigmad)
        & torch.isfinite(p_denom)
        & torch.isfinite(p_raw)
        & (torch.abs(p_denom) > 1.0e-12)
        & (p_raw > 0.0)
        & (p_raw < 1.0)
    )
    p = torch.where(valid_regime, p_raw, torch.full_like(p_raw, 0.5))

    j_full = _crr_j_full(j_cache, steps_local, Sd)
    S_layer = Sd.unsqueeze(1) * (u.unsqueeze(1) ** j_full) * (d.unsqueeze(1) ** (steps_local - j_full))
    V = _crr_intrinsic_value(S_layer, Kd, is_calld)

    V_level1 = None
    V_level2 = None
    for i in reversed(range(steps_local)):
        V = disc.unsqueeze(1) * (p.unsqueeze(1) * V[:, 1:] + (1.0 - p).unsqueeze(1) * V[:, :-1])
        S_layer = S_layer[:, :-1] / d.unsqueeze(1)
        V = torch.maximum(V, _crr_intrinsic_value(S_layer, Kd, is_calld))
        if return_primary_greeks:
            if i == 2:
                V_level2 = V.clone()
            elif i == 1:
                V_level1 = V.clone()

    price_raw = V[:, 0]
    nan = torch.full_like(price_raw, float("nan"))
    price = torch.where(valid_regime, price_raw, nan)
    if not return_primary_greeks:
        return price

    delta = nan.clone()
    gamma = nan.clone()
    theta_per_day = nan.clone()

    if V_level1 is not None:
        S_down = Sd * d
        S_up = Sd * u
        delta_raw = _safe_div(V_level1[:, 1] - V_level1[:, 0], S_up - S_down)
        delta = torch.where(valid_regime, delta_raw, nan)

    if V_level2 is not None:
        S_dd = Sd * (d * d)
        S_mid = Sd
        S_uu = Sd * (u * u)
        delta_down = _safe_div(V_level2[:, 1] - V_level2[:, 0], S_mid - S_dd)
        delta_up = _safe_div(V_level2[:, 2] - V_level2[:, 1], S_uu - S_mid)
        gamma_raw = _safe_div(delta_up - delta_down, 0.5 * (S_uu - S_dd))
        theta_raw = _safe_div(V_level2[:, 1] - price_raw, 2.0 * dt) / 365.0
        gamma = torch.where(valid_regime, gamma_raw, nan)
        theta_per_day = torch.where(valid_regime, theta_raw, nan)

    return price, delta, gamma, theta_per_day


def _crr_full_greek_scenarios(
    spot: torch.Tensor,
    strike: torch.Tensor,
    tenor: torch.Tensor,
    rate: torch.Tensor,
    dividend: torch.Tensor,
    sigma: torch.Tensor,
    is_call: torch.Tensor,
    *,
    steps: int,
    j_cache: dict,
    fused: bool,
):
    """Evaluate the unchanged six-scenario Greek stencil."""
    tenor_bump = torch.full_like(tenor, 1.0 / 365.0)
    shorter_tenor = torch.clamp(tenor - tenor_bump, min=1e-8)
    effective_tenor_bump = torch.clamp(tenor - shorter_tenor, min=1e-8)
    sigma_bump = torch.full_like(sigma, 1e-4)
    sigma_down = torch.clamp(sigma - sigma_bump, min=1e-8)
    rate_bump = torch.full_like(rate, 1e-4)
    rate_down = torch.clamp(rate - rate_bump, min=-0.99)

    if fused:
        expanded_price, expanded_delta, expanded_gamma, _expanded_theta = (
            _crr_price_tensor(
                spot.repeat(6),
                strike.repeat(6),
                torch.cat(
                    (
                        tenor,
                        shorter_tenor,
                        tenor,
                        tenor,
                        tenor,
                        tenor,
                    )
                ),
                torch.cat(
                    (
                        rate,
                        rate,
                        rate,
                        rate,
                        rate + rate_bump,
                        rate_down,
                    )
                ),
                dividend.repeat(6),
                torch.cat(
                    (
                        sigma,
                        sigma,
                        sigma + sigma_bump,
                        sigma_down,
                        sigma,
                        sigma,
                    )
                ),
                is_call.repeat(6),
                steps_local=steps,
                j_cache=j_cache,
                return_primary_greeks=True,
            )
        )
        scenario_prices = expanded_price.reshape(6, -1)
        count = len(spot)
        base_price = scenario_prices[0]
        delta = expanded_delta[:count]
        gamma = expanded_gamma[:count]
        shorter_price = scenario_prices[1]
        sigma_up_price = scenario_prices[2]
        sigma_down_price = scenario_prices[3]
        rate_up_price = scenario_prices[4]
        rate_down_price = scenario_prices[5]
    else:
        base_price, delta, gamma, _tree_theta = _crr_price_tensor(
            spot,
            strike,
            tenor,
            rate,
            dividend,
            sigma,
            is_call,
            steps_local=steps,
            j_cache=j_cache,
            return_primary_greeks=True,
        )
        shorter_price = _crr_price_tensor(
            spot,
            strike,
            shorter_tenor,
            rate,
            dividend,
            sigma,
            is_call,
            steps_local=steps,
            j_cache=j_cache,
        )
        sigma_up_price = _crr_price_tensor(
            spot,
            strike,
            tenor,
            rate,
            dividend,
            sigma + sigma_bump,
            is_call,
            steps_local=steps,
            j_cache=j_cache,
        )
        sigma_down_price = _crr_price_tensor(
            spot,
            strike,
            tenor,
            rate,
            dividend,
            sigma_down,
            is_call,
            steps_local=steps,
            j_cache=j_cache,
        )
        rate_up_price = _crr_price_tensor(
            spot,
            strike,
            tenor,
            rate + rate_bump,
            dividend,
            sigma,
            is_call,
            steps_local=steps,
            j_cache=j_cache,
        )
        rate_down_price = _crr_price_tensor(
            spot,
            strike,
            tenor,
            rate_down,
            dividend,
            sigma,
            is_call,
            steps_local=steps,
            j_cache=j_cache,
        )
    return (
        base_price,
        delta,
        gamma,
        (shorter_price - base_price) / effective_tenor_bump / 365.0,
        (sigma_up_price - sigma_down_price) / (2.0 * sigma_bump),
        (rate_up_price - rate_down_price) / (2.0 * rate_bump),
    )


@torch.no_grad()
def compute_iv_greeks_crr(
    *,
    S: Union[np.ndarray, torch.Tensor],
    K: Union[np.ndarray, torch.Tensor],
    T: Union[np.ndarray, torch.Tensor],
    r: Union[np.ndarray, torch.Tensor],
    q: Union[np.ndarray, torch.Tensor],
    is_call: Union[np.ndarray, torch.Tensor],
    price_obs: Union[np.ndarray, torch.Tensor],
    steps: int = 256,
    tol: float = 1e-6,
    max_iter: int = 80,
    sigma_bounds=(1e-4, 5.0),
    dtype: torch.dtype = torch.float32,        # float32 saves ~2x memory vs float64
    device: Optional[torch.device] = None,     # defaults to cuda if available
    chunk_size: int = 50_000,                  # tune to your GPU RAM; 5-50k typical
    to_numpy: bool = True,                     # return numpy for easy pandas merge
    show_progress: bool = False,               # show chunk-level progress and ETA
    fused_greeks_enabled: bool = True,
    fused_greeks_maximum_contracts: int = 160,
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """
    Memory-safe, chunked American-option IV & Greeks using a CRR binomial tree.
    Processes inputs in slices of size `chunk_size` so GPU memory doesn't spike.

    All inputs are 1-D and aligned (same length N). Returns dict of arrays/tensors.
    """

    # -----------------------
    # Setup & input coercion
    # -----------------------
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    S_c      = np.ascontiguousarray(_to_np(S))
    K_c      = np.ascontiguousarray(_to_np(K))
    T_c      = np.ascontiguousarray(_to_np(T))
    r_c      = np.ascontiguousarray(_to_np(r))
    q_c      = np.ascontiguousarray(_to_np(q))
    call_c   = np.ascontiguousarray(_to_np(is_call).astype(bool))
    price_c  = np.ascontiguousarray(_to_np(price_obs))

    N = len(S_c)
    assert all(len(arr) == N for arr in (K_c, T_c, r_c, q_c, call_c, price_c)), "All inputs must have same length."
    fused_greeks_enabled = bool(fused_greeks_enabled) and (
        int(chunk_size) >= 6
        and (
            int(fused_greeks_maximum_contracts) <= 0
            or N <= int(fused_greeks_maximum_contracts)
        )
    )

    # Allocate outputs on CPU (memory-safe)
    iv_out            = np.empty(N, dtype=np.float32 if dtype==torch.float32 else np.float64)
    model_price_out   = np.empty_like(iv_out)
    delta_out         = np.empty_like(iv_out)
    gamma_out         = np.empty_like(iv_out)
    vega_out          = np.empty_like(iv_out)
    theta_day_out     = np.empty_like(iv_out)
    rho_out           = np.empty_like(iv_out)

    j_cache = {}

    # -----------------------
    # Chunked processing loop
    # -----------------------
    sigma_lo, sigma_hi = sigma_bounds
    out_dtype = np.float32 if dtype == torch.float32 else np.float64
    chunk_starts = list(range(0, N, int(chunk_size)))
    total_chunks = len(chunk_starts)
    progress_bar = None
    fallback_progress = bool(show_progress)
    progress_start = time.perf_counter()
    if show_progress:
        try:
            from tqdm.auto import tqdm
            progress_bar = tqdm(
                total=total_chunks,
                desc="compute_iv_greeks_crr",
                unit="chunk",
                dynamic_ncols=True,
            )
            fallback_progress = False
        except Exception:
            fallback_progress = True

    for chunk_idx, start in enumerate(chunk_starts):
        end = min(N, start + int(chunk_size))
        n_chunk = end - start

        # Slice CPU views then move once to target device
        Sx     = torch.as_tensor(S_c[start:end], dtype=dtype, device=device)
        Kx     = torch.as_tensor(K_c[start:end], dtype=dtype, device=device)
        Tx     = torch.as_tensor(T_c[start:end], dtype=dtype, device=device)
        rx     = torch.as_tensor(r_c[start:end], dtype=dtype, device=device)
        qx     = torch.as_tensor(q_c[start:end], dtype=dtype, device=device)
        callx  = torch.as_tensor(call_c[start:end], dtype=torch.bool, device=device)
        Px     = torch.as_tensor(price_c[start:end], dtype=dtype, device=device)

        # Keep chunk-local output arrays to preserve semantics for invalid IV rows.
        iv_chunk = np.full(n_chunk, np.nan, dtype=out_dtype)
        model_chunk = np.full(n_chunk, np.nan, dtype=out_dtype)
        delta_chunk = np.full(n_chunk, np.nan, dtype=out_dtype)
        gamma_chunk = np.full(n_chunk, np.nan, dtype=out_dtype)
        vega_chunk = np.full(n_chunk, np.nan, dtype=out_dtype)
        theta_chunk = np.full(n_chunk, np.nan, dtype=out_dtype)
        rho_chunk = np.full(n_chunk, np.nan, dtype=out_dtype)

        # --- IV via bisection (on device) ---
        sigma_floor = _crr_sigma_floor(Tx, rx, qx, steps=steps, sigma_lo=sigma_lo)
        lo = sigma_floor
        hi = torch.full_like(Px, sigma_hi)

        f_lo = _crr_price_tensor(Sx, Kx, Tx, rx, qx, lo, callx, steps_local=steps, j_cache=j_cache)
        f_hi = _crr_price_tensor(Sx, Kx, Tx, rx, qx, hi, callx, steps_local=steps, j_cache=j_cache)

        low_bound = torch.minimum(f_lo, f_hi)
        high_bound = torch.maximum(f_lo, f_hi)
        valid = (Px >= low_bound) & (Px <= high_bound)
        sigma = torch.full_like(Px, float("nan"))

        if bool(valid.any()):
            valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            Sv = Sx.index_select(0, valid_idx)
            Kv = Kx.index_select(0, valid_idx)
            Tv = Tx.index_select(0, valid_idx)
            rv = rx.index_select(0, valid_idx)
            qv = qx.index_select(0, valid_idx)
            callv = callx.index_select(0, valid_idx)
            Pv = Px.index_select(0, valid_idx)
            lv = low_bound.index_select(0, valid_idx)
            hv = high_bound.index_select(0, valid_idx)
            target_v = torch.clamp(Pv, lv, hv)

            lo_v = sigma_floor.index_select(0, valid_idx)
            hi_v = torch.full_like(Pv, sigma_hi)
            sigma_v = 0.5 * (lo_v + hi_v)
            for _ in range(max_iter):
                f_mid_v = _crr_price_tensor(Sv, Kv, Tv, rv, qv, sigma_v, callv, steps_local=steps, j_cache=j_cache)
                above_v = f_mid_v > target_v
                hi_v = torch.where(above_v, sigma_v, hi_v)
                lo_v = torch.where(~above_v, sigma_v, lo_v)
                sigma_v = 0.5 * (lo_v + hi_v)
                if torch.max(hi_v - lo_v) < tol:
                    break
            sigma.index_copy_(0, valid_idx, sigma_v)

        # --- Greeks with bump-and-reprice ---
        safe_sigma = torch.nan_to_num(sigma, nan=0.3)
        if fused_greeks_enabled:
            scenario_chunk_size = max(1, int(chunk_size) // 6)
            V0 = torch.full_like(Sx, float("nan"))
            delta = torch.full_like(Sx, float("nan"))
            gamma = torch.full_like(Sx, float("nan"))
            theta_per_day = torch.full_like(Sx, float("nan"))
            vega = torch.full_like(Sx, float("nan"))
            rho = torch.full_like(Sx, float("nan"))
            destinations = (V0, delta, gamma, theta_per_day, vega, rho)
            for scenario_start in range(0, n_chunk, scenario_chunk_size):
                scenario_end = min(n_chunk, scenario_start + scenario_chunk_size)
                scenario_values = _crr_full_greek_scenarios(
                    Sx[scenario_start:scenario_end],
                    Kx[scenario_start:scenario_end],
                    Tx[scenario_start:scenario_end],
                    rx[scenario_start:scenario_end],
                    qx[scenario_start:scenario_end],
                    safe_sigma[scenario_start:scenario_end],
                    callx[scenario_start:scenario_end],
                    steps=steps,
                    j_cache=j_cache,
                    fused=True,
                )
                for destination, value in zip(destinations, scenario_values):
                    destination[scenario_start:scenario_end] = value
        else:
            V0, delta, gamma, theta_per_day, vega, rho = (
                _crr_full_greek_scenarios(
                    Sx,
                    Kx,
                    Tx,
                    rx,
                    qx,
                    safe_sigma,
                    callx,
                    steps=steps,
                    j_cache=j_cache,
                    fused=False,
                )
            )

        # --- Copy results back to CPU arrays ---
        iv_chunk[:] = sigma.detach().cpu().numpy()
        model_chunk[:] = V0.detach().cpu().numpy()
        delta_chunk[:] = delta.detach().cpu().numpy()
        gamma_chunk[:] = gamma.detach().cpu().numpy()
        vega_chunk[:] = vega.detach().cpu().numpy()
        theta_chunk[:] = theta_per_day.detach().cpu().numpy()
        rho_chunk[:] = rho.detach().cpu().numpy()

        invalid_mask = ~np.isfinite(iv_chunk)
        if np.any(invalid_mask):
            model_chunk[invalid_mask] = np.nan
            delta_chunk[invalid_mask] = np.nan
            gamma_chunk[invalid_mask] = np.nan
            vega_chunk[invalid_mask] = np.nan
            theta_chunk[invalid_mask] = np.nan
            rho_chunk[invalid_mask] = np.nan

        iv_out[start:end]          = iv_chunk
        model_price_out[start:end] = model_chunk
        delta_out[start:end]       = delta_chunk
        gamma_out[start:end]       = gamma_chunk
        vega_out[start:end]        = vega_chunk
        theta_day_out[start:end]   = theta_chunk
        rho_out[start:end]         = rho_chunk

        # Free per-chunk tensors ASAP
        del Sx, Kx, Tx, rx, qx, callx, Px, lo, hi, f_lo, f_hi, sigma, safe_sigma
        del V0, delta, gamma, theta_per_day, vega, rho

        if progress_bar is not None:
            progress_bar.update(1)
        elif fallback_progress:
            elapsed = time.perf_counter() - progress_start
            completed = chunk_idx + 1
            pct = (completed / total_chunks) * 100.0 if total_chunks else 100.0
            eta = (elapsed / completed) * (total_chunks - completed) if completed else 0.0
            print(
                f"\rcompute_iv_greeks_crr: {pct:6.2f}% ({completed}/{total_chunks}) ETA {eta:7.1f}s",
                end="",
                flush=True,
            )

    if progress_bar is not None:
        progress_bar.close()
    elif fallback_progress:
        print("", flush=True)

    result = dict(
        iv=iv_out,
        model_price=model_price_out,
        delta=delta_out,
        gamma=gamma_out,
        vega=vega_out,
        theta_per_day=theta_day_out,
        rho=rho_out,
    )
    if not to_numpy:
        # convert to tensors on CPU
        return {k: torch.from_numpy(v) for k, v in result.items()}
    return result


@torch.no_grad()
def compute_prices_crr_from_iv(
    *,
    S: Union[np.ndarray, torch.Tensor],
    K: Union[np.ndarray, torch.Tensor],
    T: Union[np.ndarray, torch.Tensor],
    r: Union[np.ndarray, torch.Tensor],
    q: Union[np.ndarray, torch.Tensor],
    is_call: Union[np.ndarray, torch.Tensor],
    sigma: Union[np.ndarray, torch.Tensor],
    steps: int = 256,
    sigma_bounds=(1e-4, 10.0),
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    chunk_size: int = 50_000,
    to_numpy: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """Price aligned American options with the shared CRR tree kernel.

    Surface IV inversion uses this price-only entry point so it does not pay
    for Greek bump calculations and cannot inherit Phase 1 contract-selection
    limits.
    """
    if int(steps) <= 0 or int(chunk_size) <= 0:
        raise ValueError("steps and chunk_size must be positive")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    def _to_np(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    arrays = [np.ascontiguousarray(_to_np(value)) for value in (S, K, T, r, q, sigma)]
    calls = np.ascontiguousarray(_to_np(is_call).astype(bool))
    count = len(arrays[0])
    if not all(len(value) == count for value in (*arrays[1:], calls)):
        raise ValueError("All CRR price inputs must have the same length")
    out_dtype = np.float32 if dtype == torch.float32 else np.float64
    output = np.full(count, np.nan, dtype=out_dtype)
    sigma_lo, sigma_hi = map(float, sigma_bounds)
    j_cache = {}
    for start in range(0, count, int(chunk_size)):
        end = min(count, start + int(chunk_size))
        sx, kx, tx, rx, qx, sigmax = (
            torch.as_tensor(value[start:end], dtype=dtype, device=device)
            for value in arrays
        )
        callx = torch.as_tensor(calls[start:end], dtype=torch.bool, device=device)
        sigma_floor = _crr_sigma_floor(
            tx, rx, qx, steps=int(steps), sigma_lo=sigma_lo
        )
        valid = (
            torch.isfinite(sx)
            & torch.isfinite(kx)
            & torch.isfinite(tx)
            & torch.isfinite(rx)
            & torch.isfinite(qx)
            & torch.isfinite(sigmax)
            & (sx > 0.0)
            & (kx > 0.0)
            & (tx > 0.0)
            & (sigmax >= sigma_floor)
            & (sigmax <= sigma_hi)
        )
        safe_sigma = torch.where(valid, sigmax, torch.full_like(sigmax, 0.3))
        prices = _crr_price_tensor(
            sx,
            kx,
            tx,
            rx,
            qx,
            safe_sigma,
            callx,
            steps_local=int(steps),
            j_cache=j_cache,
        )
        prices = torch.where(valid, prices, torch.full_like(prices, float("nan")))
        output[start:end] = prices.detach().cpu().numpy()
    if to_numpy:
        return output
    return torch.from_numpy(output)


@torch.no_grad()
def compute_price_deltas_crr_from_iv(
    *,
    S: Union[np.ndarray, torch.Tensor],
    K: Union[np.ndarray, torch.Tensor],
    T: Union[np.ndarray, torch.Tensor],
    r: Union[np.ndarray, torch.Tensor],
    q: Union[np.ndarray, torch.Tensor],
    is_call: Union[np.ndarray, torch.Tensor],
    sigma: Union[np.ndarray, torch.Tensor],
    steps: int = 256,
    sigma_bounds=(1e-4, 10.0),
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    chunk_size: int = 50_000,
    to_numpy: bool = True,
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """Return CRR price and primary tree delta without bump-and-reprice Greeks."""
    if int(steps) <= 0 or int(chunk_size) <= 0:
        raise ValueError("steps and chunk_size must be positive")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    def _to_np(value):
        return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

    arrays = [np.ascontiguousarray(_to_np(value)) for value in (S, K, T, r, q, sigma)]
    calls = np.ascontiguousarray(_to_np(is_call).astype(bool))
    count = len(arrays[0])
    if not all(len(value) == count for value in (*arrays[1:], calls)):
        raise ValueError("All CRR price/delta inputs must have the same length")
    out_dtype = np.float32 if dtype == torch.float32 else np.float64
    prices_out = np.full(count, np.nan, dtype=out_dtype)
    deltas_out = np.full(count, np.nan, dtype=out_dtype)
    sigma_lo, sigma_hi = map(float, sigma_bounds)
    j_cache = {}
    for start in range(0, count, int(chunk_size)):
        end = min(count, start + int(chunk_size))
        sx, kx, tx, rx, qx, sigmax = (
            torch.as_tensor(value[start:end], dtype=dtype, device=device)
            for value in arrays
        )
        callx = torch.as_tensor(calls[start:end], dtype=torch.bool, device=device)
        sigma_floor = _crr_sigma_floor(tx, rx, qx, steps=int(steps), sigma_lo=sigma_lo)
        valid = (
            torch.isfinite(sx) & torch.isfinite(kx) & torch.isfinite(tx)
            & torch.isfinite(rx) & torch.isfinite(qx) & torch.isfinite(sigmax)
            & (sx > 0.0) & (kx > 0.0) & (tx > 0.0)
            & (sigmax >= sigma_floor) & (sigmax <= sigma_hi)
        )
        safe_sigma = torch.where(valid, sigmax, torch.full_like(sigmax, 0.3))
        prices, deltas, _gamma, _theta = _crr_price_tensor(
            sx, kx, tx, rx, qx, safe_sigma, callx,
            steps_local=int(steps), j_cache=j_cache,
            return_primary_greeks=True,
        )
        nan = torch.full_like(prices, float("nan"))
        prices_out[start:end] = torch.where(valid, prices, nan).detach().cpu().numpy()
        deltas_out[start:end] = torch.where(valid, deltas, nan).detach().cpu().numpy()
    result = {"price": prices_out, "delta": deltas_out}
    if to_numpy:
        return result
    return {name: torch.from_numpy(value) for name, value in result.items()}


@torch.no_grad()
def compute_greeks_crr_from_iv(
    *,
    S: Union[np.ndarray, torch.Tensor],
    K: Union[np.ndarray, torch.Tensor],
    T: Union[np.ndarray, torch.Tensor],
    r: Union[np.ndarray, torch.Tensor],
    q: Union[np.ndarray, torch.Tensor],
    is_call: Union[np.ndarray, torch.Tensor],
    sigma: Union[np.ndarray, torch.Tensor],
    steps: int = 256,
    sigma_bounds=(1e-4, 5.0),
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    chunk_size: int = 50_000,
    to_numpy: bool = True,
    show_progress: bool = False,
    fused_greeks_enabled: bool = True,
    fused_greeks_maximum_contracts: int = 160,
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    S_c = np.ascontiguousarray(_to_np(S))
    K_c = np.ascontiguousarray(_to_np(K))
    T_c = np.ascontiguousarray(_to_np(T))
    r_c = np.ascontiguousarray(_to_np(r))
    q_c = np.ascontiguousarray(_to_np(q))
    call_c = np.ascontiguousarray(_to_np(is_call).astype(bool))
    sigma_c = np.ascontiguousarray(_to_np(sigma))

    N = len(S_c)
    assert all(len(arr) == N for arr in (K_c, T_c, r_c, q_c, call_c, sigma_c)), "All inputs must have same length."
    fused_greeks_enabled = bool(fused_greeks_enabled) and (
        int(chunk_size) >= 6
        and (
            int(fused_greeks_maximum_contracts) <= 0
            or N <= int(fused_greeks_maximum_contracts)
        )
    )

    out_dtype = np.float32 if dtype == torch.float32 else np.float64
    iv_out = sigma_c.astype(out_dtype, copy=True)
    model_price_out = np.empty_like(iv_out)
    delta_out = np.empty_like(iv_out)
    gamma_out = np.empty_like(iv_out)
    vega_out = np.empty_like(iv_out)
    theta_day_out = np.empty_like(iv_out)
    rho_out = np.empty_like(iv_out)

    sigma_lo, sigma_hi = sigma_bounds
    logical_chunk_size = (
        max(1, int(chunk_size) // 6) if fused_greeks_enabled else int(chunk_size)
    )
    chunk_starts = list(range(0, N, logical_chunk_size))
    total_chunks = len(chunk_starts)
    progress_bar = None
    fallback_progress = bool(show_progress)
    progress_start = time.perf_counter()
    if show_progress:
        try:
            from tqdm.auto import tqdm

            progress_bar = tqdm(
                total=total_chunks,
                desc="compute_greeks_crr_from_iv",
                unit="chunk",
                dynamic_ncols=True,
            )
            fallback_progress = False
        except Exception:
            fallback_progress = True

    j_cache = {}

    for chunk_idx, start in enumerate(chunk_starts):
        end = min(N, start + logical_chunk_size)
        n_chunk = end - start

        Sx = torch.as_tensor(S_c[start:end], dtype=dtype, device=device)
        Kx = torch.as_tensor(K_c[start:end], dtype=dtype, device=device)
        Tx = torch.as_tensor(T_c[start:end], dtype=dtype, device=device)
        rx = torch.as_tensor(r_c[start:end], dtype=dtype, device=device)
        qx = torch.as_tensor(q_c[start:end], dtype=dtype, device=device)
        callx = torch.as_tensor(call_c[start:end], dtype=torch.bool, device=device)
        sigmax = torch.as_tensor(sigma_c[start:end], dtype=dtype, device=device)

        model_chunk = np.full(n_chunk, np.nan, dtype=out_dtype)
        delta_chunk = np.full(n_chunk, np.nan, dtype=out_dtype)
        gamma_chunk = np.full(n_chunk, np.nan, dtype=out_dtype)
        vega_chunk = np.full(n_chunk, np.nan, dtype=out_dtype)
        theta_chunk = np.full(n_chunk, np.nan, dtype=out_dtype)
        rho_chunk = np.full(n_chunk, np.nan, dtype=out_dtype)

        sigma_floor = _crr_sigma_floor(Tx, rx, qx, steps=steps, sigma_lo=sigma_lo)
        valid_sigma = torch.isfinite(sigmax) & (sigmax >= sigma_floor) & (sigmax <= sigma_hi)
        safe_sigma = torch.where(valid_sigma, sigmax, torch.full_like(sigmax, 0.3))
        V0, delta, gamma, theta_per_day, vega, rho = _crr_full_greek_scenarios(
            Sx,
            Kx,
            Tx,
            rx,
            qx,
            safe_sigma,
            callx,
            steps=steps,
            j_cache=j_cache,
            fused=fused_greeks_enabled,
        )

        model_chunk[:] = V0.detach().cpu().numpy()
        delta_chunk[:] = delta.detach().cpu().numpy()
        gamma_chunk[:] = gamma.detach().cpu().numpy()
        vega_chunk[:] = vega.detach().cpu().numpy()
        theta_chunk[:] = theta_per_day.detach().cpu().numpy()
        rho_chunk[:] = rho.detach().cpu().numpy()

        invalid_mask = ~np.isfinite(iv_out[start:end])
        if np.any(invalid_mask):
            model_chunk[invalid_mask] = np.nan
            delta_chunk[invalid_mask] = np.nan
            gamma_chunk[invalid_mask] = np.nan
            vega_chunk[invalid_mask] = np.nan
            theta_chunk[invalid_mask] = np.nan
            rho_chunk[invalid_mask] = np.nan

        model_price_out[start:end] = model_chunk
        delta_out[start:end] = delta_chunk
        gamma_out[start:end] = gamma_chunk
        vega_out[start:end] = vega_chunk
        theta_day_out[start:end] = theta_chunk
        rho_out[start:end] = rho_chunk

        del Sx, Kx, Tx, rx, qx, callx, sigmax, safe_sigma
        del V0, delta, gamma, theta_per_day, vega, rho

        if progress_bar is not None:
            progress_bar.update(1)
        elif fallback_progress:
            elapsed = time.perf_counter() - progress_start
            completed = chunk_idx + 1
            pct = (completed / total_chunks) * 100.0 if total_chunks else 100.0
            eta = (elapsed / completed) * (total_chunks - completed) if completed else 0.0
            print(
                f"\rcompute_greeks_crr_from_iv: {pct:6.2f}% ({completed}/{total_chunks}) ETA {eta:7.1f}s",
                end="",
                flush=True,
            )

    if progress_bar is not None:
        progress_bar.close()
    elif fallback_progress:
        print("", flush=True)

    result = dict(
        iv=iv_out,
        model_price=model_price_out,
        delta=delta_out,
        gamma=gamma_out,
        vega=vega_out,
        theta_per_day=theta_day_out,
        rho=rho_out,
    )
    if not to_numpy:
        return {k: torch.from_numpy(v) for k, v in result.items()}
    return result


# Engine-neutral batch aliases.  The copied functions above intentionally
# retain their established API so parity with external callers remains easy to
# audit, while new package code can use one vocabulary across all solvers.
def price_from_iv_batch(**kwargs):
    return compute_prices_crr_from_iv(**kwargs)


def price_delta_from_iv_batch(**kwargs):
    return compute_price_deltas_crr_from_iv(**kwargs)


def greeks_from_iv_batch(**kwargs):
    return compute_greeks_crr_from_iv(**kwargs)


def solve_iv_batch(**kwargs):
    return compute_iv_greeks_crr(**kwargs)["iv"]


def solve_iv_and_greeks_batch(**kwargs):
    return compute_iv_greeks_crr(**kwargs)
