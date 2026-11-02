"""Math core for the Transportation Demand State Analyzer (no UI code)."""
import numpy as np


def prepare_matrix(P, label, normalize=False):
    P = np.array(P, dtype=float)
    if np.any(P < 0):
        raise ValueError(f"{label}: probabilities cannot be negative.")
    sums = P.sum(axis=1)
    if np.any(sums == 0):
        raise ValueError(f"{label}: a row of zeros has no outgoing movement.")
    if normalize:
        return P / sums[:, None]
    bad = [i + 1 for i, s in enumerate(sums) if abs(s - 1) > 1e-6]
    if bad:
        raise ValueError(f"{label}: row(s) {bad} do not sum to 1. Fix them or enable 'Normalize rows'.")
    return P


def eigen_analysis(P):
    """Left eigenvectors of P (right eigenvectors of P^T), sorted by |lambda|."""
    vals, vecs = np.linalg.eig(P.T)
    order = np.argsort(-np.abs(vals))
    vals, vecs = vals[order], vecs[:, order]
    unit = [i for i, v in enumerate(vals) if abs(v - 1) < 1e-8]
    on_circle = [i for i, v in enumerate(vals) if abs(abs(v) - 1) < 1e-8]
    steady = None
    if len(unit) == 1:
        v = np.clip(vecs[:, unit[0]].real, 0, None) if vecs[:, unit[0]].real.sum() > 0 \
            else np.clip(-vecs[:, unit[0]].real, 0, None)
        steady = v / v.sum()
    return {
        "eigenvalues": vals,
        "lambda2": float(abs(vals[1])) if len(vals) > 1 else 0.0,
        "unique_steady": len(unit) == 1,
        "periodic": len(on_circle) > len(unit),
    }, steady


def evolve(P, x0, steps, tol):
    xs, conv = [x0], None
    for k in range(1, steps + 1):
        nxt = xs[-1] @ P
        if conv is None and np.max(np.abs(nxt - xs[-1])) < tol:
            conv = k
        xs.append(nxt)
    return np.array(xs), conv


def analyze(P, x0, steps, tol):
    eig, steady = eigen_analysis(P)
    xs, conv = evolve(P, x0, steps, tol)
    note = ""
    if steady is None:
        steady = evolve(P, x0, 5000, tol)[0][-1]
        note = ("More than one eigenvalue equals 1, so the long-run split depends on the "
                "starting distribution. Showing the state after 5000 steps.")
    if eig["periodic"] and conv is None:
        note += " Passenger flow cycles and never settles (periodic chain)."
    err = 0.5 * np.abs(xs - steady).sum(axis=1)
    lam2 = eig["lambda2"]
    predicted = int(np.ceil(np.log(tol) / np.log(lam2))) if 0 < lam2 < 1 else None
    return {"P": P, "evolution": xs, "converged_at": conv, "predicted_steps": predicted,
            "steady": steady, "steady_check": float(np.abs(steady @ P - steady).max()),
            "error": err, "eigen": eig, "note": note.strip()}
