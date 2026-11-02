"""
Transportation Demand State Analyzer - Streamlit app

Passenger data -> Transition matrix -> Initial state vector -> Repeated transformations
-> State evolution -> Convergence -> Steady-state vector -> Eigen analysis -> Model comparison

Run:  pip install -r requirements.txt && streamlit run app.py
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from markov_core import analyze, prepare_matrix

st.set_page_config(page_title="Transportation Demand State Analyzer", layout="wide")

EXAMPLE = {
    "names": ["Downtown", "Suburb", "Airport"],
    "init": [5000, 3000, 2000],
    "A": [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.3, 0.3, 0.4]],
    "B": [[0.5, 0.3, 0.2], [0.15, 0.75, 0.1], [0.2, 0.2, 0.6]],
}

# ---------------- sidebar: settings ----------------
st.sidebar.title("Settings")
preset = st.sidebar.radio("Starting data", ["Example (3 zones)", "Blank"])
n = 3 if preset.startswith("Example") else st.sidebar.slider("Number of states", 2, 6, 3)
steps = st.sidebar.slider("Time steps", 1, 200, 30)
tol = st.sidebar.select_slider("Convergence tolerance", [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8], value=1e-4,
                               format_func=lambda v: f"{v:g}")
normalize = st.sidebar.checkbox("Normalize rows", help="Rescale every row of the matrices so it sums to 1.")
st.sidebar.caption("Edit the tables on the Inputs tab, then read the other tabs.")

st.title("Transportation demand state analyzer")
st.write("Follow passengers moving between stations or zones and compare the long-run demand of two transition models.")

if preset.startswith("Example"):
    names0, init0, A0, B0 = EXAMPLE["names"], EXAMPLE["init"], EXAMPLE["A"], EXAMPLE["B"]
else:
    names0 = [f"State {i+1}" for i in range(n)]
    init0 = [100] * n
    A0 = B0 = np.eye(n).tolist()

tab_in, tab_evo, tab_conv, tab_eig, tab_cmp = st.tabs(
    ["Inputs", "State evolution", "Convergence", "Steady state and eigen analysis", "Model comparison"])

# ---------------- inputs ----------------
with tab_in:
    key = f"{preset}-{n}"
    labels = [f"S{i+1}" for i in range(n)]
    st.subheader("Passenger data")
    st.caption("Name each state and enter how many passengers start there.")
    base = pd.DataFrame({"State name": names0, "Initial passengers": init0}, index=labels)
    init_df = st.data_editor(base, key=f"init-{key}", use_container_width=False)
    names = [str(v).strip() or f"State {i+1}" for i, v in enumerate(init_df["State name"])]

    st.subheader("Transition matrices")
    st.caption("S1, S2, ... follow the order of the states above. Row = from, column = to. Each row is the probability of a passenger moving to each state next step.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Model A**")
        A_df = st.data_editor(pd.DataFrame(A0, index=labels, columns=labels), key=f"A-{key}", use_container_width=True)
    with c2:
        st.markdown("**Model B**")
        B_df = st.data_editor(pd.DataFrame(B0, index=labels, columns=labels), key=f"B-{key}", use_container_width=True)

# ---------------- compute ----------------
try:
    counts = init_df["Initial passengers"].to_numpy(dtype=float)
    if np.any(counts < 0) or counts.sum() <= 0 or np.isnan(counts).any():
        raise ValueError("Starting passengers must be non-negative and not all zero.")
    total = counts.sum()
    x0 = counts / total
    PA = prepare_matrix(A_df.to_numpy(dtype=float), "Model A", normalize)
    PB = prepare_matrix(B_df.to_numpy(dtype=float), "Model B", normalize)
    R = {"A": analyze(PA, x0, steps, tol), "B": analyze(PB, x0, steps, tol)}
except (ValueError, TypeError) as e:
    for t in (tab_evo, tab_conv, tab_eig, tab_cmp):
        t.error(str(e))
    st.stop()

with tab_in:
    st.info(f"Initial state vector x₀ = {np.round(x0, 4).tolist()} from {total:,.0f} passengers. "
            "Each step applies x(k+1) = x(k) · P.")

# ---------------- evolution ----------------
with tab_evo:
    model = st.radio("Model", ["A", "B"], horizontal=True, key="evo_model")
    evo = pd.DataFrame(R[model]["evolution"], columns=names)
    evo.index.name = "Step"
    st.line_chart(evo)
    show = st.radio("Show as", ["Share of passengers", "Passenger counts"], horizontal=True)
    table = evo * total if show == "Passenger counts" else evo
    st.dataframe(table.style.format("{:,.1f}" if show == "Passenger counts" else "{:.2%}"), use_container_width=True)
    st.download_button("Download evolution (CSV)", evo.to_csv().encode(), f"evolution_model_{model}.csv", "text/csv")

# ---------------- convergence ----------------
with tab_conv:
    err = pd.DataFrame({"Step": range(steps + 1),
                        "Model A": np.clip(R["A"]["error"], 1e-12, None),
                        "Model B": np.clip(R["B"]["error"], 1e-12, None)}).melt("Step", var_name="Model", value_name="Distance")
    st.altair_chart(alt.Chart(err).mark_line().encode(
        x="Step:Q", y=alt.Y("Distance:Q", scale=alt.Scale(type="log"), title="Distance from steady state"),
        color="Model:N").properties(height=320), use_container_width=True)
    st.caption("Total variation distance on a log scale. A steeper drop means faster convergence.")
    cols = st.columns(4)
    a, b = R["A"], R["B"]
    cols[0].metric("Model A converged at step", a["converged_at"] or f"> {steps}")
    cols[1].metric("Model A predicted steps", a["predicted_steps"] or "n/a")
    cols[2].metric("Model B converged at step", b["converged_at"] or f"> {steps}")
    cols[3].metric("Model B predicted steps", b["predicted_steps"] or "n/a")
    st.caption("Predicted steps come from ln(tolerance) / ln(|λ₂|).")

# ---------------- steady state + eigen ----------------
with tab_eig:
    cols = st.columns(2)
    for col, k in zip(cols, ["A", "B"]):
        m = R[k]
        with col:
            st.markdown(f"#### Model {k}")
            ss = pd.DataFrame({"Share": m["steady"], "Passengers": m["steady"] * total}, index=names)
            st.dataframe(ss.style.format({"Share": "{:.2%}", "Passengers": "{:,.0f}"}), use_container_width=True)
            st.bar_chart(ss["Share"])
            st.caption(f"Check max|πP − π| = {m['steady_check']:.2e}. "
                       + ("Unique steady state: the eigenvector for λ = 1, scaled to sum to 1." if m["eigen"]["unique_steady"] else ""))
            if m["note"]:
                st.warning(m["note"])
            ev = m["eigen"]["eigenvalues"]
            st.dataframe(pd.DataFrame({"Re": ev.real, "Im": ev.imag, "|λ|": np.abs(ev)},
                                      index=range(1, len(ev) + 1)).round(4), use_container_width=True)
            st.caption(f"|λ₂| = {m['eigen']['lambda2']:.4f}. The closer to 0, the faster passengers forget where they started.")

# ---------------- comparison ----------------
with tab_cmp:
    sa, sb = R["A"]["steady"], R["B"]["steady"]
    diff = sb - sa
    cmp = pd.DataFrame({"Model A": sa, "Model B": sb, "Change (B − A), pts": diff * 100,
                        "Passenger change": diff * total}, index=names)
    st.dataframe(cmp.style.format({"Model A": "{:.2%}", "Model B": "{:.2%}",
                                   "Change (B − A), pts": "{:+.2f}", "Passenger change": "{:+,.0f}"}),
                 use_container_width=True)
    st.bar_chart(cmp[["Model A", "Model B"]])
    c = st.columns(4)
    c[0].metric("Busiest state, A", names[int(np.argmax(sa))])
    c[1].metric("Busiest state, B", names[int(np.argmax(sb))])
    c[2].metric("Biggest gain under B", names[int(np.argmax(diff))])
    c[3].metric("Biggest loss under B", names[int(np.argmin(diff))])
    la, lb = R["A"]["eigen"]["lambda2"], R["B"]["eigen"]["lambda2"]
    faster = "Same speed" if np.isclose(la, lb) else ("Model A" if la < lb else "Model B")
    st.write(f"Distance between steady states (L1): **{np.abs(diff).sum():.4f}**. Faster-converging model: **{faster}**.")
