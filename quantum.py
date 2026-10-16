"""
Single-qubit Bloch sphere simulator and two-qubit Bell state demo.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector, entropy, partial_trace

st.set_page_config(page_title="Qubit Simulator", layout="wide")

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def wide_button(label, **kwargs):
    """Full-width button that works on both older and newer Streamlit versions."""
    try:
        return st.button(label, width="stretch", **kwargs)
    except TypeError:
        return st.button(label, use_container_width=True, **kwargs)


def clean(x, tol=1e-9):
    """Remove floating point noise such as -1e-17 so it prints as 0."""
    return 0.0 if abs(x) < tol else float(x)


def fmt_complex(c):
    re, im = clean(c.real), clean(c.imag)
    if im == 0:
        return f"{re:.3f}"
    if re == 0:
        return f"{im:.3f}i"
    return f"{re:.3f} {'+' if im > 0 else '-'} {abs(im):.3f}i"


def histogram_figure(labels, series, xtitle, height=340):
    """Grouped bar chart of probabilities. `series` maps legend name -> values."""
    fig = go.Figure()
    for name, values in series.items():
        fig.add_trace(
            go.Bar(
                x=labels,
                y=values,
                name=name,
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
            )
        )
    fig.update_layout(
        barmode="group",
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=len(series) > 1,
        legend=dict(orientation="h", y=1.15),
        xaxis=dict(title=xtitle, type="category"),
        yaxis=dict(title="Probability", range=[0, 1.15]),
    )
    return fig


# ---------------------------------------------------------------------------
# Section 1: single qubit on the Bloch sphere
# ---------------------------------------------------------------------------

GATES = {
    "H": ("Hadamard (H)", "Maps |0> to |+>. Equivalent to a pi rotation about the (x + z) axis."),
    "X": ("Pauli-X", "Bit flip. Rotates the state by pi about the x-axis."),
    "Y": ("Pauli-Y", "Rotates the state by pi about the y-axis."),
    "Z": ("Pauli-Z", "Phase flip. Rotates the state by pi about the z-axis."),
}


def push_gate(name):
    st.session_state.gates.append(name)


def undo_gate():
    if st.session_state.gates:
        st.session_state.gates.pop()


def reset_gates():
    st.session_state.gates = []


def run_gates(gates):
    """Apply the gate sequence with Qiskit and return the circuit plus every intermediate state."""
    qc = QuantumCircuit(1)
    states = [Statevector(qc)]  # |0>
    for g in gates:
        getattr(qc, g.lower())(0)
        states.append(Statevector(qc))
    return qc, states


def bloch_vector(sv):
    """(x, y, z) = expectation values of the Pauli X, Y, Z operators."""
    return np.array([sv.expectation_value(Pauli(p)).real for p in "XYZ"])


def bloch_angles(vec):
    """Polar angle theta in [0, pi], azimuth phi in [0, 2*pi)."""
    x, y, z = vec
    theta = float(np.arccos(np.clip(z, -1.0, 1.0)))
    on_pole = np.hypot(x, y) < 1e-9
    phi = 0.0 if on_pole else float(np.arctan2(y, x) % (2 * np.pi))
    return theta, phi, on_pole


def bloch_figure(states, gates):
    vecs = [bloch_vector(s) for s in states]
    x, y, z = vecs[-1]
    theta, phi, _ = bloch_angles(vecs[-1])

    fig = go.Figure()

    # Translucent unit sphere
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    fig.add_trace(
        go.Surface(
            x=np.outer(np.cos(u), np.sin(v)),
            y=np.outer(np.sin(u), np.sin(v)),
            z=np.outer(np.ones_like(u), np.cos(v)),
            opacity=0.15,
            showscale=False,
            colorscale=[[0, "#6c8ebf"], [1, "#6c8ebf"]],
            hoverinfo="skip",
        )
    )

    # Great circles: equator and two meridians
    t = np.linspace(0, 2 * np.pi, 200)
    zero = np.zeros_like(t)
    for cx, cy, cz in (
        (np.cos(t), np.sin(t), zero),
        (np.cos(t), zero, np.sin(t)),
        (zero, np.cos(t), np.sin(t)),
    ):
        fig.add_trace(
            go.Scatter3d(
                x=cx, y=cy, z=cz, mode="lines",
                line=dict(color="rgba(120,120,120,0.45)", width=2),
                hoverinfo="skip", showlegend=False,
            )
        )

    # Coordinate axes and basis-state labels
    for ax in ((1.25, 0, 0), (0, 1.25, 0), (0, 0, 1.25)):
        fig.add_trace(
            go.Scatter3d(
                x=[-ax[0], ax[0]], y=[-ax[1], ax[1]], z=[-ax[2], ax[2]], mode="lines",
                line=dict(color="rgba(90,90,90,0.8)", width=3),
                hoverinfo="skip", showlegend=False,
            )
        )
    label_pos = {
        "|+⟩": (1.42, 0, 0), "|−⟩": (-1.42, 0, 0),
        "|+i⟩": (0, 1.42, 0), "|−i⟩": (0, -1.42, 0),
        "|0⟩": (0, 0, 1.42), "|1⟩": (0, 0, -1.42),
    }
    fig.add_trace(
        go.Scatter3d(
            x=[p[0] for p in label_pos.values()],
            y=[p[1] for p in label_pos.values()],
            z=[p[2] for p in label_pos.values()],
            text=list(label_pos.keys()), mode="text",
            textfont=dict(size=15), hoverinfo="skip", showlegend=False,
        )
    )

    # Earlier states in the gate sequence
    if len(vecs) > 1:
        past = np.array(vecs[:-1])
        names = ["initial |0⟩"] + [f"after {g}" for g in gates]
        fig.add_trace(
            go.Scatter3d(
                x=past[:, 0], y=past[:, 1], z=past[:, 2], mode="markers",
                marker=dict(size=5, color="rgba(150,150,150,0.9)"),
                text=names[:-1], hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

    # Projection onto the xy-plane (helps read off phi)
    fig.add_trace(
        go.Scatter3d(
            x=[x, x, 0], y=[y, y, 0], z=[z, 0, 0], mode="lines",
            line=dict(color="rgba(220,60,60,0.6)", width=3, dash="dash"),
            hoverinfo="skip", showlegend=False,
        )
    )

    # State vector: shaft plus arrow head
    fig.add_trace(
        go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z], mode="lines",
            line=dict(color="crimson", width=8),
            hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Cone(
            x=[x], y=[y], z=[z], u=[x], v=[y], w=[z],
            anchor="tip", sizemode="absolute", sizeref=0.22, showscale=False,
            colorscale=[[0, "crimson"], [1, "crimson"]],
            hovertemplate=f"theta = {theta:.3f} rad<br>phi = {phi:.3f} rad<extra></extra>",
        )
    )

    axis = dict(visible=False, range=[-1.5, 1.5])
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        uirevision="bloch",  # keep the camera angle when the state changes
        scene=dict(
            xaxis=axis, yaxis=axis, zaxis=axis,
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.3, z=0.9)),
        ),
    )
    return fig


if "gates" not in st.session_state:
    st.session_state.gates = []

st.title("Qubit Simulator")
st.caption("Single-qubit Bloch sphere and a two-qubit Bell state, built with Qiskit, NumPy and Plotly.")

st.header("1. Single qubit on the Bloch sphere")
st.caption(
    "The qubit starts in |0>. Each button appends a gate to a Qiskit circuit; the Bloch vector is computed "
    "from the Pauli expectation values <X>, <Y>, <Z>. Drag the sphere to rotate it."
)

cols = st.columns([1, 1, 1, 1, 1.2, 1.2])
for col, (gate, (label, tip)) in zip(cols[:4], GATES.items()):
    with col:
        wide_button(label, key=f"gate_{gate}", on_click=push_gate, args=(gate,), help=tip)
with cols[4]:
    wide_button("Undo last gate", key="undo", on_click=undo_gate, disabled=not st.session_state.gates)
with cols[5]:
    wide_button("Reset to |0>", key="reset", on_click=reset_gates)

gates = st.session_state.gates
qc1, states1 = run_gates(gates)
sv1 = states1[-1]
vec = bloch_vector(sv1)
theta, phi, on_pole = bloch_angles(vec)

st.markdown("**Sequence:** " + " → ".join(["|0⟩"] + gates))

left, right = st.columns([3, 2])

with left:
    st.plotly_chart(bloch_figure(states1, gates), key="bloch_sphere", config={"displaylogo": False})

with right:
    st.subheader("State coordinates")
    table = pd.DataFrame(
        [
            {"Quantity": "theta", "Value": f"{theta:.4f} rad  ({np.degrees(theta):.2f} deg)"},
            {"Quantity": "phi", "Value": f"{phi:.4f} rad  ({np.degrees(phi):.2f} deg)"},
            {"Quantity": "x", "Value": f"{clean(vec[0]):+.4f}"},
            {"Quantity": "y", "Value": f"{clean(vec[1]):+.4f}"},
            {"Quantity": "z", "Value": f"{clean(vec[2]):+.4f}"},
        ]
    ).set_index("Quantity")
    st.table(table)
    if on_pole:
        st.caption("phi is undefined at the poles; it is shown as 0.")

    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)
    st.latex(rf"|\psi\rangle = {fmt_complex(alpha)}\,|0\rangle + ({fmt_complex(beta)})\,|1\rangle")
    st.caption("Global phase is dropped; it does not affect the Bloch vector or measurement results.")

    st.subheader("Measurement probabilities")
    p1 = sv1.probabilities()
    st.plotly_chart(
        histogram_figure(["0", "1"], {"Exact": p1}, "Measured bit", height=260),
        key="single_qubit_probs",
        config={"displaylogo": False},
    )

    if gates:
        st.subheader("Circuit")
        st.code(str(qc1.draw("text")), language=None)

st.divider()

# ---------------------------------------------------------------------------
# Section 2: CNOT and the Bell states
# ---------------------------------------------------------------------------

BELL_NAMES = {
    (0, 0): "Phi+ = (|00> + |11>) / sqrt(2)",
    (1, 0): "Phi- = (|00> - |11>) / sqrt(2)",
    (0, 1): "Psi+ = (|01> + |10>) / sqrt(2)",
    (1, 1): "Psi- = (|01> - |10>) / sqrt(2)",
}


def bell_circuit(q0_in, q1_in):
    """Prepare |q1 q0>, apply H on q0, then CNOT (q0 control, q1 target). Snapshot each stage."""
    qc = QuantumCircuit(2)
    stages = []
    if q0_in:
        qc.x(0)
    if q1_in:
        qc.x(1)
    stages.append(("Input state", Statevector(qc)))
    qc.barrier()
    qc.h(0)
    stages.append(("After H on q0", Statevector(qc)))
    qc.barrier()
    qc.cx(0, 1)
    stages.append(("After CNOT", Statevector(qc)))
    return qc, stages


if "seed" not in st.session_state:
    st.session_state.seed = 2024


def new_seed():
    st.session_state.seed = int(np.random.default_rng().integers(0, 2**31 - 1))


st.header("2. CNOT gate and Bell states")
st.caption(
    "A Hadamard on qubit 0 followed by a CNOT (qubit 0 controls, qubit 1 is the target) entangles the pair. "
    "Bitstrings follow Qiskit ordering, written as q1 q0."
)

left, right = st.columns([1, 2])

with left:
    inputs = st.radio(
        "Input basis state",
        options=[(0, 0), (1, 0), (0, 1), (1, 1)],
        format_func=lambda b: f"q0 = {b[0]}, q1 = {b[1]}",
        help="Different inputs lead to the four different Bell states.",
    )
    qc2, stages = bell_circuit(*inputs)
    st.markdown("**Circuit**")
    st.code(str(qc2.draw("text")), language=None)
    st.markdown("**Output after CNOT**")
    st.write(BELL_NAMES[inputs])

with right:
    stage_choice = st.radio(
        "Circuit stage to inspect",
        [name for name, _ in stages],
        index=2,
        horizontal=True,
    )
    stage_sv = dict(stages)[stage_choice]

    ctrl1, ctrl2 = st.columns([2, 1])
    with ctrl1:
        shots = st.select_slider("Measurement shots", options=[100, 500, 1000, 4000, 10000], value=1000)
    with ctrl2:
        st.write("")  # vertical alignment with the slider
        wide_button("Run measurement again", key="resample", on_click=new_seed)

    # Exact probabilities from the state vector; simulated shots sampled from them
    probs = np.clip(stage_sv.probabilities(), 0, None)
    probs = probs / probs.sum()
    rng = np.random.default_rng(st.session_state.seed)
    counts = rng.multinomial(shots, probs)

    bitstrings = [format(i, "02b") for i in range(4)]
    st.plotly_chart(
        histogram_figure(
            bitstrings,
            {"Exact probability": probs, f"Measured ({shots} shots)": counts / shots},
            "Measured bitstring (q1 q0)",
        ),
        key="bell_hist",
        config={"displaylogo": False},
    )
    st.caption("Counts: " + ", ".join(f"{b}: {c}" for b, c in zip(bitstrings, counts)))

    s_ent = max(0.0, clean(float(entropy(partial_trace(stage_sv, [1]), base=2))))
    m1, m2 = st.columns(2)
    m1.metric("Entanglement entropy of q0", f"{s_ent:.3f} bits")
    m2.metric("State", "Maximally entangled" if s_ent > 0.999 else ("Product state" if s_ent < 1e-6 else "Partially entangled"))

    st.markdown("**State vector amplitudes**")
    amp_table = pd.DataFrame(
        [
            {
                "Basis (q1 q0)": format(i, "02b"),
                "Amplitude": fmt_complex(amp),
                "Probability": f"{abs(amp) ** 2:.3f}",
            }
            for i, amp in enumerate(stage_sv.data)
        ]
    ).set_index("Basis (q1 q0)")
    st.table(amp_table)