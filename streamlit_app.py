## 2D TOPOLOGY OPTIMIZATION APP

## Imports
import os, importlib.util
os.environ["MPLBACKEND"] = "Agg"  # headless matplotlib
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

## Layout
st.set_page_config(page_title="Topology Optimization of a Cantilevered Beam", layout="wide")
st.title("Topology Optimization of a Cantilevered Beam")
left, right = st.columns([7, 7], gap="large")


## Boundary Condition Plot
def bc_preview_with_grid(nelx: int, nely: int, show_grid: bool = True) -> go.Figure:
    nelx, nely = int(nelx), int(nely)
    max_cells_for_pixels = 90_000
    fig = go.Figure()

    # Background grid
    if show_grid and (nelx * nely) <= max_cells_for_pixels:
        Z = np.zeros((nely, nelx), dtype=np.uint8)
        fig.add_trace(go.Heatmap(
            z=Z,
            colorscale=[(0, "#ffffff")],
            showscale=False,
            hoverinfo="skip",
            xgap=1, ygap=1
        ))
    else:
        fig.add_shape(type="rect", x0=0, x1=nelx, y0=0, y1=nely,
                      line=dict(color="#000000", width=1),
                      fillcolor="#000000")
        step_x = max(1, nelx // 50 or 1)
        step_y = max(1, nely // 30 or 1)
        fig.update_xaxes(showgrid=True, gridcolor="#000000", dtick=step_x)
        fig.update_yaxes(showgrid=True, gridcolor="#000000", dtick=step_y)

    # Fixed
    fig.add_shape(type="rect", x0=-1, x1=0, y0=-1, y1=nely,
                  line=dict(width=0), fillcolor="#FF9100", layer="above")

        
    x_load = nelx - 0.5
    y_mid  = nely / 2.0

    L       = max(3.0, 0.25 * nely)     # shaft length 
    head_h  = max(2.0, 0.1 * nely)     # triangle height 
    head_w  = max(0.8, 0.05 * nely)     # half of base width 

    # geometry
    tip_y   = y_mid - L                 # arrow tip (downward)
    base_y  = tip_y + head_h            # y of triangle base
    # 1) shaft stops at the base center (not at the tip)
    fig.add_trace(go.Scatter(
        x=[x_load, x_load],
        y=[y_mid,  base_y],
        mode="lines",
        line=dict(color="#f60404", width=6),
        hoverinfo="skip", showlegend=False
    ))

    # 2) head: crisp filled triangle (tip at tip_y, base at base_y ± head_w)
    fig.add_shape(
        type="path",
        path=f"M {x_load},{tip_y} L {x_load - head_w},{base_y} L {x_load + head_w},{base_y} Z",
        line=dict(width=0),
        fillcolor="#f60404",
        layer="above"
    )

    # Dot (on top)
    fig.add_trace(go.Scatter(
        x=[x_load], y=[y_mid],
        mode="markers",
        marker=dict(size=12, color="#f60404", line=dict(color="#000000", width=2)),
        hoverinfo="skip", showlegend=False
    ))

    # Padding so arrowhead never clips; fixed height for stability
    pad_x = max(0.04 * nelx, 1.2 * head_w)
    pad_y = max(0.04 * nely, 1.2 * head_w)
    fig.update_xaxes(range=[-pad_x, nelx + pad_x], visible=False, constrain="domain")
    fig.update_yaxes(range=[-pad_y, nely + pad_y], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320,
                      paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")
    return fig

## Left Column Geometry and Images
with left:
    st.subheader("Geometry")
    size_col1, size_col2 = st.columns(2)
    with size_col1:
        nely = st.number_input("nely", 10, 80, 30, step=5)
    with size_col2:
        nelx = st.number_input("nelx", 10, 80, 50, step=5)

    # BC preview
    st.plotly_chart(
        bc_preview_with_grid(int(nelx), int(nely), show_grid=True),
        use_container_width=True
    )

    # Reserved result panel (doesn't jump)
    result_box = st.container()
    with result_box:
        st.info("Result will appear here after you click **Run TopOpt**.")


## Pane set up Helpers
def pane_slider(title, *, min_value, max_value, value, step, definition_p, key):
    try:
        box = st.container(border=True)   # Streamlit ≥1.32
    except TypeError:
        box = st.container()
    with box:
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"**{title}**")
            val = st.slider(
                " ", min_value=min_value, max_value=max_value,
                value=value, step=step, key=key, label_visibility="collapsed"
            )
        with right:
            st.caption(definition_p)
    return val

def pane_select(title, *, options, index, definition_p, key):
    try:
        box = st.container(border=True)
    except TypeError:
        box = st.container()
    with box:
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"**{title}**")
            val = st.selectbox(
                " ", options=options, index=index, key=key,
                label_visibility="collapsed"
            )
        with right:
            st.caption(definition_p)
    return val

## Right Column Parameter Selection and Buttons

with right:
    st.subheader("Parameters")

    penal = pane_slider(
        "Penalization (SIMP exponent)",
        min_value=1.0, max_value=5.0, value=3.0, step=0.1, key="penal",
        definition_p="Higher → crisper 0/1 material. Typical: 3.0"
    )

    volfrac = pane_slider(
        "Volume Fraction",
        min_value=0.05, max_value=0.95, value=0.60, step=0.01, key="volfrac",
        definition_p="Target material ratio (0–1); e.g., 0.40 = 40% solid."
    )

    rmin = pane_slider(
        "Filter radius (elements)",
        min_value=1.0, max_value=10.0, value=3.5, step=0.1, key="rmin",
        definition_p="Minimum feature size control in element units."
    )
    ft=1

    run = st.button("Run TopOpt", type="primary", use_container_width=False)

    st.subheader("Python Sources")
    st.link_button("TopOpt in Python", "https://www.topopt.mek.dtu.dk/apps-and-software/topology-optimization-codes-written-in-python", use_container_width=True)
    st.link_button("Cholesky Modified Verion", "https://cvxopt.org/download/", use_container_width=True)

## Image Prep

def _prep_matplotlib_hooks(capture):
    import matplotlib
    matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt
    # Nuke interactive/redraw calls the DTU code makes inside the loop:
    plt.ion   = lambda *a, **k: None
    plt.show  = lambda *a, **k: None
    plt.pause = lambda *a, **k: None   # <- IMPORTANT
    plt.draw  = lambda *a, **k: None   # <- IMPORTANT

    # Capture Axes.imshow so we can grab the final density image it plots
    from matplotlib.axes import Axes
    orig_imshow = Axes.imshow

    def imshow_hook(self, *args, **kwargs):
        img = orig_imshow(self, *args, **kwargs)
        capture["img"] = img
        return img

    Axes.imshow = imshow_hook
    return Axes, orig_imshow


@st.cache_resource(show_spinner=False)
def _load_topopt_cholmod_module():
    """Load 'topopt_cholmod.py' without executing its __main__ block."""
    import importlib.util, types, re, pathlib
    path = os.path.join(os.path.dirname(__file__), "topopt_cholmod.py")
    if not os.path.isfile(path):
        raise RuntimeError("Place topopt_cholmod.py next to this file.")

    # Read source and neuter the __main__ block if present
    src = pathlib.Path(path).read_text(encoding="utf-8")

    # Robustly disable the main-run block (covers various spacing/quotes)
    pattern = r'if\s+__name__\s*==\s*["\']__main__["\']\s*:\s*'
    src = re.sub(pattern, "if False:\n    # disabled by streamlit loader\n", src)

    # Numpy 2.x compat for legacy code using np.int
    setattr(np, "int", int)

    # Execute into a fresh module namespace (no file changes on disk)
    mod = types.ModuleType("topopt_cholmod")
    mod.__file__ = path
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod



def run_cholmod_topopt(nelx, nely, volfrac, penal, rmin, ft):
    capture = {"img": None}
    Axes, orig_imshow = _prep_matplotlib_hooks(capture)
    try:
        mod = _load_topopt_cholmod_module()
        mod.raw_input = lambda *a, **k: None  # avoid blocking at the end
        mod.main(int(nelx), int(nely), float(volfrac), float(penal), float(rmin), int(ft))

        if capture["img"] is None:
            raise RuntimeError("Could not capture image; the script may not have called imshow().")

        arr  = np.array(capture["img"].get_array())  # DTU plots -xPhys.reshape(nelx,nely).T
        dens = -arr
        if dens.shape != (int(nely), int(nelx)):
            dens = dens.reshape((int(nely), int(nelx)))
        return dens
    finally:
        Axes.imshow = orig_imshow


## Run
if run:
    with result_box:
        try:
            with st.spinner("Running…"):
                dens = run_cholmod_topopt(nelx, nely, volfrac, penal, rmin, ft)
            fig = px.imshow(
                dens, origin="lower", aspect="equal",
                zmin=0.0, zmax=1.0, color_continuous_scale="Viridis",
                labels=dict(color="ρ")
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                coloraxis_showscale=False,
                xaxis_title="x (elements)", yaxis_title="y (elements)",
                height=360
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"nelx={nelx}, nely={nely}, volfrac={volfrac:.2f}, penal={penal:.2f}, rmin={rmin:.2f}, ft={ft}")
        except Exception as e:
            st.error(str(e))
            st.exception(e)

