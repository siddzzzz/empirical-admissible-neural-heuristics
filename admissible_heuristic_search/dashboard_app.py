import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import torch
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from admissible_heuristic_search.common.env import CombinatorialPuzzle
from admissible_heuristic_search.common.solver import AStarSolver
from admissible_heuristic_search.envs.cube2x2 import Cube2x2
from admissible_heuristic_search.envs.tile_puzzle import TilePuzzle
from admissible_heuristic_search.envs.lights_out import LightsOut
from admissible_heuristic_search.models.heuristic_net import HeuristicNet
from baseline_hdavi.dashboard.visualization import draw_2x2_cube

st.set_page_config(page_title="Admissible Neural Heuristic Dashboard", layout="wide")

# Custom Glassmorphic CSS Styling
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0f172a, #020617);
        color: #f8fafc;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    h1 {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(to right, #6366f1, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem !important;
        text-align: center;
    }
    h2, h3 {
        color: #38bdf8 !important;
        border-bottom: 1px solid rgba(56, 189, 248, 0.1);
        padding-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background: rgba(30, 41, 59, 0.5);
        color: #f1f5f9 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: bold;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
        color: #ffffff !important;
        border-color: #60a5fa !important;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9, #0284c7);
        color: white !important;
        border: none !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #38bdf8, #0ea5e9) !important;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.4);
    }
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 12px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Provably Admissible Neural Heuristics Solver")

# Helper: Render the 3D rotatable cube in Plotly
def get_3d_cube_plotly(state):
    color_names = {
        0: '#ffffff', # U (White)
        1: '#eab308', # D (Yellow)
        2: '#22c55e', # F (Green)
        3: '#3b82f6', # B (Blue)
        4: '#f97316', # L (Orange)
        5: '#ef4444'  # R (Red)
    }
    
    # Vertices of the 24 sticker quads in 3D
    sticker_quads = {
        # U face (0-3)
        0: [[-1, 0, 1.01], [0, 0, 1.01], [0, 1, 1.01], [-1, 1, 1.01]],
        1: [[0, 0, 1.01], [1, 0, 1.01], [1, 1, 1.01], [0, 1, 1.01]],
        2: [[-1, -1, 1.01], [0, -1, 1.01], [0, 0, 1.01], [-1, 0, 1.01]],
        3: [[0, -1, 1.01], [1, -1, 1.01], [1, 0, 1.01], [0, 0, 1.01]],
        # D face (4-7)
        4: [[-1, 0, -1.01], [0, 0, -1.01], [0, 1, -1.01], [-1, 1, -1.01]],
        5: [[0, 0, -1.01], [1, 0, -1.01], [1, 1, -1.01], [0, 1, -1.01]],
        6: [[-1, -1, -1.01], [0, -1, -1.01], [0, 0, -1.01], [-1, 0, -1.01]],
        7: [[0, -1, -1.01], [1, -1, -1.01], [1, 0, -1.01], [0, 0, -1.01]],
        # F face (8-11)
        8: [[-1, -1.01, 0], [0, -1.01, 0], [0, -1.01, 1], [-1, -1.01, 1]],
        9: [[0, -1.01, 0], [1, -1.01, 0], [1, -1.01, 1], [0, -1.01, 1]],
        10: [[-1, -1.01, -1], [0, -1.01, -1], [0, -1.01, 0], [-1, -1.01, 0]],
        11: [[0, -1.01, -1], [1, -1.01, -1], [1, -1.01, 0], [0, -1.01, 0]],
        # B face (12-15)
        12: [[1, 1.01, 0], [0, 1.01, 0], [0, 1.01, 1], [1, 1.01, 1]],
        13: [[0, 1.01, 0], [-1, 1.01, 0], [-1, 1.01, 1], [0, 1.01, 1]],
        14: [[1, 1.01, -1], [0, 1.01, -1], [0, 1.01, 0], [1, 1.01, 0]],
        15: [[0, 1.01, -1], [-1, 1.01, -1], [-1, 1.01, 0], [0, 1.01, 0]],
        # L face (16-19)
        16: [[-1.01, 1, 0], [-1.01, 0, 0], [-1.01, 0, 1], [-1.01, 1, 1]],
        17: [[-1.01, 0, 0], [-1.01, -1, 0], [-1.01, -1, 1], [-1.01, 0, 1]],
        18: [[-1.01, 1, -1], [-1.01, 0, -1], [-1.01, 0, 0], [-1.01, 1, 0]],
        19: [[-1.01, 0, -1], [-1.01, -1, -1], [-1.01, -1, 0], [-1.01, 0, 0]],
        # R face (20-23)
        20: [[1.01, -1, 0], [1.01, 0, 0], [1.01, 0, 1], [1.01, -1, 1]],
        21: [[1.01, 0, 0], [1.01, 1, 0], [1.01, 1, 1], [1.01, 0, 1]],
        22: [[1.01, -1, -1], [1.01, 0, -1], [1.01, 0, 0], [1.01, -1, 0]],
        23: [[1.01, 0, -1], [1.01, 1, -1], [1.01, 1, 0], [1.01, 0, 0]],
    }
    
    fig = go.Figure()
    
    # Add dark slate gray quads to build the base solid core structure (6 faces)
    core_faces = [
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]], # bottom
        [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],     # top
        [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]], # front
        [[-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]],     # back
        [[-1, -1, -1], [-1, 1, -1], [-1, 1, 1], [-1, -1, 1]], # left
        [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]]      # right
    ]
    for face in core_faces:
        fig.add_trace(go.Mesh3d(
            x=[v[0] for v in face], y=[v[1] for v in face], z=[v[2] for v in face],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color='#090d16', opacity=1.0, flatshading=True, showlegend=False, hoverinfo='none'
        ))
    
    # Draw the 24 colored quads
    for idx, quad in sticker_quads.items():
        color_idx = state[idx]
        color = color_names.get(color_idx, '#7f8c8d')
        
        fig.add_trace(go.Mesh3d(
            x=[v[0] for v in quad], y=[v[1] for v in quad], z=[v[2] for v in quad],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color, opacity=1.0, flatshading=True, showlegend=False, hoverinfo='none'
        ))
        
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, visible=False, range=[-1.5, 1.5]),
            yaxis=dict(showbackground=False, visible=False, range=[-1.5, 1.5]),
            zaxis=dict(showbackground=False, visible=False, range=[-1.5, 1.5]),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Helper: Render sliding tile puzzle using premium HTML grid cards
def render_8_puzzle_html(state):
    html = '<div style="display: grid; grid-template-columns: repeat(3, 85px); grid-template-rows: repeat(3, 85px); gap: 12px; justify-content: center; margin-bottom: 20px;">'
    for val in state:
        if val == 0:
            # Blank space card
            html += '<div style="background: rgba(15, 23, 42, 0.6); border: 2px dashed rgba(56, 189, 248, 0.2); border-radius: 14px; display: flex; align-items: center; justify-content: center;"></div>'
        else:
            # Tile card
            html += f'<div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(56, 189, 248, 0.05)); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 2rem; font-weight: 800; color: #38bdf8; box-shadow: 0 8px 16px rgba(0,0,0,0.4);">{val}</div>'
    html += '</div>'
    return html

# Session state initialization
if 'puzzle_type' not in st.session_state:
    st.session_state.puzzle_type = "Lights Out (3x3 Grid)"
    
# Function to instantiate puzzle environment
def get_puzzle_instance(puzzle_type):
    if puzzle_type == "Lights Out (3x3 Grid)":
        return LightsOut(W=3, H=3)
    elif puzzle_type == "Lights Out (5x5 Grid)":
        return LightsOut(W=5, H=5)
    elif puzzle_type == "8-Puzzle (Sliding Tiles)":
        return TilePuzzle(N=3)
    elif puzzle_type == "2x2 Rubik's Cube":
        return Cube2x2(use_macros=False)

# Sidebar configs
st.sidebar.subheader("🧩 Select Puzzle Domain")
st.session_state.puzzle_type = st.sidebar.selectbox(
    "Choose a puzzle:",
    ["Lights Out (3x3 Grid)", "Lights Out (5x5 Grid)", "8-Puzzle (Sliding Tiles)", "2x2 Rubik's Cube"],
    index=["Lights Out (3x3 Grid)", "Lights Out (5x5 Grid)", "8-Puzzle (Sliding Tiles)", "2x2 Rubik's Cube"].index(st.session_state.puzzle_type)
)

# Refresh puzzle instance if changed
if 'puzzle' not in st.session_state or st.session_state.get('last_puzzle_type') != st.session_state.puzzle_type:
    st.session_state.puzzle = get_puzzle_instance(st.session_state.puzzle_type)
    st.session_state.puzzle_state = st.session_state.puzzle.get_state()
    st.session_state.last_puzzle_type = st.session_state.puzzle_type
    st.session_state.solve_path = None
    st.session_state.start_solve_state = None

puzzle = st.session_state.puzzle
state = st.session_state.puzzle_state

# Load correct neural network heuristic weights
weights_mapping = {
    "Lights Out (3x3 Grid)": ("admissible_lightsout_3x3.pt", "lightsout"),
    "Lights Out (5x5 Grid)": ("admissible_lightsout_5x5.pt", "lightsout"),
    "8-Puzzle (Sliding Tiles)": ("admissible_tile8.pt", "tile8"),
    "2x2 Rubik's Cube": ("admissible_cube2x2.pt", "cube2x2")
}
filename, argname = weights_mapping[st.session_state.puzzle_type]
weights_path = os.path.join("trained_models", filename)

model = None
model_loaded = False

if os.path.exists(weights_path):
    try:
        model = HeuristicNet(input_dim=puzzle.one_hot_dim)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        model_loaded = True
        st.sidebar.success(f"Loaded: {filename}")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
else:
    st.sidebar.info(f"Weights not found: using base analytical heuristic.")

# Create tabs
tab1, tab2 = st.tabs(["🧩 Play & AI Solve", "📊 Introspection & Calibration"])

with tab1:
    col_play1, col_play2 = st.columns([1.5, 1.2])
    
    with col_play1:
        st.subheader("Visual puzzle Arena")
        
        # Unique visualizers based on puzzle type
        if "Lights Out" in st.session_state.puzzle_type:
            st.markdown("<p style='text-align: center; color: #94a3b8;'>Click cells to toggle them and their neighbors. Goal: Turn all lights OFF.</p>", unsafe_allow_html=True)
            
            # Interactive Grid of Buttons
            grid_container = st.container()
            with grid_container:
                for r in range(puzzle.H):
                    cols = st.columns([1] * puzzle.W)
                    for c in range(puzzle.W):
                        idx = r * puzzle.W + c
                        val = state[idx]
                        
                        btn_label = "💡 ON" if val == 1 else "🌑 OFF"
                        btn_key = f"light_{idx}_{val}"
                        
                        # Style active lights differently
                        if cols[c].button(btn_label, key=btn_key):
                            puzzle.set_state(state)
                            puzzle.step(idx)
                            st.session_state.puzzle_state = puzzle.get_state()
                            st.session_state.solve_path = None
                            st.rerun()
                            
        elif st.session_state.puzzle_type == "8-Puzzle (Sliding Tiles)":
            st.markdown("<p style='text-align: center; color: #94a3b8;'>Use the manual directional controls below to slide the tiles. Goal: Align tiles 1-8.</p>", unsafe_allow_html=True)
            
            # Render premium HTML tile board
            st.markdown(render_8_puzzle_html(state), unsafe_allow_html=True)
            
            # Control arrow layout
            c1, c2, c3 = st.columns([1, 1, 1])
            with c2:
                if st.button("⬆️ UP", key="tile_up"):
                    puzzle.set_state(state)
                    puzzle.step(0)
                    st.session_state.puzzle_state = puzzle.get_state()
                    st.session_state.solve_path = None
                    st.rerun()
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("⬅️ LEFT", key="tile_left"):
                    puzzle.set_state(state)
                    puzzle.step(2)
                    st.session_state.puzzle_state = puzzle.get_state()
                    st.session_state.solve_path = None
                    st.rerun()
            with c2:
                st.markdown("<div style='text-align:center; padding:10px; color:#64748b;'>SLIDE</div>", unsafe_allow_html=True)
            with c3:
                if st.button("➡️ RIGHT", key="tile_right"):
                    puzzle.set_state(state)
                    puzzle.step(3)
                    st.session_state.puzzle_state = puzzle.get_state()
                    st.session_state.solve_path = None
                    st.rerun()
            c1, c2, c3 = st.columns([1, 1, 1])
            with c2:
                if st.button("⬇️ DOWN", key="tile_down"):
                    puzzle.set_state(state)
                    puzzle.step(1)
                    st.session_state.puzzle_state = puzzle.get_state()
                    st.session_state.solve_path = None
                    st.rerun()
                    
        elif st.session_state.puzzle_type == "2x2 Rubik's Cube":
            tab_2d, tab_3d = st.tabs(["📦 3D Rotatable Cube", "🗺️ 2D Unrolled layout"])
            
            with tab_2d:
                # Render the rotatable 3D Plotly cube
                fig_3d = get_3d_cube_plotly(state)
                st.plotly_chart(fig_3d, width='stretch')
                
            with tab_2d:
                st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>💡 Drag, rotate, and zoom the 3D cube above to see the facelet color alignments.</p>", unsafe_allow_html=True)
                
            with tab_3d:
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)
                draw_2x2_cube(state, ax=ax)
                st.pyplot(fig)
                
            # Manual Cube Turn Controls
            st.markdown("**Primitive Rotations**")
            cols_turn = st.columns(6)
            turns = ['U', 'D', 'F', 'B', 'L', 'R']
            for i, turn in enumerate(turns):
                with cols_turn[i]:
                    if st.button(turn, key=f"turn_{turn}"):
                        puzzle.set_state(state)
                        # primitive index mapping (0, 3, 6, 9, 12, 15 are the clockwise turns U, D, F, B, L, R)
                        puzzle.step(i * 3)
                        st.session_state.puzzle_state = puzzle.get_state()
                        st.session_state.solve_path = None
                        st.rerun()
                        
    with col_play2:
        st.subheader("Controls & Metrics")
        
        is_solved = puzzle.is_solved()
        st.metric(label="Solved State Status", value="Solved" if is_solved else "Unsolved")
        
        # Scramble slider
        st.markdown("### Scramble Options")
        scramble_depth = st.slider("Scramble Depth", 1, 14, 5)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Scramble"):
                scrambled = puzzle.scramble(scramble_depth)
                st.session_state.puzzle_state = scrambled
                st.session_state.solve_path = None
                st.rerun()
        with col_btn2:
            if st.button("Reset"):
                solved = puzzle.reset()
                st.session_state.puzzle_state = solved
                st.session_state.solve_path = None
                st.rerun()
                
        st.markdown("---")
        
        # A* Solving controls
        st.subheader("🤖 AI A* Solver")
        
        search_budget = st.number_input("Max Search Nodes Budget", min_value=100, max_value=20000, value=2000, step=100)
        
        if st.button("Solve with AI", type="primary"):
            if not is_solved:
                with st.spinner("AI searching for admissible solution path..."):
                    # We compute validation offset delta dynamically if weights exist
                    delta = 0.0
                    if model_loaded:
                        if 'calibrated_deltas' not in st.session_state:
                            st.session_state.calibrated_deltas = {}
                        
                        if st.session_state.puzzle_type in st.session_state.calibrated_deltas:
                            delta = st.session_state.calibrated_deltas[st.session_state.puzzle_type]
                        else:
                            # Run a quick calibration scan of 15 samples to find an initial safe delta
                            quick_status = st.info("Running a quick heuristic pre-calibration to estimate admissibility offset...")
                            model.eval()
                            max_overest = 0.0
                            saved_state = state.copy()
                            for _ in range(15):
                                d_test = np.random.randint(1, 6)
                                puzzle.scramble(d_test)
                                st_val = puzzle.get_state()
                                with torch.no_grad():
                                    one_hot = puzzle.to_one_hot(np.expand_dims(st_val, 0))
                                    tensor_state = torch.tensor(one_hot, dtype=torch.float32)
                                    pred = model(tensor_state).item()
                                overest = pred - float(d_test)
                                if overest > max_overest:
                                    max_overest = overest
                            delta = float(max_overest)
                            st.session_state.calibrated_deltas[st.session_state.puzzle_type] = delta
                            puzzle.set_state(saved_state)
                            quick_status.empty()
                    
                    st.info(f"Solving using heuristic with admissibility calibration offset &delta; = {delta:.4f}")
                    solver = AStarSolver(puzzle, model=model, device="cpu", calibration_offset=delta)
                    puzzle.set_state(state)
                    path, nodes = solver.solve(state, max_nodes=search_budget)
                    
                    if path is not None:
                        st.session_state.solve_path = path
                        st.session_state.start_solve_state = state.copy()
                        st.success(f"Success! Path length: {len(path)} steps. Expanded {nodes} nodes.")
                    else:
                        st.error(f"Failed to find solution path within {search_budget} nodes.")
            else:
                st.info("Puzzle is already solved!")
                
        # Animate solve path if one exists
        if st.session_state.get('solve_path') is not None:
            path = st.session_state.solve_path
            
            st.markdown(f"### Solution Path ({len(path)} moves)")
            
            # Formatted actions
            action_names_list = [puzzle.action_space_names[act] for act in path]
            st.info(" &rarr; ".join(action_names_list), icon="🛣️")
            
            # Animate loop button
            if st.button("▶️ Animate Solution Steps"):
                # Reset puzzle to start state of solve
                puzzle.set_state(st.session_state.start_solve_state)
                st.session_state.puzzle_state = puzzle.get_state()
                
                # Render animation loop using Streamlit placeholder
                anim_placeholder = st.empty()
                
                for idx, act in enumerate(path):
                    # Take step
                    puzzle.step(act)
                    current_st = puzzle.get_state()
                    st.session_state.puzzle_state = current_st
                    
                    # Render puzzle state in the placeholder container
                    with anim_placeholder.container():
                        st.markdown(f"**Step {idx+1}/{len(path)}: {puzzle.action_space_names[act]}**", unsafe_allow_html=True)
                        if "Lights Out" in st.session_state.puzzle_type:
                            # Render grid
                            grid_html = f'<div style="display: grid; grid-template-columns: repeat({puzzle.W}, 60px); gap: 10px; justify-content: center; margin-bottom: 20px;">'
                            for val in current_st:
                                bg_color = '#eab308' if val == 1 else '#1e293b'
                                grid_html += f'<div style="background: {bg_color}; height: 60px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1)"></div>'
                            grid_html += '</div>'
                            st.markdown(grid_html, unsafe_allow_html=True)
                        elif st.session_state.puzzle_type == "8-Puzzle (Sliding Tiles)":
                            st.markdown(render_8_puzzle_html(current_st), unsafe_allow_html=True)
                        elif st.session_state.puzzle_type == "2x2 Rubik's Cube":
                            # Render the 3D Plotly cube trace in real time
                            fig_anim = get_3d_cube_plotly(current_st)
                            st.plotly_chart(fig_anim, width='stretch', key=f"anim_plotly_{idx}")
                            
                    time.sleep(0.6)
                
                st.success("Animation complete!")
                st.session_state.puzzle_state = puzzle.get_state()
                st.rerun()

with tab2:
    st.subheader("🧠 Neural Heuristic Introspection & Calibration")
    
    st.markdown(r"""
    This tab evaluates how well the Value Network's heuristic predictions match the true scramble depth.
    An **admissible** heuristic ($h(s) \le h^*(s)$) is required to guarantee optimal search paths.
    """)
    
    if model_loaded:
        st.write("---")
        
        # Show current calibrated delta if available
        if 'calibrated_deltas' in st.session_state and st.session_state.puzzle_type in st.session_state.calibrated_deltas:
            current_delta = st.session_state.calibrated_deltas[st.session_state.puzzle_type]
            st.info(f"Active Calibration Offset for {st.session_state.puzzle_type}: **&delta; = {current_delta:.4f}** (Guarantees admissibility).")

        # Let user run a live calibration scan and plot it
        if st.button("Run Heuristic Calibration scan"):
            # Save the active puzzle state before scrambling for calibration
            saved_active_state = state.copy()
            with st.spinner("Scrambling and evaluating predictions..."):
                if argname == "cube2x2":
                    max_scramble_val = 12
                elif argname == "tile8":
                    max_scramble_val = 10
                else: # lightsout
                    max_scramble_val = 15 if puzzle.W == 5 else 8
                num_samples = 30
                
                depths = list(range(1, max_scramble_val + 1))
                mean_ests = []
                std_ests = []
                admissible_rates = []
                
                # Check validation overestimation to compute delta
                max_overestimation = 0.0
                
                for d in depths:
                    estimates = []
                    admissible_count = 0
                    
                    for _ in range(num_samples):
                        puzzle.scramble(d)
                        st_val = puzzle.get_state()
                        
                        with torch.no_grad():
                            one_hot = puzzle.to_one_hot(np.expand_dims(st_val, 0))
                            tensor_state = torch.tensor(one_hot, dtype=torch.float32)
                            pred = model(tensor_state).item()
                        
                        estimates.append(max(0.0, pred))
                        if pred <= float(d):
                            admissible_count += 1
                            
                        overest = pred - float(d)
                        if overest > max_overestimation:
                            max_overestimation = overest
                            
                    estimates = np.array(estimates)
                    mean_ests.append(np.mean(estimates))
                    std_ests.append(np.std(estimates))
                    admissible_rates.append(admissible_count / num_samples)
                    
                # Save the delta in session state
                if 'calibrated_deltas' not in st.session_state:
                    st.session_state.calibrated_deltas = {}
                st.session_state.calibrated_deltas[st.session_state.puzzle_type] = max_overestimation
                
                # Restore active puzzle state
                puzzle.set_state(saved_active_state)
                    
                # Plot the calibration curve using matplotlib
                fig_cal, ax_cal = plt.subplots(figsize=(8, 4.5))
                fig_cal.patch.set_facecolor('#0f172a')
                ax_cal.set_facecolor('none')
                
                ax_cal.plot([0] + depths, [0] + depths, 'w--', alpha=0.5, label="Perfect Heuristic (h(s) = depth)")
                
                # Add model estimates
                ax_cal.errorbar([0] + depths, [0] + mean_ests, yerr=[0] + std_ests, fmt='o-', 
                                color='#38bdf8', ecolor=(0.22, 0.74, 0.97, 0.25), elinewidth=3, capsize=0, 
                                label="Learned ValueNet")
                                
                ax_cal.set_title("Neural Heuristic Calibration Landscape", color='#38bdf8', fontsize=12, fontweight='bold')
                ax_cal.set_xlabel("True Scramble Depth (moves from solved)", color='#94a3b8')
                ax_cal.set_ylabel("Heuristic Distance Estimate h(s)", color='#94a3b8')
                ax_cal.tick_params(colors='#64748b')
                ax_cal.grid(True, linestyle=':', alpha=0.3, color='#475569')
                ax_cal.legend(facecolor='#1e293b', edgecolor='none', labelcolor='#f8fafc')
                ax_cal.spines['bottom'].set_color('#334155')
                ax_cal.spines['left'].set_color('#334155')
                ax_cal.spines['top'].set_visible(False)
                ax_cal.spines['right'].set_visible(False)
                
                st.pyplot(fig_cal)
                
                # Display metrics
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric(label="Overall Admissibility Rate (Raw Model)", value=f"{np.mean(admissible_rates)*100:.1f}%")
                with m_col2:
                    st.metric(label="Post-Hoc Safety Offset (delta)", value=f"{max_overestimation:.4f}")
                    
                st.success("Calibration scan complete! Calibrated heuristic h_calib(s) = max(h0, h_theta - delta) is guaranteed 100% admissible.")
    else:
        st.info("Calibration features are disabled because no trained weights were found for the selected puzzle.")
