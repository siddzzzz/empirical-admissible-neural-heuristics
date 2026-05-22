import streamlit as st
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

# Add parent directory to path so we can import env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.cube import Cube2x2
from env.rewards import calculate_entropy
from dashboard.visualization import draw_2x2_cube, draw_saliency_heatmap
from agents.tactical.value_agent import ValueNet, load_model
from training.solver import AStarSolver
from scan_brains import get_saliency, profile_heuristic, generate_calibration_plot

st.set_page_config(page_title="Rubik's Cube H-DAVI Solver", layout="wide")

# Custom premium styling
st.markdown("""
<style>
    /* Dark glassmorphic theme */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #111827, #030712);
        color: #f3f4f6;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    h1 {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(to right, #818cf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem !important;
        text-align: center;
    }
    h2, h3 {
        color: #a5b4fc !important;
        border-bottom: 1px solid rgba(165, 180, 252, 0.1);
        padding-bottom: 0.5rem;
    }
    /* Buttons styling */
    .stButton>button {
        width: 100%;
        background: rgba(31, 41, 55, 0.5);
        color: #e5e7eb !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4f46e5, #4338ca) !important;
        color: #ffffff !important;
        border-color: #6366f1 !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    /* Highlight AI controls */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white !important;
        border: none !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #34d399, #10b981) !important;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);
    }
    /* Metric styling */
    div[data-testid="metric-container"] {
        background: rgba(31, 41, 55, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    /* History tag styling */
    .macro-move {
        color: #34d399;
        font-weight: bold;
    }
    .primitive-move {
        color: #fbbf24;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧩 Hierarchical H-DAVI Rubik's Cube Solver")

# Session state initialization
if 'cube' not in st.session_state:
    st.session_state.cube = Cube2x2()
    st.session_state.history = []
    st.session_state.last_solve_details = None

model_path = "trained_models/value_agent.pt"
model_loaded = False

# Load model/solver globally into session state if available
if os.path.exists(model_path):
    if 'solver' not in st.session_state:
        try:
            model = load_model(model_path, device="cpu")
            st.session_state.solver = AStarSolver(model, device="cpu", use_macros=True)
            st.session_state.load_error = None
        except Exception as e:
            st.session_state.load_error = str(e)
            
    if st.session_state.get('load_error') is None:
        model_loaded = True

# Create Streamlit tabs
tab1, tab2 = st.tabs(["🧩 Solver Dashboard", "🧠 Neural Brain Introspection"])

with tab1:
    col1, col2 = st.columns([1.8, 1.2])
    
    with col1:
        st.subheader("Cube Representation")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        draw_2x2_cube(st.session_state.cube.state, ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.subheader("State Metrics & Controls")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            entropy = calculate_entropy(st.session_state.cube.state)
            st.metric(label="Cube Entropy (Disorder)", value=f"{entropy:.2f}")
        with m_col2:
            is_solved = st.session_state.cube.is_solved()
            st.metric(label="Is Solved?", value="Solved" if is_solved else "Unsolved")
            
        st.markdown("### Scramble Options")
        scramble_moves = st.slider("Number of random moves", 1, 20, 5)
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Scramble"):
                st.session_state.cube.scramble(scramble_moves)
                st.session_state.history = []
                st.session_state.last_solve_details = None
                st.rerun()
        with btn_col2:
            if st.button("Reset / Solved"):
                st.session_state.cube.reset()
                st.session_state.history = []
                st.session_state.last_solve_details = None
                st.rerun()
                
    st.markdown("---")
    
    st.subheader("Take Action (Manual Control)")
    actions = st.session_state.cube.action_space_names
    primitives = actions[:18]
    macros = actions[18:]
    
    st.markdown("**Primitive Rotations (90° & 180°)**")
    prim_cols = st.columns(9)
    for i, action in enumerate(primitives):
        col = prim_cols[i % 9]
        with col:
            if st.button(action, key=f"prim_{action}"):
                st.session_state.cube.step(i)
                st.session_state.history.append(f"<span class='primitive-move'>{action}</span>")
                st.rerun()
                
    st.markdown("**Compiled Hierarchical Macros**")
    macro_cols = st.columns(4)
    for i, action in enumerate(macros):
        col = macro_cols[i]
        with col:
            macro_idx = 18 + i
            if st.button(action, key=f"macro_{action}"):
                st.session_state.cube.step(macro_idx)
                st.session_state.history.append(f"<span class='macro-move'>{action}</span>")
                st.rerun()
                
    st.markdown("---")
    
    st.subheader("🤖 Hierarchical AI Solver (H-DAVI)")
    
    if model_loaded:
        ai_col1, ai_col2 = st.columns(2)
        
        with ai_col1:
            if st.button("AI Next Move (Single Step)", type="primary"):
                if not st.session_state.cube.is_solved():
                    with st.spinner("AI thinking..."):
                        action = st.session_state.solver.get_next_action(st.session_state.cube.state)
                        act_name = actions[action]
                        st.session_state.cube.step(action)
                        if action >= 18:
                            st.session_state.history.append(f"<span class='macro-move'>AI:{act_name}</span>")
                        else:
                            st.session_state.history.append(f"<span class='primitive-move'>AI:{act_name}</span>")
                    st.rerun()
                    
        with ai_col2:
            if st.button("AI A* Solve (Full Path Search)", type="primary"):
                if not st.session_state.cube.is_solved():
                    with st.spinner("Searching for hierarchical solution path..."):
                        path, nodes = st.session_state.solver.solve(st.session_state.cube.state)
                        if path is not None:
                            solve_steps = []
                            for action in path:
                                act_name = actions[action]
                                st.session_state.cube.step(action)
                                if action >= 18:
                                    tag = f"<span class='macro-move'>AI:{act_name}</span>"
                                else:
                                    tag = f"<span class='primitive-move'>AI:{act_name}</span>"
                                st.session_state.history.append(tag)
                                solve_steps.append((act_name, action >= 18))
                            st.session_state.last_solve_details = {
                                "steps": solve_steps,
                                "nodes": nodes,
                                "length": len(path)
                            }
                        else:
                            st.error("Failed to find a path within the expanded nodes limit.")
                    st.rerun()
                    
        if st.session_state.last_solve_details is not None:
            details = st.session_state.last_solve_details
            st.success(f"🎉 Solved successfully! Path length: **{details['length']}** steps. Nodes expanded during search: **{details['nodes']}**.")
            
            st.markdown("### Solution Path Breakdown:")
            cols = st.columns(min(12, len(details['steps'])))
            for step_idx, (act_name, is_macro) in enumerate(details['steps']):
                c_idx = step_idx % 12
                with cols[c_idx]:
                    if is_macro:
                        st.markdown(f"<div style='border:1px solid #10b981; padding:5px; border-radius:5px; text-align:center; background:rgba(16,185,129,0.1)'><b style='color:#10b981'>{act_name}</b><br><small style='font-size:0.7rem'>Macro</small></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='border:1px solid #fbbf24; padding:5px; border-radius:5px; text-align:center; background:rgba(251,191,36,0.1)'><b style='color:#fbbf24'>{act_name}</b><br><small style='font-size:0.7rem'>Primitive</small></div>", unsafe_allow_html=True)
                        
    else:
        if st.session_state.get('load_error') is not None:
            st.error(f"Error loading model weights: {st.session_state.load_error}")
        else:
            st.info("💡 Train the tactical agent using `python training/train_loop.py` to unlock AI solving capabilities with the Value Network.")
            
    # Display History
    if st.session_state.history:
        st.markdown("### Move History")
        history_html = " &rarr; ".join(st.session_state.history)
        st.markdown(f"<div style='background:rgba(255,255,255,0.05); padding:1rem; border-radius:10px; font-family:monospace;'>{history_html}</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("🧠 Neural Brain Introspection")
    
    if model_loaded:
        st.markdown("""
        Peer inside the learned value network ("brain") to analyze its attention focus and distance predictions.
        """)
        
        col_brain1, col_brain2 = st.columns([1.5, 1.5])
        
        with col_brain1:
            st.markdown("### Sticker-Level Attention (Saliency Map)")
            st.markdown("""
            This map highlights which stickers the Value Network focuses on to estimate remaining distance.
            Brighter colors (white/yellow/orange) represent higher gradient sensitivities.
            """)
            
            # Compute live saliency map for current state
            with st.spinner("Computing attention gradients..."):
                saliency = get_saliency(st.session_state.cube.state, st.session_state.solver.model, device="cpu")
                
            fig_sal, ax_sal = plt.subplots(figsize=(6, 4))
            fig_sal.patch.set_alpha(0.0)
            ax_sal.patch.set_alpha(0.0)
            draw_saliency_heatmap(saliency, ax=ax_sal)
            st.pyplot(fig_sal)
            
        with col_brain2:
            st.markdown("### Heuristic Calibration Landscape")
            st.markdown("""
            Evaluates how well the Value Network's distance prediction matches the true scramble depth.
            An **admissible** heuristic ($h(s) \le \text{true\_cost}$) guarantees A* optimality.
            """)
            
            plot_path = "trained_models/brain_calibration.png"
            
            # Run live calibration profile check button
            if st.button("Run Live Calibration Scan"):
                with st.spinner("Profiling heuristic on random scrambles..."):
                    depths, mean_ests, std_ests, _ = profile_heuristic(st.session_state.solver.model, device="cpu", num_samples=30)
                    generate_calibration_plot(depths, mean_ests, std_ests, output_path=plot_path)
                st.success("Calibration scan complete and plot updated!")
                
            if os.path.exists(plot_path):
                st.image(plot_path, caption="Value Network Calibration Curve")
            else:
                st.info("No calibration plot found. Click 'Run Live Calibration Scan' to generate one.")
    else:
        st.info("💡 Introspection features are disabled because no trained model weights were found at `trained_models/value_agent.pt`.")
