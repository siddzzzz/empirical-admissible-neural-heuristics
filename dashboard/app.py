import streamlit as st
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path so we can import env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.cube import Cube2x2
from env.rewards import calculate_entropy
from dashboard.visualization import draw_2x2_cube
import numpy as np
from stable_baselines3 import PPO

st.set_page_config(page_title="Rubik's Cube HRL", layout="wide")

st.title("Hierarchical RL: Rubik's Cube Dashboard")

if 'cube' not in st.session_state:
    st.session_state.cube = Cube2x2()
    st.session_state.history = []

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Cube State")
    fig, ax = plt.subplots(figsize=(6, 4))
    draw_2x2_cube(st.session_state.cube.state, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Controls")
    
    if st.button("Reset / Solved"):
        st.session_state.cube.reset()
        st.session_state.history = []
        st.rerun()
        
    scramble_moves = st.slider("Scramble Moves", 1, 20, 5)
    if st.button("Scramble"):
        st.session_state.cube.scramble(scramble_moves)
        st.rerun()
        
    st.markdown("### Metrics")
    entropy = calculate_entropy(st.session_state.cube.state)
    st.metric(label="Entropy (Disorder)", value=entropy)
    st.metric(label="Is Solved?", value="Yes" if st.session_state.cube.is_solved() else "No")
    
st.markdown("---")
st.subheader("Take Action")

action_cols = st.columns(6)
actions = st.session_state.cube.action_space_names

for i, action in enumerate(actions):
    col = action_cols[i % 6]
    with col:
        if st.button(action):
            st.session_state.cube.step(i)
            st.session_state.history.append(action)
            st.rerun()

st.markdown("---")
st.subheader("AI Agent")

model_path = "trained_models/tactical_agent.zip"
if os.path.exists(model_path):
    if 'model' not in st.session_state:
        st.session_state.model = PPO.load(model_path)
        
    ai_col1, ai_col2 = st.columns(2)
    
    def get_obs(state):
        one_hot = np.zeros((24, 6), dtype=np.float32)
        one_hot[np.arange(24), state] = 1.0
        return one_hot.flatten()

    with ai_col1:
        if st.button("AI Next Move"):
            if not st.session_state.cube.is_solved():
                obs = get_obs(st.session_state.cube.state)
                action, _ = st.session_state.model.predict(obs, deterministic=False)
                act_name = actions[action]
                st.session_state.cube.step(action)
                st.session_state.history.append(f"AI({act_name})")
                st.rerun()
                
    with ai_col2:
        if st.button("AI Solve"):
            for _ in range(20): # Cap to 20 moves
                if st.session_state.cube.is_solved():
                    break
                obs = get_obs(st.session_state.cube.state)
                action, _ = st.session_state.model.predict(obs, deterministic=False)
                act_name = actions[action]
                st.session_state.cube.step(action)
                st.session_state.history.append(f"AI({act_name})")
            st.rerun()
else:
    st.info("Train the tactical agent (run training/train_loop.py) to unlock AI solving capabilities.")

if st.session_state.history:
    st.text(f"Move History: {' '.join(st.session_state.history)}")
