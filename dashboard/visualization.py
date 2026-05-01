import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

COLOR_MAP = {
    0: 'white',   # U
    1: 'yellow',  # D
    2: 'green',   # F
    3: 'blue',    # B
    4: 'orange',  # L
    5: 'red'      # R
}

def draw_2x2_cube(state, ax=None):
    """
    Draws a 2D unrolled representation of the 2x2 cube state using matplotlib.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        ax.clear()
        
    ax.axis('off')
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)

    # Coordinates for faces in the unrolled format
    # Layout:
    #   U
    # L F R B
    #   D
    # Grid sizes: each sticker is 1x1. Each face is 2x2.
    face_offsets = {
        'U': (2, 4), # Top center
        'L': (0, 2), # Left
        'F': (2, 2), # Center
        'R': (4, 2), # Right
        'B': (6, 2), # Far right
        'D': (2, 0)  # Bottom
    }

    # Order of faces in state array
    face_order = ['U', 'D', 'F', 'B', 'L', 'R']
    
    for i, face_name in enumerate(face_order):
        offset_x, offset_y = face_offsets[face_name]
        face_stickers = state[i*4:(i+1)*4]
        
        # Draw the 4 stickers
        # Sticker indices:
        # 0 1
        # 2 3
        # In plot (0,0 is bottom left), so:
        # 0 is top-left: x, y+1
        # 1 is top-right: x+1, y+1
        # 2 is bottom-left: x, y
        # 3 is bottom-right: x+1, y
        
        coords = [
            (offset_x, offset_y + 1),
            (offset_x + 1, offset_y + 1),
            (offset_x, offset_y),
            (offset_x + 1, offset_y)
        ]
        
        for j, (sx, sy) in enumerate(coords):
            color_idx = face_stickers[j]
            color = COLOR_MAP.get(color_idx, 'gray')
            rect = patches.Rectangle((sx, sy), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            
    return ax
