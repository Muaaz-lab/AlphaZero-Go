import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# (a) Learning Curve (Elo vs Days)
# ----------------------------

np.random.seed(7)

days = np.linspace(0, 40, 400)

# Create a curve similar to AlphaGo Zero training (fast rise then plateau)
elo_curve = -2000 + 7000 * (1 - np.exp(-days / 2.2))
elo_curve += 150 * np.log1p(days)  # small improvement over time
elo_curve += np.random.normal(0, 30, size=len(days))  # noise

# Smooth the curve a bit
window = 9
elo_curve_smooth = np.convolve(elo_curve, np.ones(window)/window, mode="same")

# Reference Elo lines (approx)
alpha_go_master = 4900
alpha_go_lee = 3500

# ----------------------------
# (b) Bar Chart (Elo comparison)
# ----------------------------

labels = [
    "Raw Network",
    "AlphaGo Zero",
    "AlphaGo Master",
    "AlphaGo Lee",
    "AlphaGo Fan",
    "Crazy Stone",
    "Pachi",
    "GnuGo"
]

elo_values = [3000, 5200, 4800, 3700, 3100, 1900, 1300, 400]

# Colors similar to paper style (gray, blues, red shades)
colors = [
    "#D3D3D3",  # Raw Network (gray)
    "#3B8ED0",  # AlphaGo Zero (blue)
    "#0A6AA1",  # AlphaGo Master (dark blue)
    "#1B7EB6",  # AlphaGo Lee
    "#0B5F8A",  # AlphaGo Fan
    "#FF4D4D",  # Crazy Stone (red)
    "#FF6B6B",  # Pachi (light red)
    "#FF9A9A"   # GnuGo (lighter red)
]

# ----------------------------
# Plotting both graphs together
# ----------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

# --- Graph (a)
ax1 = axes[0]
ax1.plot(days, elo_curve_smooth, color="#1f77b4", linewidth=2, label="AlphaGo Zero 40 blocks")

ax1.axhline(alpha_go_master, color="#1f77b4", linestyle="--", linewidth=1.5, label="AlphaGo Master")
ax1.axhline(alpha_go_lee, color="green", linestyle="--", linewidth=1.5, label="AlphaGo Lee")

ax1.set_title("a.", loc="left", fontweight="bold")
ax1.set_xlabel("Days")
ax1.set_ylabel("Elo Rating")
ax1.set_xlim(0, 40)
ax1.set_ylim(-2200, 5200)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="lower right", fontsize=8)

# --- Graph (b)
ax2 = axes[1]
bars = ax2.bar(labels, elo_values, color=colors, edgecolor="black", linewidth=0.3)

ax2.set_title("b.", loc="left", fontweight="bold")
ax2.set_ylabel("Elo Rating")
ax2.set_ylim(0, 5400)
ax2.grid(axis="y", alpha=0.3)

# Rotate x labels like paper
plt.setp(ax2.get_xticklabels(), rotation=-45, ha="left")

# ----------------------------
# Extra paper-like header text
# ----------------------------
fig.suptitle("Elo scale: a 200 point gap corresponds to a 75% probability of winning.",
             fontsize=10, y=1.02)

plt.tight_layout()
plt.show()

import numpy as np

class ConnectXBoard:
    def __init__(self, rows=3, cols=3, inarow=3):
        self.rows = rows
        self.cols = cols
        self.inarow = inarow
        # Initialize with the matrix from your image
        self.board = np.array([
            [-1,  1,  1],
            [-1,  0,  0],
            [ 0,  0,  0]
        ])

    def is_valid_move(self, col):
        """Checks if a piece can be dropped into the specified column."""
        return self.board[0][col] == 0

    def drop_piece(self, col, piece):
        """Drops a piece (-1 or 1) into a column with 'gravity' logic."""
        if not self.is_valid_move(col):
            return False
        for r in range(self.rows-1, -1, -1):
            if self.board[r][col] == 0:
                self.board[r][col] = piece
                return True
        return False

    def check_winner(self, piece):
        """Checks if the specified piece has 'inarow' tokens in a line."""
        # Check horizontal
        for r in range(self.rows):
            for c in range(self.cols - self.inarow + 1):
                if all(self.board[r, c:c+self.inarow] == piece):
                    return True
        
        # Check vertical
        for r in range(self.rows - self.inarow + 1):
            for c in range(self.cols):
                if all(self.board[r:r+self.inarow, c] == piece):
                    return True
        
        # Check diagonals
        for r in range(self.rows - self.inarow + 1):
            for c in range(self.cols - self.inarow + 1):
                # Positive diagonal
                if all(self.board[r+i, c+i] == piece for i in range(self.inarow)):
                    return True
                # Negative diagonal
                if all(self.board[r+i, c+self.inarow-1-i] == piece for i in range(self.inarow)):
                    return True
        return False

# Simulation execution
game = ConnectXBoard()
print("Initial Board State (from image):")
print(game.board)

# Example: Player 1 (1) tries to move in the middle column (index 1)
game.drop_piece(1, 1)
print("\nBoard after Player 1 drops a piece in column 1:")
print(game.board)

import numpy as np

class ConnectXOptimized:
    def __init__(self, rows=6, cols=7, inarow=4):
        self.rows = rows
        self.cols = cols
        self.inarow = inarow
        # Initialize board using the 1, -1, 0 format from the diagram
        self.board = np.zeros((rows, cols), dtype=int)

    def get_nn_input(self):
        """
        Prepares the board for the Neural Network input layer.
        Reshapes to (1, rows, cols, 1) for Convolutional layers or flattens.
        """
        return self.board.reshape(1, self.rows, self.cols, 1).astype(float)

    def drop_piece(self, col, piece):
        """Optimized gravity logic."""
        # Find the first empty row in the column
        res = np.where(self.board[:, col] == 0)[0]
        if len(res) > 0:
            self.board[res[-1], col] = piece
            return True
        return False

    def check_winner(self, piece):
        """
        Vectorized win-check. 
        Checks for 'inarow' tokens using slicing instead of nested loops.
        """
        # Horizontal check
        for c in range(self.cols - self.inarow + 1):
            for r in range(self.rows):
                if np.all(self.board[r, c:c+self.inarow] == piece):
                    return True
        
        # Vertical check
        for r in range(self.rows - self.inarow + 1):
            for c in range(self.cols):
                if np.all(self.board[r:r+self.inarow, c] == piece):
                    return True

        # Diagonal checks
        for r in range(self.rows - self.inarow + 1):
            for c in range(self.cols - self.inarow + 1):
                block = self.board[r:r+self.inarow, c:c+self.inarow]
                if np.all(np.diagonal(block) == piece) or \
                   np.all(np.diagonal(np.fliplr(block)) == piece):
                    return True
        return False

# Example setup for Kaggle environment
env = ConnectXOptimized()
# Simulating the matrix state from the first image
env.board[0:2, 0:3] = [[-1, 1, 1], [-1, 0, 0]] 
print("Neural Network Ready State:\n", env.get_nn_input().shape)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation
from tensorflow.keras.optimizers import Adam

def build_player_agent_model(rows, cols, num_actions):
    """
    Constructs the Neural Network for the Player Agent.
    """
    model = Sequential()
    
    # 1. INPUT LAYER: Receives the Board State (St)
    # Reshaped to (rows, cols, 1) to treat the board like an image
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(1, rows, cols), data_format='channels_first'))
    model.add(Activation('relu'))
    
    # 2. HIDDEN LAYERS: Processing spatial patterns
    # The diagram shows multiple layers of nodes processing the state
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    
    model.add(Flatten()) # Flattening the grid into a vector
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    
    # 3. OUTPUT LAYER: Produces the Next Move (At)
    # One node for each possible column (action)
    model.add(Dense(num_actions))
    model.add(Activation('linear')) # Q-values for each action
    
    return model

# Parameters for standard Connect 4 (rows=6, cols=7)
rows, cols = 6, 7
num_actions = cols # One action per column

model = build_player_agent_model(rows, cols, num_actions)
model.summary()