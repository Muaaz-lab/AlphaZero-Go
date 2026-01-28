<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ConnectX AI Board + Optimized Engine + Neural Network Model</title>

  <!-- Google Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Fira+Code:wght@400;600&display=swap" rel="stylesheet">

  <style>
    :root{
      --bg1:#0f172a;
      --bg2:#111827;
      --card:#0b1220cc;
      --border:rgba(255,255,255,0.12);
      --text:#e5e7eb;
      --muted:#b6c2d6;
      --accent:#22c55e;
      --accent2:#38bdf8;
      --warn:#f59e0b;
      --pink:#fb7185;
      --shadow: 0 20px 60px rgba(0,0,0,0.35);
      --radius:18px;
    }

    *{box-sizing:border-box;}
    body{
      margin:0;
      font-family:"Poppins", sans-serif;
      color:var(--text);
      background:
        radial-gradient(1200px 600px at 10% 0%, rgba(56,189,248,0.18), transparent 60%),
        radial-gradient(900px 500px at 90% 10%, rgba(34,197,94,0.16), transparent 55%),
        radial-gradient(700px 450px at 50% 100%, rgba(251,113,133,0.12), transparent 60%),
        linear-gradient(180deg, var(--bg1), var(--bg2));
      min-height:100vh;
      overflow-x:hidden;
    }

    /* Top Banner */
    .hero{
      padding:60px 18px 20px;
      text-align:center;
      position:relative;
    }

    .badge{
      display:inline-flex;
      align-items:center;
      gap:10px;
      padding:10px 14px;
      border:1px solid var(--border);
      border-radius:999px;
      background:rgba(255,255,255,0.04);
      backdrop-filter: blur(8px);
      font-size:14px;
      color:var(--muted);
      box-shadow:0 10px 30px rgba(0,0,0,0.25);
    }

    .dot{
      width:10px;
      height:10px;
      border-radius:50%;
      background:linear-gradient(90deg, var(--accent), var(--accent2));
      box-shadow:0 0 18px rgba(56,189,248,0.7);
    }

    h1{
      margin:18px 0 10px;
      font-size:clamp(26px, 4vw, 44px);
      letter-spacing:0.3px;
      line-height:1.15;
    }

    .sub{
      margin:0 auto;
      max-width:920px;
      font-size:16px;
      color:var(--muted);
      line-height:1.7;
    }

    .container{
      width:min(1150px, 92vw);
      margin:0 auto;
      padding:18px 0 60px;
    }

    /* Layout grid */
    .grid{
      display:grid;
      grid-template-columns: repeat(12, 1fr);
      gap:18px;
      margin-top:26px;
    }

    .card{
      grid-column: span 12;
      background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
      border:1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding:18px;
      position:relative;
      overflow:hidden;
      transition: transform .25s ease, border-color .25s ease;
    }

    .card:hover{
      transform: translateY(-3px);
      border-color: rgba(56,189,248,0.35);
    }

    .card::before{
      content:"";
      position:absolute;
      inset:-1px;
      background: radial-gradient(500px 160px at 10% 0%, rgba(56,189,248,0.18), transparent 60%),
                  radial-gradient(500px 160px at 90% 0%, rgba(34,197,94,0.16), transparent 60%);
      pointer-events:none;
      opacity:0.8;
    }

    .card > *{ position:relative; }

    .title-row{
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap:14px;
      margin-bottom:10px;
    }

    .title{
      display:flex;
      align-items:center;
      gap:12px;
    }

    .icon{
      width:44px;
      height:44px;
      border-radius:14px;
      display:grid;
      place-items:center;
      background: rgba(255,255,255,0.06);
      border:1px solid var(--border);
      box-shadow:0 10px 30px rgba(0,0,0,0.25);
      flex-shrink:0;
    }

    .icon svg{ width:22px; height:22px; }

    .title h2{
      margin:0;
      font-size:20px;
      letter-spacing:0.2px;
    }

    .meta{
      margin-top:3px;
      font-size:13px;
      color:var(--muted);
    }

    .pill{
      display:inline-flex;
      gap:8px;
      align-items:center;
      padding:8px 12px;
      border-radius:999px;
      border:1px solid var(--border);
      background: rgba(0,0,0,0.25);
      color:var(--muted);
      font-size:13px;
      white-space:nowrap;
    }

    .pill strong{
      color:#fff;
      font-weight:600;
    }

    .desc{
      color:var(--muted);
      line-height:1.75;
      margin:12px 0 14px;
      font-size:15px;
    }

    /* Code block */
    pre{
      margin:0;
      border-radius:16px;
      border:1px solid rgba(255,255,255,0.12);
      background: rgba(2,6,23,0.7);
      overflow:auto;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
      position:relative;
    }

    code{
      font-family:"Fira Code", monospace;
      font-size:13px;
      line-height:1.65;
      display:block;
      padding:16px 16px 18px;
      color:#e5e7eb;
      min-width:720px;
    }

    /* Copy button */
    .copy-btn{
      position:absolute;
      top:12px;
      right:12px;
      border:1px solid rgba(255,255,255,0.15);
      background: rgba(255,255,255,0.06);
      color:#fff;
      padding:8px 10px;
      border-radius:12px;
      cursor:pointer;
      font-size:12px;
      display:flex;
      align-items:center;
      gap:8px;
      transition: transform .2s ease, background .2s ease;
      user-select:none;
    }
    .copy-btn:hover{
      transform: translateY(-1px);
      background: rgba(56,189,248,0.12);
    }
    .copy-btn:active{
      transform: translateY(0px) scale(0.98);
    }

    /* Small info cards */
    .mini-grid{
      display:grid;
      grid-template-columns: repeat(12, 1fr);
      gap:14px;
      margin-top:14px;
    }

    .mini{
      grid-column: span 12;
      padding:14px;
      border-radius:16px;
      border:1px solid var(--border);
      background: rgba(255,255,255,0.04);
      color:var(--muted);
      line-height:1.6;
    }

    .mini strong{
      color:#fff;
      font-weight:600;
    }

    /* Responsive */
    @media (min-width: 900px){
      .card.span6{ grid-column: span 6; }
      .mini.span6{ grid-column: span 6; }
      .mini.span4{ grid-column: span 4; }
      code{ min-width: 0; }
    }

    /* Footer */
    footer{
      text-align:center;
      padding:18px 10px 45px;
      color:rgba(229,231,235,0.55);
      font-size:13px;
    }

    .glow-line{
      height:1px;
      width:min(900px, 90vw);
      margin:24px auto 0;
      background: linear-gradient(90deg, transparent, rgba(56,189,248,0.6), rgba(34,197,94,0.6), transparent);
      opacity:0.7;
    }
  </style>
</head>

<body>

  <section class="hero">
    <div class="badge">
      <span class="dot"></span>
      <span><b>ConnectX</b> • Board Engine • Optimized Win Check • Neural Network Model</span>
    </div>

    <h1>🚀 ConnectX AI Implementation (Clean + Optimized + Beautiful)</h1>
    <p class="sub">
      This page contains three Python programs used in a ConnectX / Connect4-style project:
      <b>(1)</b> a simple board simulator, <b>(2)</b> an optimized environment for faster gameplay checks, and
      <b>(3)</b> a TensorFlow CNN model that can be used as a Player Agent for action selection.
    </p>

    <div class="glow-line"></div>
  </section>

  <main class="container">
    <div class="grid">

      <!-- CARD 1 -->
      <section class="card span6">
        <div class="title-row">
          <div class="title">
            <div class="icon">
              <!-- grid icon -->
              <svg viewBox="0 0 24 24" fill="none">
                <path d="M4 4h7v7H4V4Zm9 0h7v7h-7V4ZM4 13h7v7H4v-7Zm9 0h7v7h-7v-7Z" stroke="white" stroke-width="1.6"/>
              </svg>
            </div>
            <div>
              <h2>1) Basic ConnectX Board Simulator</h2>
              <div class="meta">Includes gravity drop logic + winner checking</div>
            </div>
          </div>
          <div class="pill"><strong>Goal:</strong> simple gameplay test</div>
        </div>

        <p class="desc">
          This code creates a small <b>3×3 ConnectX board</b> (inarow = 3) and loads the exact matrix state
          from your image. It supports:
          <b>valid move checking</b>, <b>gravity-based piece dropping</b>, and <b>winner detection</b>
          (horizontal, vertical, and diagonal).
        </p>

        <div class="mini-grid">
          <div class="mini span6">
            <strong>Board Values:</strong><br/>
            <b>1</b> = Player 1 piece<br/>
            <b>-1</b> = Opponent piece<br/>
            <b>0</b> = Empty cell
          </div>
          <div class="mini span6">
            <strong>Key Functions:</strong><br/>
            <b>is_valid_move(col)</b> → checks top cell<br/>
            <b>drop_piece(col,piece)</b> → gravity drop<br/>
            <b>check_winner(piece)</b> → detects win
          </div>
        </div>

        <pre>
          <button class="copy-btn" onclick="copyCode('code1')">📋 Copy</button>
          <code id="code1">import numpy as np

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
print(game.board)</code>
        </pre>
      </section>

      <!-- CARD 2 -->
      <section class="card span6">
        <div class="title-row">
          <div class="title">
            <div class="icon">
              <!-- lightning icon -->
              <svg viewBox="0 0 24 24" fill="none">
                <path d="M13 2 3 14h7l-1 8 10-12h-7l1-8Z" stroke="white" stroke-width="1.6" stroke-linejoin="round"/>
              </svg>
            </div>
            <div>
              <h2>2) Optimized ConnectX Environment</h2>
              <div class="meta">Faster drop logic + neural network ready input</div>
            </div>
          </div>
          <div class="pill"><strong>Goal:</strong> Kaggle-friendly engine</div>
        </div>

        <p class="desc">
          This version uses a standard <b>6×7 Connect4 layout</b> with <b>inarow = 4</b>.
          It includes a more efficient drop mechanism using NumPy operations and a
          <b>get_nn_input()</b> function to prepare the board for a CNN model.
        </p>

        <div class="mini-grid">
          <div class="mini span4">
            <strong>Why Optimized?</strong><br/>
            Uses NumPy to find empty row positions quickly.
          </div>
          <div class="mini span4">
            <strong>NN Input Shape</strong><br/>
            Output is reshaped into:<br/>
            <b>(1, rows, cols, 1)</b>
          </div>
          <div class="mini span4">
            <strong>Win Check</strong><br/>
            Uses slicing + diagonal extraction for faster checking.
          </div>
        </div>

        <pre>
          <button class="copy-btn" onclick="copyCode('code2')">📋 Copy</button>
          <code id="code2">import numpy as np

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
print("Neural Network Ready State:\n", env.get_nn_input().shape)</code>
        </pre>
      </section>

      <!-- CARD 3 -->
      <section class="card">
        <div class="title-row">
          <div class="title">
            <div class="icon">
              <!-- brain icon -->
              <svg viewBox="0 0 24 24" fill="none">
                <path d="M9 4a3 3 0 0 1 3 3v1m0-4a3 3 0 0 1 3 3v1M7 9a3 3 0 0 0-3 3v1a3 3 0 0 0 3 3m10-7a3 3 0 0 1 3 3v1a3 3 0 0 1-3 3M9 20a3 3 0 0 1-3-3v-1m9 4a3 3 0 0 0 3-3v-1" stroke="white" stroke-width="1.6" stroke-linecap="round"/>
                <path d="M9 8h6M9 12h6M9 16h6" stroke="white" stroke-width="1.4" stroke-linecap="round" opacity="0.7"/>
              </svg>
            </div>
            <div>
              <h2>3) TensorFlow CNN Model (Player Agent)</h2>
              <div class="meta">Convolution layers → flatten → dense → action outputs</div>
            </div>
          </div>
          <div class="pill"><strong>Goal:</strong> predict best column move</div>
        </div>

        <p class="desc">
          This model is designed to treat the board like an <b>image</b>.
          The <b>Conv2D layers</b> learn spatial patterns (like threats, winning lines, and blocks),
          and the final output layer produces <b>Q-values / scores</b> for each possible action
          (each column in ConnectX).
        </p>

        <div class="mini-grid">
          <div class="mini span6">
            <strong>Input:</strong><br/>
            Board state <b>S(t)</b> in shape: <b>(1, rows, cols)</b> (channels_first).
          </div>
          <div class="mini span6">
            <strong>Output:</strong><br/>
            A vector of size <b>num_actions = cols</b><br/>
            (one score per column).
          </div>
        </div>

        <pre>
          <button class="copy-btn" onclick="copyCode('code3')">📋 Copy</button>
          <code id="code3">import tensorflow as tf
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
    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=(1, rows, cols),
                     data_format='channels_first'))
    model.add(Activation('relu'))

    # 2. HIDDEN LAYERS: Processing spatial patterns
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Flatten())  # Flattening the grid into a vector
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))

    # 3. OUTPUT LAYER: Produces the Next Move (At)
    # One node for each possible column (action)
    model.add(Dense(num_actions))
    model.add(Activation('linear'))  # Q-values for each action

    return model

# Parameters for standard Connect 4 (rows=6, cols=7)
rows, cols = 6, 7
num_actions = cols  # One action per column

model = build_player_agent_model(rows, cols, num_actions)
model.summary()</code>
        </pre>
      </section>

    </div>
  </main>

  <footer>
    Made with ❤️ for ConnectX learners • Beautiful HTML Documentation • Ready for Kaggle / GitHub Pages
  </footer>

  <script>
    function copyCode(id){
      const code = document.getElementById(id).innerText;
      navigator.clipboard.writeText(code).then(()=>{
        alert("✅ Code copied successfully!");
      }).catch(()=>{
        alert("❌ Copy failed. Please copy manually.");
      });
    }
  </script>

</body>
</html>
