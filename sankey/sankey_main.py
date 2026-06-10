"""
CORE-Bench Curation & Evaluation Pipeline — Sankey diagram
Requires:  pip install plotly kaleido
Output:    sankey_pipeline.png  (300 dpi, for paper)
           sankey_pipeline.svg  (vector)
           sankey_pipeline.html (interactive)
"""

import plotly.graph_objects as go

FONT     = "Times New Roman"
TEXT_COL = "#111111"

# Seaborn colorblind palette  (sns.color_palette("colorblind"))
BLUE_D   = "#0173B2"   # blue        — T1 source / curation
BLUE_M   = "#56B4E9"   # sky blue    — T1 branches / Codex CLI
RED      = "#D55E00"   # vermillion  — all removed nodes
GREEN_D  = "#029E73"   # green       — curated tasks / regrade
ORG_D    = "#DE8F05"   # orange      — T2 source / curation
ORG_M    = "#CA9161"   # tan         — T2 branches
PURPLE   = "#CC78BC"   # purple      — combined node
PURPLE_L = "#56B4E9"   # sky blue    — Codex CLI (reused; distinct from purple section)
AMBER    = "#949494"   # gray        — grading update

# ── Nodes ─────────────────────────────────────────────────────────────────────
node_labels = [
    # Track 1  (nodes 0–6)
    "<b>CORE-Bench Hard</b><br>45 tasks",                                           
    "<b>Checked for process & computation incorrectness</b><br>(Claude Code · Opus 4.5)<br>all correct tasks",               
    "<b>Checked incorrect logs for benchmark errors</b><br>(Claude Code · Opus 4.5)<br>all incorrect tasks",             
    "<b>Checked for answers in pre-existing artifacts</b><br>(OpenCode + CORE-Agent<br>Opus 4.5 and 4.6)<br>all 45 tasks",
    "<b>Curation</b><br>Updated tasks based on validity errors <br>found in the previous step<br>Remove 14 · Edit 12<br>from all 45 tasks",                                         
    "Removed 14 tasks",                                                            
    "<b>31 tasks</b>",                                                         

    # Track 2  (nodes 7–14)
    "<b>27 new tasks</b>",                                                            
    "<b>Checked for process & computation incorrectness</b><br>(OpenCode · GPT-5.2 + CORE-Agent · <br>Opus 4.5 and 4.6) | all correct tasks", 
    "<b>Checked incorrect logs for benchmark errors</b><br>(OpenCode · GPT-5.2 + CORE-Agent · <br>Opus 4.5 and 4.6) | all incorrect tasks",
    "<b>Checked for answers in pre-existing artifacts</b><br>(OpenCode · GPT-5.2 + CORE-Agent · <br>Opus 4.5 and 4.6) | all 27 tasks",  
    "<b>Manually inspected<br>task questions</b><br>all 27 tasks",                                                             
    "<b>Curation</b><br>Updated tasks based on validity errors <br>found in the previous step<br>Remove 16 · Edit 5<br>from all 27 tasks",                                          
    "Removed <br>16 tasks",                                                              
    "<b>11 tasks</b>",                                                        

    # Combined pipeline  (nodes 15–20)
    "<b>42 tasks</b><br>all 20 runs<br>evaluated",                               
    "<b>Inspected incorrect logs<br>for further errors</b><br>all 20 agent runs<br>42 tasks",           
    "Removed 3 tasks",                                                                 
    "<b>Updated grading</b><br>5 tasks affected",                                      
    "<b>39 tasks</b>",                                     
]

node_colors = [
    BLUE_D,    # 0  CORE-Bench Hard
    BLUE_M,    # 1  T1 branch
    BLUE_M,    # 2
    BLUE_M,    # 3
    BLUE_D,    # 4  T1 curation
    RED,       # 5  removed
    GREEN_D,   # 6  curated T1
    ORG_D,     # 7  27 new tasks
    ORG_M,     # 8  T2 branch
    ORG_M,     # 9
    ORG_M,     # 10
    ORG_M,     # 11  new T2 branch
    ORG_D,     # 12 T2 curation
    RED,       # 13 removed
    GREEN_D,   # 14 curated T2
    PURPLE,    # 15 combined
    PURPLE_L,  # 16 Codex CLI
    RED,       # 17 removed
    AMBER,     # 18 grading update
    GREEN_D,   # 19 regrade
]

node_x = [
    0.01, 0.22, 0.22, 0.22, 0.44, 0.62, 0.62,         # Track 1: nodes 0–6
    0.01, 0.22, 0.22, 0.22, 0.22, 0.44, 0.62, 0.62,   # Track 2: nodes 7–14
    0.70, 0.79, 0.88, 0.88, 0.99,                      # Combined: nodes 15–19
]
node_y = [
    0.17,  # 0  CORE-Bench Hard
    0.04,  # 1  T1 Process & Comp
    0.20,  # 2  T1 Manual Inspect
    0.36,  # 3  T1 Artifacts
    0.20,  # 4  T1 Curation
    0.04,  # 5  14 Removed
    0.36,  # 6  31 Curated
    0.68,  # 7  27 New Tasks  (centred on branches: (0.44+0.93)/2)
    0.44,  # 8  T2 Process & Comp
    0.60,  # 9  T2 Manual Inspect Logs
    0.76,  # 10 T2 Artifacts
    0.93,  # 11 T2 Manual Inspect Tasks
    0.68,  # 12 T2 Curation
    0.44,  # 13 16 Removed
    0.90,  # 14 11 Curated
    0.50,  # 15 Combined
    0.22,  # 16 Codex CLI
    0.08,  # 17 3 Removed
    0.68,  # 18 Grading Update
    0.50,  # 19 Regrade
]

# ── Links ─────────────────────────────────────────────────────────────────────
# Flow accounting (task units):
#   Track 1: 45 → branches (3×15) → curation → 14 removed + 31 continue
#   Track 2: 27 → branches (4×~7) → curation → 16 removed + 11 continue
#   Combined (42 tasks, 20 runs):
#     → all 42 → Codex CLI (flag incorrect logs): 3 removed, 4 → grading update, 35 → regrade
#   Regrade total: 35 + 4 = 39  (= 42 − 3 removed)

src = [
    # Track 1
    0, 0, 0,         # CORE-Bench → 3 branches
    1, 2, 3,         # T1 branches → Curation
    4, 4,            # T1 Curation → removed / curated
    # Track 2
    7, 7, 7, 7,      # 27 New Tasks → 4 branches
    8, 9, 10, 11,    # T2 branches → Curation
    12, 12,          # T2 Curation → removed / curated
    # Combined
    6, 14,           # curated tracks → Combined
    15,              # Combined → Codex CLI (all 42)
    16, 16, 16,      # Codex CLI → 3 removed / grading update / regrade
    18,              # Grading update → Regrade
]
tgt = [
    1, 2, 3,
    4, 4, 4,
    5, 6,
    8, 9, 10, 11,
    12, 12, 12, 12,
    13, 14,
    15, 15,
    16,
    17, 18, 19,
    19,
]
val = [
    15, 15, 15,      # 45 ÷ 3 branches
    15, 15, 15,      # branches → curation (sum = 45)
    14, 31,          # 14 removed; 31 continue
    7, 7, 7, 6,      # 27 ÷ 4 branches (7+7+7+6=27)
    7, 7, 7, 6,      # branches → curation (sum = 27)
    16, 11,          # 16 removed; 11 continue
    31, 11,          # merge → 42 combined
    42,              # all 42 → Codex CLI
    3, 4, 35,        # Codex CLI: 3 removed, 4 → grading update, 35 → regrade
    4,               # grading update → regrade
]

def rgba(hex_col: str, a: float = 0.35) -> str:
    h = hex_col.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return f"rgba({r},{g},{b},{a})"

link_colors = [rgba(node_colors[s]) for s in src]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = go.Figure(go.Sankey(
    arrangement="snap",
    textfont=dict(family=FONT, size=18, color=TEXT_COL),
    node=dict(
        pad=20,
        thickness=26,
        line=dict(color="white", width=0.8),
        label=node_labels,
        color=node_colors,
        x=node_x,
        y=node_y,
        hovertemplate="%{label}<extra></extra>",
    ),
    link=dict(
        source=src,
        target=tgt,
        value=val,
        color=link_colors,
        hovertemplate="%{value} tasks<extra></extra>",
    ),
))

fig.update_layout(
    title=dict(
        text="<b>CORE-Bench v1.1 Construction Pipeline</b>",
        font=dict(family=FONT, size=22, color=TEXT_COL),
        x=0.5,
    ),
    font=dict(family=FONT, size=18, color=TEXT_COL),
    width=2000,
    height=1000,
    paper_bgcolor="white",
    margin=dict(l=20, r=20, t=70, b=20),
)

# ── Export ────────────────────────────────────────────────────────────────────
fig.write_html("sankey_pipeline.html")
print("Saved → sankey_pipeline.html")

try:
    #fig.write_image("sankey_pipeline.png", scale=3)
    # fig.write_image("sankey_pipeline.svg")
    print("Saved → sankey_pipeline.png  (scale=3, ~300 dpi)")
    # print("Saved → sankey_pipeline.svg  (vector)")
except Exception as e:
    print(f"Static export failed — install kaleido:  pip install kaleido\n  {e}")

fig.show()
