"""
CORE-Bench OOD Construction Pipeline — Sankey diagram
Requires:  pip install plotly kaleido
Output:    sankey_ood_pipeline.png  (300 dpi, for paper)
           sankey_ood_pipeline.svg  (vector)
           sankey_ood_pipeline.html (interactive)
"""

import plotly.graph_objects as go

FONT     = "Times New Roman"
TEXT_COL = "#111111"

# Seaborn colorblind palette  (sns.color_palette("colorblind"))
BLUE_D   = "#0173B2"   # blue        — source / curation
BLUE_M   = "#56B4E9"   # sky blue    — branches
RED      = "#D55E00"   # vermillion  — all removed nodes
GREEN_D  = "#029E73"   # green       — curated / final tasks
ORG_D    = "#DE8F05"   # orange      — added tasks
PURPLE   = "#CC78BC"   # purple      — combined node
PURPLE_L = "#56B4E9"   # sky blue    — Codex CLI inspect logs
AMBER    = "#949494"   # gray        — grading update

# ── Nodes ─────────────────────────────────────────────────────────────────────
node_labels = [
    # Source (node 0)
    "<b>CORE-Bench OOD original</b><br>(30 tasks)",                                       # 0

    # Branches (nodes 1–3)
    "<b>Categorize process & computation <br>correctness </b><br>(CORE-Agent Opus 4.5 and 4.6<br>+ OpenCode GPT-5.2)<br>all 30 tasks",  # 1
    "<b>Check for answers in pre-existing artifact </b><br>(CORE-Agent Opus 4.5 and 4.6<br>+ OpenCode GPT-5.2)<br>all 30 tasks",    # 2
    "<b>Manual inspection of tasks for errors</b><br>all 30 tasks",                                       # 3

    # Curation (nodes 4–6)
    "<b>Curation</b><br>Update tasks based on <br>validity errors found in <br>the previous step<br>Remove 12 · Edit 8<br>from 30 tasks",                                              # 4
    "12 tasks removed",                                                                    # 5  dead end
    "<b>18 tasks</b>",                                                                     # 6

    # Added tasks (node 7)
    "<b>+6 new tasks</b>",                                                                 # 7

    # Combined (node 8)
    "<b>24 tasks</b>",                                                                     # 8

    # Inspect logs (node 9)
    "<b>Inspect incorrect logs<br>for further errors</b><br>12 Codex CLI runs<br>24 tasks",                        # 9

    # Post-inspection (nodes 10–11)
    "5 tasks removed",                                                                     # 10 dead end
    "<b>19 tasks</b><br>All 12 runs evaluated",                                            # 11

    # Grading update (nodes 12–13)
    "<b>Update grading</b><br>(1 task affected)",                                           # 12
    "<b>19 tasks</b><br>Regrade all runs",                                                 # 13
]

node_colors = [
    BLUE_D,    # 0  source
    BLUE_M,    # 1  branch 1
    BLUE_M,    # 2  branch 2
    BLUE_M,    # 3  branch 3
    BLUE_D,    # 4  curation
    RED,       # 5  12 removed
    GREEN_D,   # 6  18 curated
    ORG_D,     # 7  +6 tasks
    PURPLE,    # 8  24 combined
    PURPLE_L,  # 9  inspect logs
    RED,       # 10 5 removed
    GREEN_D,   # 11 19 tasks
    AMBER,     # 12 update grading
    GREEN_D,   # 13 final regrade
]

node_x = [
    0.02,                          # 0  source
    0.20, 0.20, 0.20,              # 1–3 branches
    0.40, 0.52, 0.52,              # 4–6 curation / 12 removed / 18 tasks
    0.52,                          # 7  +6 tasks
    0.64,                          # 8  24 combined
    0.74,                          # 9  inspect logs
    0.84, 0.84,                    # 10–11 5 removed / 19 tasks
    0.87,                          # 12 update grading (shifted left so label clears node 13)
    0.99,                          # 13 final
]
node_y = [
    0.40,                          # 0  source (center)
    0.10, 0.38, 0.66,              # 1–3 branches
    0.38, 0.10, 0.52,              # 4–6 curation / 12 removed / 18 tasks
    0.76,                          # 7  +6 tasks
    0.60,                          # 8  24 combined
    0.55,                          # 9  inspect logs
    0.82, 0.45,                    # 10–11 5 removed / 19 tasks (raised to avoid overlap)
    0.72,                          # 12 update grading (lowered away from node 11)
    0.52,                          # 13 final
]

# ── Links ─────────────────────────────────────────────────────────────────────
# Flow accounting (task units):
#   30 tasks → 3 branches (10 each) → Curation → 12 removed + 18 continue
#   18 tasks + 6 new tasks → 24 combined
#   24 → Inspect logs → 5 removed + 19 continue (all 12 runs evaluated)
#   19 tasks → 1 → update grading + 18 → regrade directly
#   update grading → regrade  (total regrade = 18 + 1 = 19)

src = [
    0, 0, 0,         # source → 3 branches
    1, 2, 3,         # branches → curation
    4, 4,            # curation → 12 removed / 18 tasks
    6, 7,            # 18 tasks + 6 new → 24 combined
    8,               # 24 → inspect logs
    9, 9,            # inspect logs → 5 removed / 19 tasks
    11, 11,          # 19 tasks → update grading / regrade
    12,              # update grading → regrade
]
tgt = [
    1, 2, 3,
    4, 4, 4,
    5, 6,
    8, 8,
    9,
    10, 11,
    12, 13,
    13,
]
val = [
    10, 10, 10,      # 30 ÷ 3 branches
    10, 10, 10,      # branches → curation (sum = 30)
    12, 18,          # 12 removed; 18 continue
    18, 6,           # merge: 18 curated + 6 new = 24
    24,              # all 24 → inspect logs
    5, 19,           # 5 removed; 19 continue
    1, 18,           # 1 run → update grading; 18 tasks → regrade directly
    1,               # update grading → regrade
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
        text="<b>CORE-Bench OOD Construction Pipeline</b>",
        font=dict(family=FONT, size=22, color=TEXT_COL),
        x=0.5,
    ),
    font=dict(family=FONT, size=18, color=TEXT_COL),
    width=2000,
    height=900,
    paper_bgcolor="white",
    margin=dict(l=20, r=20, t=70, b=20),
)

# ── Export ────────────────────────────────────────────────────────────────────
fig.write_html("sankey_ood_pipeline.html")
print("Saved → sankey_ood_pipeline.html")

try:
    pass
    #fig.write_image("sankey_ood_pipeline.png", scale=3)
    # fig.write_image("sankey_ood_pipeline.svg")
    # print("Saved → sankey_ood_pipeline.png  (scale=3, ~300 dpi)")
    # print("Saved → sankey_ood_pipeline.svg  (vector)")
except Exception as e:
    print(f"Static export failed — install kaleido:  pip install kaleido\n  {e}")

fig.show()
