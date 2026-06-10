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
    "<b>CORE-Bench OOD original</b><br>30 tasks",                                       

    # Branches (nodes 1–3)
    "<b>Checked for process & computation <br>incorrectness </b><br>(CORE-Agent · Opus 4.5 and 4.6<br>+ OpenCode · GPT-5.2)<br>all correct tasks", 
    "<b>Checked for answers in pre-existing artifacts</b><br>(CORE-Agent · Opus 4.5 and 4.6<br>+ OpenCode · GPT-5.2)<br>all 30 tasks",
    "<b>Manually inspected task questions</b><br>all 30 tasks",                                      

    # Curation (nodes 4–6)
    "<b>Curation</b><br>Updated tasks based on <br>validity errors found in <br>the previous step<br>Remove 12 · Edit 8<br>from 30 tasks",                                              # 4
    "Removed 12 tasks",                                                                   
    "<b>18 tasks</b>",                                                                    

    # Added tasks (node 7)
    "<b>+6 new tasks</b>",                                                              

    # Combined (node 8)
    "<b>24 tasks</b><br>All 12 runs evaluated",                                                                 

    # Inspect logs (node 9)
    "<b>Inspected incorrect <br>logs for further <br>errors</b><br>12 Codex CLI runs<br>24 tasks",                      

    # Post-inspection (node 10)
    "Removed 5 tasks",                                                                     # 10 dead end

    # Grading update (nodes 11–12)
    "<b>Updated grading</b><br>1 task affected",                                           # 11
    "<b>19 tasks</b>",                                                                     # 12
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
    AMBER,     # 11 update grading
    GREEN_D,   # 12 final regrade
]

node_x = [
    0.02,                          # 0  source
    0.20, 0.20, 0.20,              # 1–3 branches
    0.40, 0.52, 0.52,              # 4–6 curation / 12 removed / 18 tasks
    0.52,                          # 7  +6 tasks
    0.64,                          # 8  24 combined
    0.74,                          # 9  inspect logs
    0.84,                          # 10  5 removed
    0.87,                          # 11 update grading (shifted left so label clears node 12)
    0.99,                          # 12 final
]
node_y = [
    0.40,                          # 0  source (center)
    0.10, 0.38, 0.66,              # 1–3 branches
    0.38, 0.10, 0.52,              # 4–6 curation / 12 removed / 18 tasks
    0.76,                          # 7  +6 tasks
    0.60,                          # 8  24 combined
    0.55,                          # 9  inspect logs
    0.82,                          # 10  5 removed
    0.72,                          # 11 update grading
    0.52,                          # 12 final
]

# ── Links ─────────────────────────────────────────────────────────────────────
# Flow accounting (task units):
#   30 tasks → 3 branches (10 each) → Curation → 12 removed + 18 continue
#   18 tasks + 6 new tasks → 24 combined
#   24 → Inspect logs → 5 removed + 18 direct to final + 1 → update grading
#   update grading → final  (total final = 18 + 1 = 19)

src = [
    0, 0, 0,         # source → 3 branches
    1, 2, 3,         # branches → curation
    4, 4,            # curation → 12 removed / 18 tasks
    6, 7,            # 18 tasks + 6 new → 24 combined
    8,               # 24 → inspect logs
    9, 9, 9,         # inspect logs → 5 removed / update grading / 18 direct to final
    11,              # update grading → final
]
tgt = [
    1, 2, 3,
    4, 4, 4,
    5, 6,
    8, 8,
    9,
    10, 11, 12,
    12,
]
val = [
    10, 10, 10,      # 30 ÷ 3 branches
    10, 10, 10,      # branches → curation (sum = 30)
    12, 18,          # 12 removed; 18 continue
    18, 6,           # merge: 18 curated + 6 new = 24
    24,              # all 24 → inspect logs
    5, 1, 18,        # 5 removed; 1 → update grading; 18 direct to final
    1,               # update grading → final
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
