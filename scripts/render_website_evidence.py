"""Render static allocation figures directly from the published smoke records."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(source: str, target: str, label: str) -> None:
    data = json.loads((ROOT / "website/evidence/smoke" / source).read_text())
    trials = sorted(data["trials"], key=lambda trial: trial["id"])
    cap = 27
    assert len(trials) == 16
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 392 308" '
        'role="img" aria-labelledby="title desc">',
        f'<title id="title">{label}: objective evaluations by candidate</title>',
        '<desc id="desc">Each square is one replication. '
        "Rows are candidate IDs, columns are replication indices.</desc>",
        '<g font-family="Arial, sans-serif" font-size="10" fill="#5d6961">',
        '<text x="0" y="15">ID</text>',
        '<text x="47" y="15">01</text>',
        '<text x="143" y="15">09</text>',
        '<text x="251" y="15">18</text>',
        '<text x="359" y="15">27</text>',
        "</g>",
    ]
    for row, trial in enumerate(trials):
        count = len(trial["observations"])
        assert count <= cap
        y = 29 + row * 16
        svg.append(
            f'<text x="0" y="{y + 9}" font-family="Arial, sans-serif" '
            f'font-size="10" fill="#5d6961">{trial["id"]:02d}</text>'
        )
        for col in range(cap):
            color = "#204d91" if col < count else "#e7ece6"
            svg.append(f'<rect x="{47 + col * 12}" y="{y}" width="9" height="9" fill="{color}"/>')
    svg.append(
        '<text x="47" y="300" font-family="Arial, sans-serif" font-size="10" '
        'fill="#5d6961">Replication index →</text></svg>'
    )
    (ROOT / "website/assets" / target).write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    render("full_budget.json", "full-budget.svg", "Full budget")
    render("race.json", "replication-race.svg", "Replication race")
