"""
Welcome tab for the desktop application.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class HomeTab(QWidget):
    """Landing tab with workflow guidance and quick actions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(18)

        hero = QFrame()
        hero.setStyleSheet(
            "QFrame {"
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:1,"
            "stop:0 rgba(201,139,46,0.20), stop:1 rgba(78,121,167,0.16));"
            "border: 1px solid #435058; border-radius: 18px;}"
        )
        hero_layout = QVBoxLayout(hero)
        hero_layout.setSpacing(10)

        title = QLabel("Multistage Extraction Digital Twin")
        title.setStyleSheet("font-size: 28px; font-weight: 700; color: #f7f2e8;")
        subtitle = QLabel(
            "Simulate crosscurrent and countercurrent extraction, inspect stage-wise "
            "behavior, and train a surrogate model from generated operating data."
        )
        subtitle.setWordWrap(True)
        subtitle.setProperty("class", "sectionIntro")

        pill_row = QHBoxLayout()
        for text in [
            "Numerical Solvers",
            "Interactive Visuals",
            "ANN Surrogate Model",
            "Animated Workflows",
        ]:
            pill = QLabel(text)
            pill.setStyleSheet(
                "background: rgba(12, 16, 18, 0.72); border: 1px solid #4e5d65; "
                "border-radius: 14px; padding: 6px 10px; color: #f1eee6;"
            )
            pill_row.addWidget(pill)
        pill_row.addStretch()

        action_row = QHBoxLayout()
        self.load_default_btn = QPushButton("Load Default Data")
        self.load_default_btn.setProperty("class", "primary")
        self.open_sim_btn = QPushButton("Start Simulation")
        self.open_surrogate_btn = QPushButton("Open Surrogate Model")
        action_row.addWidget(self.load_default_btn)
        action_row.addWidget(self.open_sim_btn)
        action_row.addWidget(self.open_surrogate_btn)
        action_row.addStretch()

        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        hero_layout.addLayout(pill_row)
        hero_layout.addLayout(action_row)
        root.addWidget(hero)

        cards = QGridLayout()
        cards.setHorizontalSpacing(14)
        cards.setVerticalSpacing(14)

        cards.addWidget(self._make_card(
            "Recommended Workflow",
            [
                "1. Confirm or edit tie-line data in Data Input.",
                "2. Run a simulation and inspect the stage diagram.",
                "3. Review heatmaps and comparisons.",
                "4. Generate data and train the surrogate model.",
            ],
        ), 0, 0)
        cards.addWidget(self._make_card(
            "What Loads By Default",
            [
                "Cottonseed oil / oleic acid / propane tie-line data.",
                "Ready-to-run crosscurrent and countercurrent defaults.",
                "Packaged JSON resource, so startup no longer depends on repo-root paths.",
            ],
        ), 0, 1)
        cards.addWidget(self._make_card(
            "Best Places To Start",
            [
                "Use Simulation for process calculations.",
                "Use Heatmaps for stage-by-stage trends.",
                "Use Surrogate Model for training, prediction, and optimization.",
            ],
        ), 1, 0)
        cards.addWidget(self._make_card(
            "Friendly Reminders",
            [
                "Blank plots now indicate the next action instead of showing an empty canvas.",
                "Primary actions are highlighted in amber.",
                "The Surrogate controls scroll on smaller windows.",
            ],
        ), 1, 1)
        root.addLayout(cards)

        footer = QLabel(
            "Tip: keep the default data loaded for a fast first pass, then swap in your own "
            "JSON data file from the File menu when you want to compare another ternary system."
        )
        footer.setProperty("class", "helperText")
        footer.setWordWrap(True)
        root.addWidget(footer)
        root.addStretch()

    def _make_card(self, title: str, lines: list[str]) -> QWidget:
        card = QFrame()
        card.setStyleSheet(
            "QFrame {background: rgba(23, 29, 32, 0.94); border: 1px solid #384349; "
            "border-radius: 16px;}"
        )
        layout = QVBoxLayout(card)
        header = QLabel(title)
        header.setStyleSheet("font-size: 17px; font-weight: 700; color: #f6c36d;")
        layout.addWidget(header)

        for line in lines:
            label = QLabel(line)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            label.setProperty("class", "sectionIntro")
            layout.addWidget(label)

        layout.addStretch()
        return card
