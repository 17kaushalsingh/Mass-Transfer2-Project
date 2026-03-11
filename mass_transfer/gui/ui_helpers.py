"""
Shared GUI helper utilities.
"""

from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation
from PyQt6.QtWidgets import QGraphicsOpacityEffect


def draw_empty_figure(figure, title: str, subtitle: str) -> None:
    """Render a consistent empty state inside a Matplotlib figure."""
    figure.clear()
    ax = figure.add_subplot(111)
    ax.set_facecolor("#12181c")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        0.5, 0.58, title,
        ha="center", va="center",
        fontsize=18, fontweight="bold",
        color="#f5efe3",
        transform=ax.transAxes,
    )
    ax.text(
        0.5, 0.44, subtitle,
        ha="center", va="center",
        fontsize=11,
        color="#aab6ba",
        transform=ax.transAxes,
        wrap=True,
    )

    # A subtle engineering-grid motif keeps blank states from feeling broken.
    for idx in range(1, 10):
        alpha = 0.08 if idx % 2 == 0 else 0.04
        ax.axhline(idx / 10, color="#c98b2e", lw=0.6, alpha=alpha)
        ax.axvline(idx / 10, color="#66a7b8", lw=0.6, alpha=alpha)

    figure.tight_layout()


def animate_widget_in(widget, duration: int = 220) -> None:
    """Fade a widget in after a major content change."""
    effect = widget.graphicsEffect()
    if not isinstance(effect, QGraphicsOpacityEffect):
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)

    animation = QPropertyAnimation(effect, b"opacity", widget)
    animation.setDuration(duration)
    animation.setStartValue(0.35)
    animation.setEndValue(1.0)
    animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    animation.start()
    widget._fade_animation = animation
