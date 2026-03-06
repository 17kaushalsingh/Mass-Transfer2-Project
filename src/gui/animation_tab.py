"""
Animation tab: generate and preview animated GIF visualizations of
extraction processes — stage-by-stage, ternary, composition, and
parameter sweeps.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..core.equilibrium import EquilibriumModel


# -------------------------------------------------------------------
# Worker that generates the animation in a background thread
# -------------------------------------------------------------------

class AnimationWorker(QThread):
    """Generate an animation object (and optional GIF bytes) off-thread."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)   # emits dict with anim + metadata
    error = pyqtSignal(str)

    def __init__(self, anim_type: str, eq_model, result, params: dict):
        super().__init__()
        self.anim_type = anim_type
        self.eq_model = eq_model
        self.result = result
        self.params = params

    def run(self):
        try:
            from ..viz.animations import (
                animate_xy_stepping,
                animate_ternary_buildup,
                animate_composition_profile,
                animate_parameter_sweep,
                save_animation_gif,
            )
            import tempfile, os

            fps = self.params.get("fps", 3)

            if self.anim_type == "xy_stepping":
                anim = animate_xy_stepping(
                    self.eq_model, self.result.stages, self.result, fps=fps,
                )
            elif self.anim_type == "ternary":
                anim = animate_ternary_buildup(
                    self.eq_model, self.result.stages, self.result, fps=fps,
                )
            elif self.anim_type == "composition":
                anim = animate_composition_profile(
                    self.result.stages, self.result, fps=fps,
                )
            elif self.anim_type == "param_sweep":
                anim = animate_parameter_sweep(
                    self.eq_model,
                    sweep_var=self.params["sweep_var"],
                    sweep_range=self.params["sweep_range"],
                    fixed_vals=self.params["fixed_vals"],
                    n_frames=self.params.get("n_frames", 30),
                    fps=fps,
                    progress_callback=lambda c, t: self.progress.emit(c, t),
                )
            else:
                raise ValueError(f"Unknown animation type: {self.anim_type}")

            # Save to a temp GIF so we can replay frame-by-frame
            tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
            tmp_path = tmp.name
            tmp.close()
            save_animation_gif(anim, tmp_path, fps=fps, dpi=90)

            self.finished.emit({
                "anim": anim,
                "gif_path": tmp_path,
                "fps": fps,
                "n_frames": anim._fig is not None,
            })

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


# -------------------------------------------------------------------
# Solver worker (re-run a sim if user doesn't have a result yet)
# -------------------------------------------------------------------

class _SolverWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, solver_func, kwargs):
        super().__init__()
        self.solver_func = solver_func
        self.kwargs = kwargs

    def run(self):
        try:
            self.finished.emit(self.solver_func(**self.kwargs))
        except Exception as e:
            self.error.emit(str(e))


# -------------------------------------------------------------------
# Animation Tab
# -------------------------------------------------------------------

ANIM_TYPES = [
    ("Stage-by-Stage X-Y Stepping", "xy_stepping"),
    ("Ternary Diagram Build-up", "ternary"),
    ("Composition Profile Evolution", "composition"),
    ("Parameter Sensitivity Sweep", "param_sweep"),
]


class AnimationTab(QWidget):
    """Tab for generating and previewing animated GIF visualizations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.eq_model: Optional[EquilibriumModel] = None
        self._last_result = None       # most recent solver result
        self._gif_path: Optional[str] = None
        self._anim = None
        self._frame_idx = 0
        self._frame_images = []
        self._timer: Optional[QTimer] = None
        self._worker: Optional[AnimationWorker] = None
        self._setup_ui()

    # ==================================================================
    # UI setup
    # ==================================================================

    def _setup_ui(self):
        root = QHBoxLayout(self)

        # ---- LEFT: Controls ----
        left = QVBoxLayout()

        # Animation type
        type_group = QGroupBox("Animation Type")
        tl = QFormLayout(type_group)
        self.type_combo = QComboBox()
        for label, _ in ANIM_TYPES:
            self.type_combo.addItem(label)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        tl.addRow("Type:", self.type_combo)
        left.addWidget(type_group)

        # Source controls
        src_group = QGroupBox("Simulation Source")
        sl = QFormLayout(src_group)

        self.src_combo = QComboBox()
        self.src_combo.addItems([
            "Use last simulation result",
            "Run fresh Crosscurrent",
        ])
        sl.addRow("Source:", self.src_combo)

        self.src_stages = QSpinBox()
        self.src_stages.setRange(1, 50); self.src_stages.setValue(3)
        sl.addRow("Stages:", self.src_stages)

        self.src_solvent = QDoubleSpinBox()
        self.src_solvent.setRange(1, 100000); self.src_solvent.setValue(1000)
        self.src_solvent.setSuffix(" kg")
        sl.addRow("Solvent/stage:", self.src_solvent)

        self.src_feed_A = QDoubleSpinBox()
        self.src_feed_A.setRange(0, 100); self.src_feed_A.setValue(75.0)
        self.src_feed_A.setSuffix(" wt%")
        sl.addRow("Carrier (A):", self.src_feed_A)

        self.src_feed_C = QDoubleSpinBox()
        self.src_feed_C.setRange(0, 100); self.src_feed_C.setValue(25.0)
        self.src_feed_C.setSuffix(" wt%")
        sl.addRow("Solute (C):", self.src_feed_C)

        left.addWidget(src_group)

        # Speed
        speed_group = QGroupBox("Playback")
        spl = QFormLayout(speed_group)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 10); self.fps_spin.setValue(3)
        self.fps_spin.setSuffix(" FPS")
        spl.addRow("Speed:", self.fps_spin)
        left.addWidget(speed_group)

        # Parameter sweep controls (shown only in sweep mode)
        self.sweep_group = QGroupBox("Sweep Parameters")
        swl = QFormLayout(self.sweep_group)

        self.sweep_var_combo = QComboBox()
        self.sweep_var_combo.addItems(["solvent_per_stage", "n_stages", "feed_acid_pct"])
        swl.addRow("Sweep variable:", self.sweep_var_combo)

        self.sweep_min_spin = QDoubleSpinBox()
        self.sweep_min_spin.setRange(1, 50000); self.sweep_min_spin.setValue(100)
        swl.addRow("Range min:", self.sweep_min_spin)

        self.sweep_max_spin = QDoubleSpinBox()
        self.sweep_max_spin.setRange(1, 50000); self.sweep_max_spin.setValue(5000)
        swl.addRow("Range max:", self.sweep_max_spin)

        self.sweep_frames_spin = QSpinBox()
        self.sweep_frames_spin.setRange(5, 100); self.sweep_frames_spin.setValue(30)
        swl.addRow("Frames:", self.sweep_frames_spin)

        self.sweep_fixed_stages = QSpinBox()
        self.sweep_fixed_stages.setRange(1, 20); self.sweep_fixed_stages.setValue(5)
        swl.addRow("Fixed stages:", self.sweep_fixed_stages)

        self.sweep_fixed_solvent = QDoubleSpinBox()
        self.sweep_fixed_solvent.setRange(1, 50000); self.sweep_fixed_solvent.setValue(1000)
        swl.addRow("Fixed solvent:", self.sweep_fixed_solvent)

        self.sweep_fixed_acid = QDoubleSpinBox()
        self.sweep_fixed_acid.setRange(1, 50); self.sweep_fixed_acid.setValue(25)
        swl.addRow("Fixed acid %:", self.sweep_fixed_acid)

        left.addWidget(self.sweep_group)
        self.sweep_group.hide()

        # Buttons
        btn_layout = QHBoxLayout()
        self.gen_btn = QPushButton("▶ Generate Animation")
        self.gen_btn.setMinimumHeight(44)
        self.gen_btn.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.gen_btn.clicked.connect(self._generate)
        btn_layout.addWidget(self.gen_btn)

        self.save_btn = QPushButton("💾 Save GIF")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_gif)
        btn_layout.addWidget(self.save_btn)
        left.addLayout(btn_layout)

        self.progress = QProgressBar()
        left.addWidget(self.progress)

        self.status_label = QLabel("Select animation type and click Generate.")
        self.status_label.setWordWrap(True)
        left.addWidget(self.status_label)
        left.addStretch()

        root.addLayout(left, stretch=1)

        # ---- RIGHT: Preview ----
        right = QVBoxLayout()

        preview_group = QGroupBox("Animation Preview")
        pv = QVBoxLayout(preview_group)

        self.canvas = FigureCanvas(Figure(figsize=(11, 8)))
        pv.addWidget(self.canvas)

        # Playback controls
        playback_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._toggle_playback)
        playback_layout.addWidget(self.play_btn)

        self.frame_label = QLabel("Frame: —")
        playback_layout.addWidget(self.frame_label)
        playback_layout.addStretch()
        pv.addLayout(playback_layout)

        right.addWidget(preview_group)
        root.addLayout(right, stretch=3)

    # ==================================================================
    # Public
    # ==================================================================

    def set_model(self, eq_model: "EquilibriumModel"):
        self.eq_model = eq_model

    def set_result(self, result):
        """Called from sim tab when a new result is available."""
        self._last_result = result

    # ==================================================================
    # Type toggle
    # ==================================================================

    def _on_type_changed(self, idx):
        is_sweep = (ANIM_TYPES[idx][1] == "param_sweep")
        if is_sweep:
            self.sweep_group.show()
        else:
            self.sweep_group.hide()

    # ==================================================================
    # Generate animation
    # ==================================================================

    def _generate(self):
        if self.eq_model is None:
            QMessageBox.warning(self, "No Model", "Load equilibrium data first.")
            return

        anim_key = ANIM_TYPES[self.type_combo.currentIndex()][1]
        fps = self.fps_spin.value()

        if anim_key == "param_sweep":
            # Sweep doesn't need a prior result; it solves internally
            self._start_animation_worker(anim_key, None, fps)
            return

        # Non-sweep: need a solver result
        if self.src_combo.currentIndex() == 0 and self._last_result is not None:
            self._start_animation_worker(anim_key, self._last_result, fps)
        else:
            # Run a fresh crosscurrent solve first
            self.gen_btn.setEnabled(False)
            self.status_label.setText("Running solver…")
            from ..core.crosscurrent import solve_crosscurrent
            kwargs = dict(
                feed_A=self.src_feed_A.value(),
                feed_C=self.src_feed_C.value(),
                feed_flow=100.0,
                solvent_per_stage=self.src_solvent.value(),
                n_stages=self.src_stages.value(),
                eq_model=self.eq_model,
            )
            self._solver_worker = _SolverWorker(solve_crosscurrent, kwargs)
            self._solver_worker.finished.connect(
                lambda r: self._on_solver_done(r, anim_key, fps)
            )
            self._solver_worker.error.connect(self._on_solver_error)
            self._solver_worker.start()

    def _on_solver_done(self, result, anim_key, fps):
        self._last_result = result
        self._start_animation_worker(anim_key, result, fps)

    def _on_solver_error(self, msg):
        self.gen_btn.setEnabled(True)
        self.status_label.setText("Solver error!")
        QMessageBox.critical(self, "Solver Error", msg)

    def _start_animation_worker(self, anim_key, result, fps):
        self.gen_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.status_label.setText("Generating animation frames…")
        self.progress.setValue(0)

        params = {"fps": fps}
        if anim_key == "param_sweep":
            sweep_var = self.sweep_var_combo.currentText()
            params["sweep_var"] = sweep_var
            params["sweep_range"] = (self.sweep_min_spin.value(), self.sweep_max_spin.value())
            params["n_frames"] = self.sweep_frames_spin.value()
            params["fixed_vals"] = {
                "n_stages": float(self.sweep_fixed_stages.value()),
                "solvent_per_stage": self.sweep_fixed_solvent.value(),
                "feed_acid_pct": self.sweep_fixed_acid.value(),
            }

        self._worker = AnimationWorker(anim_key, self.eq_model, result, params)
        self._worker.progress.connect(self._on_gen_progress)
        self._worker.finished.connect(self._on_gen_done)
        self._worker.error.connect(self._on_gen_error)
        self._worker.start()

    def _on_gen_progress(self, current, total):
        self.progress.setMaximum(total)
        self.progress.setValue(current)

    def _on_gen_done(self, data: dict):
        self.gen_btn.setEnabled(True)
        self._gif_path = data.get("gif_path")
        self._anim = data.get("anim")

        # Load GIF frames into PIL for preview playback
        try:
            from PIL import Image
            import io
            img = Image.open(self._gif_path)
            self._frame_images = []
            try:
                while True:
                    frame = img.copy().convert("RGBA")
                    self._frame_images.append(frame)
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

            n = len(self._frame_images)
            import os
            size_kb = os.path.getsize(self._gif_path) / 1024

            self.status_label.setText(
                f"<b>Done!</b> {n} frames, "
                f"{n / data['fps']:.1f}s duration, "
                f"{size_kb:.0f} KB"
            )
            self.play_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self._frame_idx = 0
            self._show_frame(0)
            self.frame_label.setText(f"Frame: 1 / {n}")

        except Exception as e:
            self.status_label.setText(f"GIF saved but preview failed: {e}")
            self.save_btn.setEnabled(True)

    def _on_gen_error(self, msg):
        self.gen_btn.setEnabled(True)
        self.status_label.setText("Error generating animation!")
        QMessageBox.critical(self, "Animation Error", msg)

    # ==================================================================
    # Playback
    # ==================================================================

    def _show_frame(self, idx):
        """Render a single PIL frame onto the canvas."""
        if not self._frame_images or idx >= len(self._frame_images):
            return

        import numpy as np
        frame = self._frame_images[idx]
        arr = np.array(frame)

        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(arr)
        ax.set_axis_off()
        self.canvas.draw()

    def _toggle_playback(self):
        if self._timer is not None and self._timer.isActive():
            self._timer.stop()
            self.play_btn.setText("▶ Play")
            return

        if not self._frame_images:
            return

        self._frame_idx = 0
        fps = self.fps_spin.value()
        interval_ms = max(50, 1000 // fps)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)
        self._timer.start(interval_ms)
        self.play_btn.setText("⏸ Pause")

    def _advance_frame(self):
        n = len(self._frame_images)
        if self._frame_idx >= n:
            self._timer.stop()
            self.play_btn.setText("▶ Play")
            return

        self._show_frame(self._frame_idx)
        self.frame_label.setText(f"Frame: {self._frame_idx + 1} / {n}")
        self._frame_idx += 1

    # ==================================================================
    # Save
    # ==================================================================

    def _save_gif(self):
        if self._gif_path is None:
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Animation GIF", "extraction_animation.gif",
            "GIF Files (*.gif);;All Files (*)"
        )
        if filepath:
            import shutil
            shutil.copy2(self._gif_path, filepath)
            QMessageBox.information(self, "Saved", f"Animation saved to {filepath}")
