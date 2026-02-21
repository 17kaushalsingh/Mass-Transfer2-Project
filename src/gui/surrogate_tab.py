"""
Surrogate Model tab: data generation, ANN training, prediction, and response surfaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
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

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..core.equilibrium import EquilibriumModel


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

class DataGenWorker(QThread):
    """Background thread for dataset generation."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, eq_model, config):
        super().__init__()
        self.eq_model = eq_model
        self.config = config

    def run(self):
        try:
            from ..ml.data_generator import generate_crosscurrent_dataset_serial
            df = generate_crosscurrent_dataset_serial(
                self.eq_model, self.config,
                progress_callback=lambda cur, tot: self.progress.emit(cur, tot),
            )
            self.finished.emit(df)
        except Exception as e:
            self.error.emit(str(e))


class TrainWorker(QThread):
    """Background thread for model training."""
    progress = pyqtSignal(int, int, float, float)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, df, config):
        super().__init__()
        self.df = df
        self.config = config

    def run(self):
        try:
            from ..ml.neural_net import train_model
            result = train_model(
                self.df, self.config,
                progress_callback=lambda ep, tot, tl, vl: self.progress.emit(ep, tot, tl, vl),
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Surrogate Tab
# ---------------------------------------------------------------------------

class SurrogateTab(QWidget):
    """Tab for surrogate model operations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.eq_model: Optional[EquilibriumModel] = None
        self.data_path: str = "data.json"
        self.dataset = None
        self.training_result = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)

        # Left panel: controls
        left = QVBoxLayout()

        # Data generation
        gen_group = QGroupBox("1. Generate Training Data")
        gen_layout = QFormLayout(gen_group)

        self.n_samples_spin = QSpinBox()
        self.n_samples_spin.setRange(100, 50000)
        self.n_samples_spin.setValue(2000)
        self.n_samples_spin.setSingleStep(500)
        gen_layout.addRow("Samples:", self.n_samples_spin)

        self.gen_btn = QPushButton("Generate Data")
        self.gen_btn.clicked.connect(self._generate_data)
        gen_layout.addRow(self.gen_btn)

        self.gen_progress = QProgressBar()
        gen_layout.addRow(self.gen_progress)

        self.gen_info = QLabel("")
        gen_layout.addRow(self.gen_info)

        left.addWidget(gen_group)

        # Training
        train_group = QGroupBox("2. Train Model")
        train_layout = QFormLayout(train_group)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 1000); self.epochs_spin.setValue(200)
        train_layout.addRow("Epochs:", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1); self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(4); self.lr_spin.setSingleStep(0.0001)
        train_layout.addRow("Learning rate:", self.lr_spin)

        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self._train_model)
        train_layout.addRow(self.train_btn)

        self.train_progress = QProgressBar()
        train_layout.addRow(self.train_progress)

        self.train_info = QLabel("")
        self.train_info.setWordWrap(True)
        train_layout.addRow(self.train_info)

        left.addWidget(train_group)

        # Prediction
        pred_group = QGroupBox("3. Predict")
        pred_layout = QFormLayout(pred_group)

        self.pred_stages = QSpinBox()
        self.pred_stages.setRange(1, 20); self.pred_stages.setValue(3)
        pred_layout.addRow("Stages:", self.pred_stages)

        self.pred_solvent = QDoubleSpinBox()
        self.pred_solvent.setRange(1, 50000); self.pred_solvent.setValue(1000)
        pred_layout.addRow("Solvent (kg):", self.pred_solvent)

        self.pred_acid = QDoubleSpinBox()
        self.pred_acid.setRange(1, 50); self.pred_acid.setValue(25)
        pred_layout.addRow("Feed acid %:", self.pred_acid)

        self.pred_btn = QPushButton("Predict")
        self.pred_btn.clicked.connect(self._predict)
        pred_layout.addRow(self.pred_btn)

        self.pred_result = QLabel("")
        self.pred_result.setStyleSheet("font-size: 16px; font-weight: bold;")
        pred_layout.addRow(self.pred_result)

        left.addWidget(pred_group)

        # Optimization
        opt_group = QGroupBox("4. Optimize")
        opt_layout = QFormLayout(opt_group)

        self.target_removal_spin = QDoubleSpinBox()
        self.target_removal_spin.setRange(10, 99); self.target_removal_spin.setValue(95)
        self.target_removal_spin.setSuffix(" %")
        opt_layout.addRow("Target removal:", self.target_removal_spin)

        self.opt_btn = QPushButton("Find Optimal")
        self.opt_btn.clicked.connect(self._optimize)
        opt_layout.addRow(self.opt_btn)

        self.opt_result = QLabel("")
        self.opt_result.setWordWrap(True)
        opt_layout.addRow(self.opt_result)

        left.addWidget(opt_group)
        left.addStretch()

        main_layout.addLayout(left, stretch=1)

        # Right panel: plots
        right = QVBoxLayout()

        # Response surface controls
        surf_ctrl = QHBoxLayout()
        surf_ctrl.addWidget(QLabel("Fixed var:"))
        self.fixed_var_combo = QComboBox()
        self.fixed_var_combo.addItems(["feed_acid_pct", "n_stages", "solvent_per_stage"])
        surf_ctrl.addWidget(self.fixed_var_combo)

        surf_ctrl.addWidget(QLabel("Value:"))
        self.fixed_var_slider = QDoubleSpinBox()
        self.fixed_var_slider.setRange(1, 5000)
        self.fixed_var_slider.setValue(25.0)
        surf_ctrl.addWidget(self.fixed_var_slider)

        self.surface_btn = QPushButton("Plot Surface")
        self.surface_btn.clicked.connect(self._plot_surface)
        surf_ctrl.addWidget(self.surface_btn)
        right.addLayout(surf_ctrl)

        # Canvas for loss curve and response surface
        self.canvas = FigureCanvas(Figure(figsize=(10, 8)))
        right.addWidget(self.canvas)

        main_layout.addLayout(right, stretch=2)

    def set_model(self, eq_model: EquilibriumModel, data_path: str = "data.json"):
        self.eq_model = eq_model
        self.data_path = data_path

    def _generate_data(self):
        if self.eq_model is None:
            QMessageBox.warning(self, "No Model", "Load equilibrium data first.")
            return

        from ..ml.data_generator import DataGenConfig
        config = DataGenConfig(n_samples=self.n_samples_spin.value())

        self.gen_btn.setEnabled(False)
        self.gen_info.setText("Generating...")

        self.gen_worker = DataGenWorker(self.eq_model, config)
        self.gen_worker.progress.connect(self._on_gen_progress)
        self.gen_worker.finished.connect(self._on_gen_done)
        self.gen_worker.error.connect(self._on_gen_error)
        self.gen_worker.start()

    def _on_gen_progress(self, current, total):
        self.gen_progress.setMaximum(total)
        self.gen_progress.setValue(current)

    def _on_gen_done(self, df):
        self.gen_btn.setEnabled(True)
        self.dataset = df
        self.gen_info.setText(f"Generated {len(df)} samples.")

    def _on_gen_error(self, msg):
        self.gen_btn.setEnabled(True)
        self.gen_info.setText("Error!")
        QMessageBox.critical(self, "Generation Error", msg)

    def _train_model(self):
        if self.dataset is None:
            QMessageBox.warning(self, "No Data", "Generate training data first.")
            return

        from ..ml.neural_net import TrainingConfig
        config = TrainingConfig(
            epochs=self.epochs_spin.value(),
            learning_rate=self.lr_spin.value(),
        )

        self.train_btn.setEnabled(False)
        self.train_info.setText("Training...")

        self.train_worker = TrainWorker(self.dataset, config)
        self.train_worker.progress.connect(self._on_train_progress)
        self.train_worker.finished.connect(self._on_train_done)
        self.train_worker.error.connect(self._on_train_error)
        self.train_worker.start()

    def _on_train_progress(self, epoch, total, train_loss, val_loss):
        self.train_progress.setMaximum(total)
        self.train_progress.setValue(epoch)
        self.train_info.setText(f"Epoch {epoch}/{total} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Update live loss curve
        if epoch % 5 == 0 or epoch == total:
            self._update_loss_plot()

    def _on_train_done(self, result):
        self.train_btn.setEnabled(True)
        self.training_result = result
        self.train_info.setText(
            f"<b>Done!</b> R²={result.test_r_squared:.4f}, "
            f"MAE={result.test_mae:.2f}%, RMSE={result.test_rmse:.2f}%"
        )
        self._update_loss_plot()

    def _on_train_error(self, msg):
        self.train_btn.setEnabled(True)
        self.train_info.setText("Error!")
        QMessageBox.critical(self, "Training Error", msg)

    def _update_loss_plot(self):
        if self.training_result is None:
            return
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.plot(self.training_result.train_losses, label="Train Loss")
        ax.plot(self.training_result.val_losses, label="Val Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.set_title(f"Training Curve (R²={self.training_result.test_r_squared:.4f})")
        ax.legend(); ax.grid(True, alpha=0.3)
        if len(self.training_result.train_losses) > 10:
            ax.set_yscale("log")
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def _predict(self):
        if self.training_result is None:
            QMessageBox.warning(self, "No Model", "Train a model first.")
            return

        from ..ml.neural_net import predict
        result = predict(
            self.training_result.model,
            self.training_result.scaler_X,
            self.training_result.scaler_y,
            n_stages=self.pred_stages.value(),
            solvent=self.pred_solvent.value(),
            feed_comp=self.pred_acid.value(),
        )
        self.pred_result.setText(f"Predicted removal: {result:.2f}%")

    def _optimize(self):
        if self.training_result is None:
            QMessageBox.warning(self, "No Model", "Train a model first.")
            return

        from ..ml.optimization import find_optimal_conditions
        result = find_optimal_conditions(
            self.training_result,
            target_removal=self.target_removal_spin.value(),
        )
        self.opt_result.setText(
            f"<b>Optimal:</b><br>"
            f"Stages: {result['n_stages']}<br>"
            f"Solvent/stage: {result['solvent_per_stage']:.0f} kg<br>"
            f"Predicted removal: {result['predicted_removal']:.2f}%<br>"
            f"Total solvent: {result['total_solvent']:.0f} kg"
        )

    def _plot_surface(self):
        if self.training_result is None:
            QMessageBox.warning(self, "No Model", "Train a model first.")
            return

        from ..ml.optimization import generate_response_surface

        fixed_var = self.fixed_var_combo.currentText()
        fixed_val = self.fixed_var_slider.value()

        # Determine the other two variables
        all_vars = ["n_stages", "solvent_per_stage", "feed_acid_pct"]
        var_ranges = {
            "n_stages": (1, 15),
            "solvent_per_stage": (100, 5000),
            "feed_acid_pct": (5, 45),
        }
        var_labels = {
            "n_stages": "Number of Stages",
            "solvent_per_stage": "Solvent per Stage (kg)",
            "feed_acid_pct": "Feed Acid (%)",
        }

        others = [v for v in all_vars if v != fixed_var]

        X, Y, Z = generate_response_surface(
            self.training_result,
            var1_name=others[0],
            var2_name=others[1],
            fixed_var_name=fixed_var,
            fixed_var_value=fixed_val,
            var1_range=var_ranges[others[0]],
            var2_range=var_ranges[others[1]],
            grid_size=30,
        )

        self.canvas.figure.clear()

        # 3D surface
        ax = self.canvas.figure.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8, edgecolor="none")
        self.canvas.figure.colorbar(surf, ax=ax, label="% Removal", shrink=0.6)
        ax.set_xlabel(var_labels[others[0]])
        ax.set_ylabel(var_labels[others[1]])
        ax.set_zlabel("% Removal")
        ax.set_title(f"Response Surface ({var_labels[fixed_var]} = {fixed_val:.1f})")

        self.canvas.figure.tight_layout()
        self.canvas.draw()
