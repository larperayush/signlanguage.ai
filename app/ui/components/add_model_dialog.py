"""
Smart Model Importer & Cloud Models Hub Dialog for SignFlow Studio.
Features:
1. Official GitHub Models Hub:
   - Direct 1-click download, auto-extraction, validation, and registration of:
     * SignFlow-model-v1 (USER) (rajatcoder132/SignFlow-model-v1-USER)
     * SignFlow-model-v1 (DEV) (rajatcoder132/SignFlow-model-v1-DEV)
2. Local Model Folder Import:
   - Discover and validate custom local model folders (User or Developer suites).
"""
from pathlib import Path
import urllib.request
import zipfile
import io
import shutil
import time
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QFrame, QMessageBox, QTabWidget,
    QWidget, QProgressBar, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QThread, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap, QPainter, QPainterPath, QColor, QPen

from app.core.config import MODELS_DIR
from app.core.theme import ThemeManager
from app.core.model_registry import ModelRegistry, detect_model_in_folder, ModelPackage
from app.resources.icons import get_icon


def make_circular_avatar(image_path: Path, diameter: int = 36, is_official: bool = True) -> QPixmap:
    """Create a guaranteed, mathematically perfect anti-aliased circular avatar with a subtle ring."""
    if not image_path.exists():
        return QPixmap()

    src = QPixmap(str(image_path))
    if src.isNull():
        return QPixmap()

    scale = 2  # 2x for High-DPI sharpness
    canvas_size = diameter * scale

    min_dim = min(src.width(), src.height())
    sx = (src.width() - min_dim) // 2
    sy = (src.height() - min_dim) // 2
    cropped = src.copy(sx, sy, min_dim, min_dim)
    scaled = cropped.scaled(
        canvas_size, canvas_size,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation
    )

    out = QPixmap(canvas_size, canvas_size)
    out.fill(Qt.GlobalColor.transparent)

    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    inset = 2 * scale
    draw_size = canvas_size - (2 * inset)

    path = QPainterPath()
    path.addEllipse(inset, inset, draw_size, draw_size)
    p.setClipPath(path)
    p.drawPixmap(0, 0, scaled)

    p.setClipping(False)
    ring_color = QColor(245, 180, 25, 220) if is_official else QColor(160, 160, 160, 140)
    p.setPen(QPen(ring_color, 1.5 * scale))
    p.drawEllipse(inset, inset, draw_size, draw_size)
    p.end()

    out.setDevicePixelRatio(scale)
    return out


class GitHubModelDownloader(QThread):
    """Background worker for downloading and extracting GitHub repository archives."""
    progress = Signal(int, str)             # percent (0-100), status message
    finished = Signal(bool, str, object)    # success, message, ModelPackage

    def __init__(self, repo_name: str, display_name: str, target_dir: Path):
        super().__init__()
        self.repo_name = repo_name
        self.display_name = display_name
        self.target_dir = target_dir
        self.zip_url = f"https://github.com/rajatcoder132/{repo_name}/archive/refs/heads/main.zip"
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            self.progress.emit(5, f"Connecting to GitHub (rajatcoder132/{self.repo_name})...")
            req = urllib.request.Request(
                self.zip_url,
                headers={"User-Agent": "SignFlow-Studio-Desktop"}
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                total_bytes = int(resp.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 64 * 1024
                buffer = io.BytesIO()

                while not self._is_cancelled:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    buffer.write(chunk)
                    downloaded += len(chunk)

                    if total_bytes > 0:
                        pct = int(10 + (downloaded / total_bytes) * 70)
                        dl_mb = downloaded / (1024 * 1024)
                        tot_mb = total_bytes / (1024 * 1024)
                        self.progress.emit(pct, f"Downloading: {dl_mb:.1f} MB / {tot_mb:.1f} MB ({pct}%)")
                    else:
                        dl_mb = downloaded / (1024 * 1024)
                        self.progress.emit(45, f"Downloading: {dl_mb:.1f} MB...")

                if self._is_cancelled:
                    self.finished.emit(False, "Download cancelled by user.", None)
                    return

            self.progress.emit(82, "Unpacking model package archive...")
            buffer.seek(0)
            zf = zipfile.ZipFile(buffer)

            # Ensure clean destination directory
            if self.target_dir.exists():
                shutil.rmtree(self.target_dir)
            self.target_dir.mkdir(parents=True, exist_ok=True)

            # GitHub zips put files inside <repo_name>-main/ -> strip root folder
            for member in zf.infolist():
                parts = member.filename.split("/", 1)
                if len(parts) > 1 and parts[1]:
                    rel_path = parts[1]
                    dest_file = self.target_dir / rel_path
                    if member.is_dir():
                        dest_file.mkdir(parents=True, exist_ok=True)
                    else:
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(member) as src, open(dest_file, "wb") as dst:
                            dst.write(src.read())

            self.progress.emit(92, "Verifying model artifacts and architecture...")
            reg = ModelRegistry.get_instance()
            success, pkg, msg = reg.register_model_package(
                folder_path=str(self.target_dir),
                custom_name=self.display_name
            )

            if success and pkg:
                self.progress.emit(100, f"Successfully installed and registered {self.display_name}!")
                self.finished.emit(True, f"Model '{self.display_name}' is ready to use!", pkg)
            else:
                self.finished.emit(False, f"Verification failed: {msg}", None)

        except Exception as e:
            self.finished.emit(False, f"Download failed: {str(e)}", None)


class AddModelDialog(QDialog):
    """Editorial dialog for discovering, downloading, and adding SignFlow AI models."""
    model_imported = Signal(object)  # Emits ModelPackage

    def __init__(self, parent=None, initial_tab: int = 0):
        super().__init__(parent)
        self.setWindowTitle("SignFlow AI Models Hub")
        self.setMinimumWidth(720)
        self.setMinimumHeight(600)
        self.setModal(True)

        self._active_downloader: Optional[GitHubModelDownloader] = None
        self._detected_data = None
        self._setup_ui()

        if initial_tab > 0 and initial_tab < self.tabs.count():
            self.tabs.setCurrentIndex(initial_tab)

    def _setup_ui(self):
        palette = ThemeManager.get_palette()
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(14)

        # Header Title & Editorial Note
        header_vbox = QVBoxLayout()
        header_vbox.setSpacing(3)
        title = QLabel("AI Model Harness Hub")
        title.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {palette.text_primary};")
        desc = QLabel(
            "Download official cloud-hosted SignFlow models directly from GitHub, import "
            "any local model package folder from your computer, or manage your installed models."
        )
        desc.setStyleSheet(f"font-size: 12px; color: {palette.text_secondary}; line-height: 1.4;")
        desc.setWordWrap(True)
        header_vbox.addWidget(title)
        header_vbox.addWidget(desc)
        main_layout.addLayout(header_vbox)

        # Main Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {palette.border_default};
                border-radius: 8px;
                background-color: {palette.bg_window};
                padding: 10px;
            }}
            QTabBar::tab {{
                background-color: {palette.bg_sidebar};
                color: {palette.text_secondary};
                font-weight: 600;
                font-size: 12px;
                padding: 8px 18px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                border: 1px solid {palette.border_subtle};
                border-bottom: none;
            }}
            QTabBar::tab:selected {{
                background-color: {palette.bg_card};
                color: {palette.accent_label if palette.is_dark else palette.text_primary};
                border-color: {palette.border_default};
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {palette.bg_hover};
            }}
        """)

        # Tab 1: Official GitHub Cloud Models
        cloud_tab = self._create_cloud_models_tab()
        self.tabs.addTab(cloud_tab, "Official GitHub Models")

        # Tab 2: Local Folder Import
        local_tab = self._create_local_import_tab()
        self.tabs.addTab(local_tab, "Local Model Folder")

        # Tab 3: Installed Models (Manage & Delete)
        installed_tab = self._create_installed_tab()
        self.tabs.addTab(installed_tab, "Installed Models")

        main_layout.addWidget(self.tabs, 1)

        # Bottom Close Button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.close_btn = QPushButton("Close")
        self.close_btn.setProperty("class", "secondaryBtn")
        self.close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.close_btn)
        main_layout.addLayout(btn_row)

    # -------------------------------------------------------------------------
    # TAB 1: OFFICIAL GITHUB CLOUD MODELS
    # -------------------------------------------------------------------------
    def _create_cloud_models_tab(self) -> QWidget:
        palette = ThemeManager.get_palette()
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(14)

        intro_lbl = QLabel(
            "Select an official repository below to download the latest trained model directly from GitHub. "
            "Downloaded packages are stored in your application models directory and automatically activated."
        )
        intro_lbl.setStyleSheet(f"font-size: 11.5px; color: {palette.text_secondary};")
        intro_lbl.setWordWrap(True)
        layout.addWidget(intro_lbl)

        self.cloud_cards_container = QWidget()
        self.cloud_cards_layout = QVBoxLayout(self.cloud_cards_container)
        self.cloud_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cloud_cards_layout.setSpacing(10)
        layout.addWidget(self.cloud_cards_container)

        self._refresh_cloud_tab()

        # Shared Download Progress Box (Hidden until download starts)
        self.progress_box = QFrame()
        self.progress_box.setProperty("class", "cardPanelElevated")
        self.progress_box.setVisible(False)
        p_layout = QVBoxLayout(self.progress_box)
        p_layout.setContentsMargins(14, 12, 14, 12)
        p_layout.setSpacing(8)

        self.progress_status = QLabel("Downloading model package from GitHub...")
        self.progress_status.setStyleSheet(f"font-size: 11.5px; font-weight: 600; color: {palette.text_primary};")
        p_layout.addWidget(self.progress_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {palette.bg_sidebar};
                border: 1px solid {palette.border_default};
                border-radius: 7px;
                text-align: center;
                font-size: 9.5px;
                color: {palette.text_primary};
            }}
            QProgressBar::chunk {{
                background-color: {palette.accent};
                border-radius: 6px;
            }}
        """)
        p_layout.addWidget(self.progress_bar)

        p_btn_row = QHBoxLayout()
        p_btn_row.addStretch()
        self.cancel_dl_btn = QPushButton("Cancel Download")
        self.cancel_dl_btn.setProperty("class", "secondaryBtn")
        self.cancel_dl_btn.clicked.connect(self._cancel_active_download)
        p_btn_row.addWidget(self.cancel_dl_btn)
        p_layout.addLayout(p_btn_row)

        layout.addWidget(self.progress_box)
        layout.addStretch()
        return tab_widget

    def _refresh_cloud_tab(self):
        while self.cloud_cards_layout.count():
            item = self.cloud_cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        palette = ThemeManager.get_palette()
        self.card_user = self._build_cloud_model_card(
            repo_slug="SignFlow-model-v1-USER",
            display_name="SignFlow-model-v1 (USER)",
            badge_text="USER MODEL • INFERENCE ONLY",
            badge_color=palette.accent_label,
            desc_text="Production-ready model package for high-speed live webcam ASL translation (30–60 FPS). "
                      "Includes pre-trained ASL classifier and MediaPipe hand landmarker bundle.",
            size_text="~6.0 MB archive",
            is_developer=False,
            author_name="Rajat",
            author_handle="@rajatcoder132",
            author_avatar_filename="author_avatar.png",
            author_url="https://github.com/rajatcoder132",
            is_official=True,
            role_badge="OFFICIAL CREATOR"
        )
        self.cloud_cards_layout.addWidget(self.card_user)

        self.card_dev = self._build_cloud_model_card(
            repo_slug="SignFlow-model-v1-DEV",
            display_name="SignFlow-model-v1 (DEV)",
            badge_text="DEVELOPER SUITE • TRAINING + INFERENCE",
            badge_color=palette.success,
            desc_text="Complete developer suite featuring automated static & dynamic gesture recorders, "
                      "multi-core Random Forest training pipeline, CSV datasets, and live HUD inference engine.",
            size_text="~6.5 MB archive",
            is_developer=True,
            author_name="Rajat",
            author_handle="@rajatcoder132",
            author_avatar_filename="author_avatar.png",
            author_url="https://github.com/rajatcoder132",
            is_official=True,
            role_badge="OFFICIAL CREATOR"
        )
        self.cloud_cards_layout.addWidget(self.card_dev)

    def _build_cloud_model_card(
        self,
        repo_slug: str,
        display_name: str,
        badge_text: str,
        badge_color: str,
        desc_text: str,
        size_text: str,
        is_developer: bool,
        author_name: str = "Rajat",
        author_handle: str = "@rajatcoder132",
        author_avatar_filename: str = "author_avatar.png",
        author_url: str = "https://github.com/rajatcoder132",
        is_official: bool = True,
        role_badge: str = "OFFICIAL CREATOR"
    ) -> QFrame:
        palette = ThemeManager.get_palette()
        card = QFrame()
        card.setProperty("class", "cardPanel")
        c_layout = QVBoxLayout(card)
        c_layout.setContentsMargins(16, 14, 16, 14)
        c_layout.setSpacing(10)

        # Header: Name, Badge, Size
        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        name_lbl = QLabel(display_name)
        name_lbl.setStyleSheet(f"font-size: 14.5px; font-weight: 700; color: {palette.text_primary};")
        top_row.addWidget(name_lbl)

        badge = QLabel(badge_text)
        badge.setStyleSheet(f"""
            font-size: 9.5px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
            background-color: {palette.bg_sidebar};
            color: {badge_color};
            border: 1px solid {palette.border_subtle};
        """)
        top_row.addWidget(badge)
        top_row.addStretch()

        size_lbl = QLabel(size_text)
        size_lbl.setStyleSheet(f"font-size: 11px; color: {palette.text_muted};")
        top_row.addWidget(size_lbl)
        c_layout.addLayout(top_row)

        # Creator Attribution Strip directly on each model bar
        creator_strip = QFrame()
        creator_strip.setStyleSheet(f"""
            QFrame {{
                background-color: {palette.bg_sidebar};
                border: 1px solid {palette.border_subtle};
                border-radius: 6px;
                padding: 4px 8px;
            }}
        """)
        cs_layout = QHBoxLayout(creator_strip)
        cs_layout.setContentsMargins(6, 4, 6, 4)
        cs_layout.setSpacing(10)

        # Profile avatar (Guaranteed pure circle with smooth ring and 18px radius)
        avatar_lbl = QLabel()
        avatar_lbl.setFixedSize(36, 36)
        avatar_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        avatar_lbl.setStyleSheet("background: transparent; border: none; border-radius: 18px;")
        avatar_path = Path(__file__).resolve().parent.parent.parent / "resources" / "icons" / author_avatar_filename
        avatar_pix = make_circular_avatar(avatar_path, diameter=36, is_official=is_official)
        if not avatar_pix.isNull():
            avatar_lbl.setPixmap(avatar_pix)

        author_info = QHBoxLayout()
        author_info.setSpacing(6)

        lbl_author = QLabel(author_name)
        lbl_author.setStyleSheet(f"font-size: 12px; font-weight: 700; color: {palette.text_primary};")

        lbl_handle = QLabel(author_handle)
        lbl_handle.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {palette.accent_label};")

        badge_official = QLabel(role_badge)
        badge_official.setStyleSheet(f"""
            font-size: 8.5px;
            font-weight: 800;
            padding: 2px 6px;
            border-radius: 3px;
            background-color: {palette.bg_card};
            color: {palette.success if is_official else palette.text_secondary};
            border: 1px solid {palette.border_subtle};
        """)

        author_info.addWidget(lbl_author)
        author_info.addWidget(lbl_handle)
        author_info.addWidget(badge_official)
        author_info.addStretch()

        btn_creator_gh = QPushButton(" Creator Profile")
        btn_creator_gh.setProperty("class", "secondaryBtn")
        btn_creator_gh.setFixedHeight(24)
        btn_creator_gh.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        btn_creator_gh.setIcon(get_icon("link", palette.text_secondary, palette.accent, size=11))
        btn_creator_gh.clicked.connect(lambda checked=False, url=author_url: QDesktopServices.openUrl(QUrl(url)))

        cs_layout.addWidget(avatar_lbl)
        cs_layout.addLayout(author_info, 1)
        cs_layout.addWidget(btn_creator_gh)
        c_layout.addWidget(creator_strip)

        # Description
        desc_lbl = QLabel(desc_text)
        desc_lbl.setStyleSheet(f"font-size: 11.5px; color: {palette.text_secondary}; line-height: 1.4;")
        desc_lbl.setWordWrap(True)
        c_layout.addWidget(desc_lbl)

        # Footer Actions
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)

        repo_url = f"https://github.com/rajatcoder132/{repo_slug}"
        btn_view_gh = QPushButton(" View on GitHub")
        btn_view_gh.setProperty("class", "secondaryBtn")
        btn_view_gh.setIcon(get_icon("link", palette.text_secondary, palette.accent, size=14))
        btn_view_gh.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(repo_url)))
        bottom_row.addWidget(btn_view_gh)

        bottom_row.addStretch()

        target_dir = MODELS_DIR / repo_slug
        is_installed = self._is_model_installed(target_dir)

        status_lbl = QLabel("Installed on PC" if is_installed else "Available on GitHub")
        status_lbl_color = palette.success if is_installed else palette.text_muted
        status_lbl.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {status_lbl_color};")
        bottom_row.addWidget(status_lbl)

        if is_installed:
            btn_activate = QPushButton("Activate Model")
            btn_activate.setProperty("class", "secondaryBtn")
            btn_activate.clicked.connect(lambda checked=False, td=target_dir, dn=display_name: self._activate_existing_cloud_model(td, dn))
            bottom_row.addWidget(btn_activate)

            btn_del = QPushButton(" 🗑 Delete")
            btn_del.setProperty("class", "dangerBtn")
            btn_del.setToolTip(f"Delete '{display_name}' from models folder")
            btn_del.clicked.connect(lambda checked=False, td=target_dir, dn=display_name: self._delete_cloud_model(td, dn))
            bottom_row.addWidget(btn_del)

        btn_dl = QPushButton(" Re-download" if is_installed else "Download & Install")
        btn_dl.setProperty("class", "primaryBtn" if not is_installed else "secondaryBtn")
        btn_dl.clicked.connect(lambda checked=False, rs=repo_slug, dn=display_name, td=target_dir: self._start_github_download(rs, dn, td))
        bottom_row.addWidget(btn_dl)

        c_layout.addLayout(bottom_row)
        return card

    def _delete_cloud_model(self, target_dir: Path, display_name: str):
        reply = QMessageBox.question(
            self,
            "Delete Model Package",
            f"Are you sure you want to delete model '{display_name}'?\n\n"
            f"Location: {target_dir}\n\n"
            "This will delete the model directory and unregister it from SignFlow.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel
        )
        if reply == QMessageBox.StandardButton.Yes:
            reg = ModelRegistry.get_instance()
            deleted = False
            for pkg in reg.get_all_packages():
                if Path(pkg.folder_path).resolve() == target_dir.resolve():
                    reg.delete_package(pkg.id, delete_disk_files=True)
                    deleted = True
                    break
            if not deleted and target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)

            QMessageBox.information(self, "Model Deleted", f"Model '{display_name}' has been deleted.")
            self._refresh_cloud_tab()
            self._refresh_installed_tab()

    def _is_model_installed(self, target_dir: Path) -> bool:
        if not target_dir.exists() or not target_dir.is_dir():
            return False
        res = detect_model_in_folder(str(target_dir))
        return res["is_valid"]

    def _activate_existing_cloud_model(self, target_dir: Path, display_name: str):
        reg = ModelRegistry.get_instance()
        success, pkg, msg = reg.register_model_package(str(target_dir), custom_name=display_name)
        if success and pkg:
            self.model_imported.emit(pkg)
            self.accept()
        else:
            QMessageBox.warning(self, "Activation Error", msg)

    def _start_github_download(self, repo_slug: str, display_name: str, target_dir: Path):
        if self._active_downloader and self._active_downloader.isRunning():
            QMessageBox.information(self, "Download in Progress", "A model download is already running. Please wait.")
            return

        self.progress_box.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_status.setText(f"Connecting to GitHub for {display_name}...")

        self._active_downloader = GitHubModelDownloader(repo_slug, display_name, target_dir)
        self._active_downloader.progress.connect(self._on_download_progress)
        self._active_downloader.finished.connect(self._on_download_finished)
        self._active_downloader.start()

    def _on_download_progress(self, percent: int, msg: str):
        self.progress_bar.setValue(percent)
        self.progress_status.setText(msg)

    def _on_download_finished(self, success: bool, msg: str, pkg: Optional[ModelPackage]):
        self.progress_box.setVisible(False)
        self._active_downloader = None

        if success and pkg:
            self.model_imported.emit(pkg)
            QMessageBox.information(
                self,
                "Model Installed Successfully",
                f"Successfully downloaded and registered:\n\n"
                f"Model: {pkg.name}\n"
                f"Type: {'Developer Suite' if pkg.is_developer else 'User Model'}\n"
                f"Saved to: {pkg.folder_path}"
            )
            self.accept()
        else:
            QMessageBox.warning(self, "Download / Verification Error", msg)

    def _cancel_active_download(self):
        if self._active_downloader:
            self._active_downloader.cancel()
            self.progress_status.setText("Cancelling download...")

    # -------------------------------------------------------------------------
    # TAB 2: LOCAL FOLDER IMPORT
    # -------------------------------------------------------------------------
    def _create_local_import_tab(self) -> QWidget:
        palette = ThemeManager.get_palette()
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(14)

        # Folder Picker Card
        folder_card = QFrame()
        folder_card.setProperty("class", "cardPanelElevated")
        fc_layout = QVBoxLayout(folder_card)
        fc_layout.setContentsMargins(16, 14, 16, 14)
        fc_layout.setSpacing(10)

        f_label = QLabel("LOCAL MODEL DIRECTORY PATH")
        f_label.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        fc_layout.addWidget(f_label)

        f_input_row = QHBoxLayout()
        f_input_row.setSpacing(10)

        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Select folder (e.g. C:/Users/.../SignLanguageAI)")
        self.folder_input.textChanged.connect(self._on_path_changed)
        f_input_row.addWidget(self.folder_input, 1)

        self.browse_btn = QPushButton("Browse Folder")
        self.browse_btn.setProperty("class", "secondaryBtn")
        self.browse_btn.setIcon(get_icon("folder", palette.text_secondary, palette.accent, size=15))
        self.browse_btn.clicked.connect(self._browse_folder)
        f_input_row.addWidget(self.browse_btn)

        fc_layout.addLayout(f_input_row)
        layout.addWidget(folder_card)

        # Verification Status Panel
        self.status_panel = QFrame()
        self.status_panel.setProperty("class", "cardPanel")
        self.status_panel.setVisible(False)
        sp_layout = QVBoxLayout(self.status_panel)
        sp_layout.setContentsMargins(16, 14, 16, 14)
        sp_layout.setSpacing(6)

        self.status_title = QLabel("Model Verification")
        self.status_title.setStyleSheet("font-size: 12.5px; font-weight: 700;")
        sp_layout.addWidget(self.status_title)

        self.lbl_type_badge = QLabel("—")
        self.lbl_type_badge.setStyleSheet(f"font-size: 11.5px; font-weight: 600; color: {palette.accent_label};")
        sp_layout.addWidget(self.lbl_type_badge)

        self.lbl_predict = QLabel("—")
        self.lbl_classifier = QLabel("—")
        self.lbl_task = QLabel("—")
        self.lbl_dev_suite = QLabel("—")

        for lbl in (self.lbl_predict, self.lbl_classifier, self.lbl_task, self.lbl_dev_suite):
            lbl.setStyleSheet(f"font-size: 11.5px; color: {palette.text_secondary};")
            sp_layout.addWidget(lbl)

        layout.addWidget(self.status_panel)

        # Model Custom Name Card
        name_card = QFrame()
        name_card.setProperty("class", "cardPanel")
        nc_layout = QVBoxLayout(name_card)
        nc_layout.setContentsMargins(16, 14, 16, 14)
        nc_layout.setSpacing(8)

        name_lbl = QLabel("DISPLAY NAME FOR THIS MODEL")
        name_lbl.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        nc_layout.addWidget(name_lbl)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. My Custom ASL Model")
        nc_layout.addWidget(self.name_input)
        layout.addWidget(name_card)

        # Import Button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.import_btn = QPushButton("Import & Register Local Model")
        self.import_btn.setProperty("class", "primaryBtn")
        self.import_btn.setEnabled(False)
        self.import_btn.clicked.connect(self._on_import_clicked)
        btn_row.addWidget(self.import_btn)
        layout.addLayout(btn_row)

        layout.addStretch()
        return tab_widget

    def _browse_folder(self):
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select AI Model Directory",
            str(Path.home() / "Downloads")
        )
        if chosen:
            self.folder_input.setText(chosen)

    def _on_path_changed(self, path_text: str):
        path = path_text.strip()
        if not path or not Path(path).exists() or not Path(path).is_dir():
            self.status_panel.setVisible(False)
            self.import_btn.setEnabled(False)
            self._detected_data = None
            return

        res = detect_model_in_folder(path)
        self._detected_data = res
        self.status_panel.setVisible(True)

        palette = ThemeManager.get_palette()
        if res["is_valid"]:
            self.status_title.setText("✓ Verified Model Package")
            self.status_title.setStyleSheet(f"font-size: 12.5px; font-weight: 700; color: {palette.success};")

            is_dev = res.get("is_developer", False)
            if is_dev:
                self.lbl_type_badge.setText("PACKAGE TYPE: DEVELOPER MODEL SUITE (TRAINING + INFERENCE)")
                self.lbl_type_badge.setStyleSheet(f"font-size: 11.5px; font-weight: 700; color: {palette.accent_label};")
                self.lbl_dev_suite.setText("• Developer Suite: train_model.py, collect scripts, and datasets verified!")
            else:
                self.lbl_type_badge.setText("PACKAGE TYPE: USER MODEL (INFERENCE ONLY)")
                self.lbl_type_badge.setStyleSheet(f"font-size: 11.5px; font-weight: 700; color: {palette.text_secondary};")
                self.lbl_dev_suite.setText("• Standalone User Model (no training files detected)")

            p_txt = Path(res['predict_script']).name
            self.lbl_predict.setText(f"• Predict Script: {p_txt} (✓ Found)")
            self.lbl_classifier.setText(f"• Classifier: {Path(res['classifier_path']).name} (✓ Found)")
            self.lbl_task.setText(f"• Landmarker: {Path(res['task_path']).name} (✓ Found)")

            if not self.name_input.text():
                self.name_input.setText(res["suggested_name"])

            self.import_btn.setEnabled(True)
        else:
            self.status_title.setText("✗ Incomplete Model Directory")
            self.status_title.setStyleSheet(f"font-size: 12.5px; font-weight: 700; color: {palette.danger};")
            self.lbl_type_badge.setText(f"Error: {res['error_msg']}")
            self.lbl_type_badge.setStyleSheet(f"font-size: 11.5px; color: {palette.danger};")
            self.lbl_predict.setText("Required structure:")
            self.lbl_classifier.setText("  - predict_live.py (in root)")
            self.lbl_task.setText("  - models/asl_classifier.joblib & models/hand_landmarker.task")
            self.lbl_dev_suite.setText("")
            self.import_btn.setEnabled(False)

    def _on_import_clicked(self):
        if not self._detected_data or not self._detected_data["is_valid"]:
            return

        reg = ModelRegistry.get_instance()
        custom_name = self.name_input.text().strip() or self._detected_data["suggested_name"]
        success, pkg, msg = reg.register_model_package(
            folder_path=self._detected_data["folder_path"],
            custom_name=custom_name
        )

        if success and pkg:
            self.model_imported.emit(pkg)
            self.accept()
        else:
            QMessageBox.warning(self, "Registration Error", msg)

    # -------------------------------------------------------------------------
    # TAB 3: INSTALLED MODELS (MANAGE & DELETE)
    # -------------------------------------------------------------------------
    def _create_installed_tab(self) -> QWidget:
        palette = ThemeManager.get_palette()
        self.installed_tab_widget = QWidget()
        layout = QVBoxLayout(self.installed_tab_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        intro_lbl = QLabel(
            "Manage your registered model packages. You can activate any model for live recognition or "
            "delete unwanted models directly using the delete button beside each model."
        )
        intro_lbl.setStyleSheet(f"font-size: 11.5px; color: {palette.text_secondary};")
        intro_lbl.setWordWrap(True)
        layout.addWidget(intro_lbl)

        self.installed_scroll = QScrollArea()
        self.installed_scroll.setWidgetResizable(True)
        self.installed_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.installed_container = QWidget()
        self.installed_container_layout = QVBoxLayout(self.installed_container)
        self.installed_container_layout.setContentsMargins(0, 0, 0, 0)
        self.installed_container_layout.setSpacing(10)

        self.installed_scroll.setWidget(self.installed_container)
        layout.addWidget(self.installed_scroll, 1)

        self._refresh_installed_tab()
        return self.installed_tab_widget

    def _refresh_installed_tab(self):
        if not hasattr(self, 'installed_container_layout'):
            return

        while self.installed_container_layout.count():
            item = self.installed_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        palette = ThemeManager.get_palette()
        reg = ModelRegistry.get_instance()
        packages = reg.get_all_packages()
        active = reg.get_active_package()

        if not packages:
            empty_box = QFrame()
            empty_box.setProperty("class", "cardPanel")
            eb_layout = QVBoxLayout(empty_box)
            eb_layout.setContentsMargins(20, 28, 20, 28)
            eb_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl = QLabel("No models currently installed or registered.")
            lbl.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {palette.text_muted};")
            lbl_sub = QLabel("Download an official model from the 'Official GitHub Models' tab or import a local folder.")
            lbl_sub.setStyleSheet(f"font-size: 11.5px; color: {palette.text_secondary};")
            eb_layout.addWidget(lbl, alignment=Qt.AlignmentFlag.AlignCenter)
            eb_layout.addWidget(lbl_sub, alignment=Qt.AlignmentFlag.AlignCenter)
            self.installed_container_layout.addWidget(empty_box)
            self.installed_container_layout.addStretch()
            return

        for pkg in packages:
            card = QFrame()
            card.setProperty("class", "cardPanel")
            c_layout = QHBoxLayout(card)
            c_layout.setContentsMargins(14, 12, 14, 12)
            c_layout.setSpacing(12)

            info_vbox = QVBoxLayout()
            info_vbox.setSpacing(4)

            title_row = QHBoxLayout()
            title_row.setSpacing(8)

            lbl_name = QLabel(f"📦 {pkg.name}")
            lbl_name.setStyleSheet(f"font-size: 13.5px; font-weight: 700; color: {palette.text_primary};")
            title_row.addWidget(lbl_name)

            badge_text = "DEVELOPER SUITE" if pkg.is_developer else "USER MODEL"
            badge_color = palette.success if pkg.is_developer else palette.accent_label
            badge = QLabel(badge_text)
            badge.setStyleSheet(f"""
                font-size: 9px;
                font-weight: 700;
                padding: 2px 6px;
                border-radius: 4px;
                background-color: {palette.bg_sidebar};
                color: {badge_color};
                border: 1px solid {palette.border_subtle};
            """)
            title_row.addWidget(badge)

            is_active = (active and active.id == pkg.id)
            if is_active:
                active_badge = QLabel("✓ ACTIVE MODEL")
                active_badge.setStyleSheet(f"""
                    font-size: 9px;
                    font-weight: 800;
                    padding: 2px 6px;
                    border-radius: 4px;
                    background-color: {palette.bg_sidebar};
                    color: {palette.success};
                    border: 1px solid {palette.success};
                """)
                title_row.addWidget(active_badge)

            title_row.addStretch()
            info_vbox.addLayout(title_row)

            # Creator Attribution on each installed model card
            is_signflow_official = ("signflow" in pkg.name.lower() or "rajatcoder132" in pkg.folder_path.lower() or "SignFlow" in pkg.folder_path)
            creator_lbl = QLabel(
                "Creator: Rajat (@rajatcoder132) • Official SignFlow Model" if is_signflow_official else "Creator: Custom Local Model"
            )
            creator_lbl.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {palette.accent_label if is_signflow_official else palette.text_secondary};")
            info_vbox.addWidget(creator_lbl)

            path_lbl = QLabel(f"Path: {pkg.folder_path}")
            path_lbl.setStyleSheet(f"font-size: 11px; color: {palette.text_muted};")
            info_vbox.addWidget(path_lbl)

            c_layout.addLayout(info_vbox, 1)

            # Action buttons beside each model
            if not is_active:
                btn_act = QPushButton("Activate")
                btn_act.setProperty("class", "secondaryBtn")
                btn_act.clicked.connect(lambda checked=False, p=pkg: self._activate_installed_pkg(p))
                c_layout.addWidget(btn_act)

            btn_del = QPushButton(" 🗑 Delete")
            btn_del.setProperty("class", "dangerBtn")
            btn_del.setToolTip(f"Delete model '{pkg.name}'")
            btn_del.clicked.connect(lambda checked=False, p=pkg: self._delete_installed_pkg(p))
            c_layout.addWidget(btn_del)

            self.installed_container_layout.addWidget(card)

        self.installed_container_layout.addStretch()

    def _activate_installed_pkg(self, pkg: ModelPackage):
        reg = ModelRegistry.get_instance()
        reg.set_active_package(pkg.id)
        self.model_imported.emit(pkg)
        self._refresh_installed_tab()
        self.accept()

    def _delete_installed_pkg(self, pkg: ModelPackage):
        folder = Path(pkg.folder_path)
        is_under_models = (MODELS_DIR.resolve() in folder.resolve().parents) or (folder.resolve() == MODELS_DIR.resolve()) or (folder.parent.resolve() == MODELS_DIR.resolve())

        prompt = (
            f"Are you sure you want to delete model package '{pkg.name}'?\n\n"
            f"Folder: {pkg.folder_path}\n\n"
            + ("This will permanently delete the model files and unregister it."
               if is_under_models else "This will unregister the model from SignFlow.")
        )
        reply = QMessageBox.question(
            self,
            "Delete Model Package",
            prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel
        )
        if reply == QMessageBox.StandardButton.Yes:
            reg = ModelRegistry.get_instance()
            reg.delete_package(pkg.id, delete_disk_files=is_under_models)
            QMessageBox.information(self, "Model Deleted", f"'{pkg.name}' was removed successfully.")
            self._refresh_installed_tab()
            self._refresh_cloud_tab()
