"""
Connect Apps Modal Dialog.
Clean visual placeholder for future live streaming, meeting, and OS integrations.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QFrame, QScrollArea, QWidget, QMessageBox
)
from PySide6.QtCore import Qt

from app.resources.icons import get_icon, get_pixmap
from app.core.theme import ThemeManager


class IntegrationCard(QFrame):
    """Card representing an app integration target."""
    def __init__(self, title: str, category: str, description: str, icon_name: str, parent=None):
        super().__init__(parent)
        self.setProperty("class", "cardPanel")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Top row: icon + title + category badge
        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        palette = ThemeManager.get_palette()
        icon_lbl = QLabel()
        icon_lbl.setPixmap(get_pixmap(icon_name, palette.accent, size=24))

        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        
        name_lbl = QLabel(title)
        name_lbl.setStyleSheet("font-size: 14px; font-weight: 600;")
        
        cat_lbl = QLabel(category.upper())
        cat_lbl.setStyleSheet(f"font-size: 10px; color: {palette.text_muted}; font-weight: 600; letter-spacing: 0.5px;")

        title_layout.addWidget(name_lbl)
        title_layout.addWidget(cat_lbl)

        top_row.addWidget(icon_lbl)
        top_row.addLayout(title_layout)
        top_row.addStretch()

        # Description
        desc_lbl = QLabel(description)
        desc_lbl.setStyleSheet(f"font-size: 12px; color: {palette.text_secondary};")
        desc_lbl.setWordWrap(True)

        # Bottom row: Action
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        
        self.config_btn = QPushButton("Configure")
        self.config_btn.setProperty("class", "secondaryBtn")
        self.config_btn.clicked.connect(self._on_click)
        btn_row.addWidget(self.config_btn)

        layout.addLayout(top_row)
        layout.addWidget(desc_lbl)
        layout.addStretch()
        layout.addLayout(btn_row)

    def _on_click(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Integration Connector")
        msg.setText("Integration Connector Preview")
        msg.setInformativeText(
            "This integration slot is a frontend architectural placeholder. "
            "Backend pipelines can bridge into this endpoint once external SDKs are configured."
        )
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()


class ConnectAppsDialog(QDialog):
    """Professional dialog modal displaying ecosystem integrations."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connected Applications & Services")
        self.setMinimumSize(780, 560)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        palette = ThemeManager.get_palette()

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(20)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        
        title_lbl = QLabel("Connected Applications & Bridges")
        title_lbl.setStyleSheet("font-size: 20px; font-weight: 700;")
        
        subtitle_lbl = QLabel(
            "Link real-time sign recognition output directly to your virtual cameras, conferencing feeds, and desktop applications."
        )
        subtitle_lbl.setStyleSheet(f"font-size: 13px; color: {palette.text_secondary};")
        subtitle_lbl.setWordWrap(True)

        header_layout.addWidget(title_lbl)
        header_layout.addWidget(subtitle_lbl)
        main_layout.addLayout(header_layout)

        # Integrations Grid Container inside ScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName("contentScrollArea")

        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(14)
        grid.setContentsMargins(0, 0, 0, 0)

        apps = [
            ("Zoom Video Meetings", "Conferencing", "Broadcast real-time ASL closed captions and visual subtitles directly into Zoom video calls.", "camera"),
            ("Microsoft Teams", "Enterprise", "Inject live sign interpretation subtitles directly into corporate meetings and sidecar channels.", "apps"),
            ("Google Meet", "Web Meetings", "Direct browser pipe feeding recognized phrases into Google Meet closed caption streams.", "overview"),
            ("OBS Studio", "Broadcasting", "Direct Alpha-channel lower-third subtitle overlay and virtual camera pipeline.", "sliders"),
            ("Discord Accessibility", "Community", "Live sign stream commentary bot and real-time text-to-speech bridge.", "dynamic_training"),
            ("Windows Virtual Input", "System Utility", "Emulate real-time keyboard strokes to type recognized letters directly into any active Windows application.", "terminal"),
        ]

        for i, (name, cat, desc, icon) in enumerate(apps):
            card = IntegrationCard(name, cat, desc, icon, self)
            grid.addWidget(card, i // 2, i % 2)

        scroll.setWidget(grid_widget)
        main_layout.addWidget(scroll)

        # Footer Notice & Close Button
        footer = QHBoxLayout()
        footer.setSpacing(12)

        notice_lbl = QLabel("Interface preview: No external credentials or background cloud services required.")
        notice_lbl.setStyleSheet(f"font-size: 11px; color: {palette.text_muted}; font-style: italic;")

        close_btn = QPushButton("Close")
        close_btn.setProperty("class", "secondaryBtn")
        close_btn.clicked.connect(self.accept)

        footer.addWidget(notice_lbl)
        footer.addStretch()
        footer.addWidget(close_btn)
        main_layout.addLayout(footer)
