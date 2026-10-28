"""
Theme management, color tokens, and comprehensive QSS stylesheets.
Inspired by Linear and modern developer tools: minimalist, refined, high-contrast.
"""

class ThemePalette:
    def __init__(self, is_dark: bool = True):
        self.is_dark = is_dark
        if is_dark:
            self.bg_window = "#0D0F14"
            self.bg_sidebar = "#11141B"
            self.bg_card = "#161922"
            self.bg_card_elevated = "#1D212D"
            self.bg_input = "#141720"
            self.bg_hover = "#202534"
            self.bg_active = "#272E40"
            
            self.border_subtle = "#222735"
            self.border_default = "#2C3345"
            self.border_active = "#5E6AD2"
            
            self.text_primary = "#F8FAFC"
            self.text_secondary = "#94A3B8"
            self.text_muted = "#64748B"
            self.text_inverse = "#0D0F14"
            
            self.accent = "#5E6AD2"
            self.accent_hover = "#6E7BF2"
            self.accent_pressed = "#4F59B5"
            self.accent_subtle = "rgba(94, 106, 210, 0.15)"
            
            self.success = "#10B981"
            self.success_subtle = "rgba(16, 185, 129, 0.14)"
            self.warning = "#F59E0B"
            self.warning_subtle = "rgba(245, 158, 11, 0.14)"
            self.danger = "#EF4444"
            self.danger_subtle = "rgba(239, 68, 68, 0.14)"
            
            self.video_bg = "#07080B"
            self.code_bg = "#0F1218"
            self.scrollbar_thumb = "#2C3345"
            self.scrollbar_thumb_hover = "#404B63"
        else:
            self.bg_window = "#F8FAFC"
            self.bg_sidebar = "#FFFFFF"
            self.bg_card = "#FFFFFF"
            self.bg_card_elevated = "#F1F5F9"
            self.bg_input = "#FFFFFF"
            self.bg_hover = "#F1F5F9"
            self.bg_active = "#E2E8F0"
            
            self.border_subtle = "#E2E8F0"
            self.border_default = "#CBD5E1"
            self.border_active = "#4F46E5"
            
            self.text_primary = "#0F172A"
            self.text_secondary = "#475569"
            self.text_muted = "#94A3B8"
            self.text_inverse = "#FFFFFF"
            
            self.accent = "#4F46E5"
            self.accent_hover = "#4338CA"
            self.accent_pressed = "#3730A3"
            self.accent_subtle = "rgba(79, 70, 229, 0.10)"
            
            self.success = "#059669"
            self.success_subtle = "rgba(5, 150, 105, 0.10)"
            self.warning = "#D97706"
            self.warning_subtle = "rgba(217, 119, 6, 0.10)"
            self.danger = "#DC2626"
            self.danger_subtle = "rgba(220, 38, 38, 0.10)"
            
            self.video_bg = "#0F172A"
            self.code_bg = "#F8FAFC"
            self.scrollbar_thumb = "#CBD5E1"
            self.scrollbar_thumb_hover = "#94A3B8"

    def generate_qss(self) -> str:
        p = self
        return f"""
        /* Global Reset & Typography */
        * {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 13px;
            color: {p.text_primary};
        }}

        QMainWindow, QDialog {{
            background-color: {p.bg_window};
        }}

        QWidget#centralWidget {{
            background-color: {p.bg_window};
        }}

        /* Sidebar Styling */
        QWidget#sidebarContainer {{
            background-color: {p.bg_sidebar};
            border-right: 1px solid {p.border_subtle};
        }}

        /* Top Bar Styling */
        QWidget#topBarContainer {{
            background-color: {p.bg_sidebar};
            border-bottom: 1px solid {p.border_subtle};
        }}

        /* Content Area */
        QScrollArea#contentScrollArea {{
            background-color: {p.bg_window};
            border: none;
        }}
        QWidget#contentContainer {{
            background-color: {p.bg_window};
        }}

        /* Card Panels */
        QFrame.cardPanel {{
            background-color: {p.bg_card};
            border: 1px solid {p.border_subtle};
            border-radius: 8px;
        }}
        QFrame.cardPanelElevated {{
            background-color: {p.bg_card_elevated};
            border: 1px solid {p.border_default};
            border-radius: 8px;
        }}

        /* Navigation Buttons */
        QPushButton.navBtn {{
            text-align: left;
            padding: 8px 12px;
            border: none;
            border-radius: 6px;
            background-color: transparent;
            color: {p.text_secondary};
            font-weight: 500;
            font-size: 13px;
        }}
        QPushButton.navBtn:hover {{
            background-color: {p.bg_hover};
            color: {p.text_primary};
        }}
        QPushButton.navBtn:checked, QPushButton.navBtn.active {{
            background-color: {p.accent_subtle};
            color: {p.accent};
            font-weight: 600;
        }}

        /* Primary Action Nav Button (Predict) */
        QPushButton.navBtnPredict {{
            text-align: left;
            padding: 9px 12px;
            border: 1px solid {p.border_subtle};
            border-radius: 6px;
            background-color: {p.bg_card};
            color: {p.text_primary};
            font-weight: 600;
            font-size: 13px;
        }}
        QPushButton.navBtnPredict:hover {{
            background-color: {p.bg_hover};
            border-color: {p.accent};
            color: {p.accent};
        }}
        QPushButton.navBtnPredict:checked, QPushButton.navBtnPredict.active {{
            background-color: {p.accent_subtle};
            border-color: {p.accent};
            color: {p.accent};
        }}

        /* General Buttons */
        QPushButton.primaryBtn {{
            background-color: {p.accent};
            color: #FFFFFF;
            border: none;
            border-radius: 6px;
            padding: 7px 16px;
            font-weight: 600;
            font-size: 13px;
        }}
        QPushButton.primaryBtn:hover {{
            background-color: {p.accent_hover};
        }}
        QPushButton.primaryBtn:pressed {{
            background-color: {p.accent_pressed};
        }}
        QPushButton.primaryBtn:disabled {{
            background-color: {p.border_subtle};
            color: {p.text_muted};
        }}

        QPushButton.secondaryBtn {{
            background-color: {p.bg_card};
            color: {p.text_primary};
            border: 1px solid {p.border_default};
            border-radius: 6px;
            padding: 6px 14px;
            font-weight: 500;
            font-size: 13px;
        }}
        QPushButton.secondaryBtn:hover {{
            background-color: {p.bg_hover};
            border-color: {p.text_secondary};
        }}
        QPushButton.secondaryBtn:pressed {{
            background-color: {p.bg_active};
        }}
        QPushButton.secondaryBtn:disabled {{
            background-color: transparent;
            border-color: {p.border_subtle};
            color: {p.text_muted};
        }}

        QPushButton.subtleBtn {{
            background-color: transparent;
            color: {p.text_secondary};
            border: none;
            border-radius: 6px;
            padding: 6px 10px;
        }}
        QPushButton.subtleBtn:hover {{
            background-color: {p.bg_hover};
            color: {p.text_primary};
        }}

        QPushButton.dangerBtn {{
            background-color: {p.danger_subtle};
            color: {p.danger};
            border: 1px solid {p.danger};
            border-radius: 6px;
            padding: 6px 14px;
            font-weight: 500;
        }}
        QPushButton.dangerBtn:hover {{
            background-color: {p.danger};
            color: #FFFFFF;
        }}

        /* Inputs & Combos */
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background-color: {p.bg_input};
            border: 1px solid {p.border_default};
            border-radius: 6px;
            padding: 6px 10px;
            color: {p.text_primary};
            selection-background-color: {p.accent};
            selection-color: #FFFFFF;
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px solid {p.accent};
        }}
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{
            background-color: {p.bg_hover};
            color: {p.text_muted};
            border-color: {p.border_subtle};
        }}

        QComboBox {{
            padding-right: 24px;
        }}
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 22px;
            border-left: none;
        }}
        QComboBox QAbstractItemView {{
            background-color: {p.bg_card};
            border: 1px solid {p.border_default};
            border-radius: 6px;
            selection-background-color: {p.accent_subtle};
            selection-color: {p.accent};
            color: {p.text_primary};
            padding: 4px;
            outline: none;
        }}
        QComboBox QAbstractItemView::item {{
            padding: 6px 10px;
            border-radius: 4px;
            min-height: 22px;
        }}
        QComboBox QAbstractItemView::item:hover {{
            background-color: {p.bg_hover};
        }}

        /* Sliders */
        QSlider::groove:horizontal {{
            height: 4px;
            background: {p.border_default};
            border-radius: 2px;
        }}
        QSlider::sub-page:horizontal {{
            background: {p.accent};
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {p.text_primary};
            border: 2px solid {p.accent};
            width: 14px;
            margin-top: -5px;
            margin-bottom: -5px;
            border-radius: 7px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {p.accent_hover};
        }}

        /* Progress Bar */
        QProgressBar {{
            background-color: {p.bg_hover};
            border: 1px solid {p.border_subtle};
            border-radius: 4px;
            height: 8px;
            text-align: right;
        }}
        QProgressBar::chunk {{
            background-color: {p.accent};
            border-radius: 3px;
        }}

        /* Monospace / Code Text Areas */
        QTextEdit.consoleText, QPlainTextEdit.consoleText {{
            background-color: {p.code_bg};
            border: 1px solid {p.border_subtle};
            border-radius: 6px;
            font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
            font-size: 12px;
            color: {p.text_secondary};
            padding: 8px;
            line-height: 1.4;
        }}

        /* Scrollbars (Modern Thin Slender) */
        QScrollBar:vertical {{
            border: none;
            background: transparent;
            width: 7px;
            margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background: {p.scrollbar_thumb};
            min-height: 24px;
            border-radius: 3px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {p.scrollbar_thumb_hover};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: none;
        }}

        QScrollBar:horizontal {{
            border: none;
            background: transparent;
            height: 7px;
            margin: 0px;
        }}
        QScrollBar::handle:horizontal {{
            background: {p.scrollbar_thumb};
            min-width: 24px;
            border-radius: 3px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {p.scrollbar_thumb_hover};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            background: none;
        }}

        /* Badges & Status */
        QLabel.badge {{
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
        }}
        QLabel.badgeNeutral {{
            background-color: {p.bg_hover};
            color: {p.text_secondary};
            border: 1px solid {p.border_subtle};
        }}
        QLabel.badgeActive {{
            background-color: {p.success_subtle};
            color: {p.success};
            border: 1px solid {p.success};
        }}
        QLabel.badgeWarning {{
            background-color: {p.warning_subtle};
            color: {p.warning};
            border: 1px solid {p.warning};
        }}
        QLabel.badgeAccent {{
            background-color: {p.accent_subtle};
            color: {p.accent};
            border: 1px solid {p.accent};
        }}

        /* Checkbox */
        QCheckBox {{
            spacing: 8px;
            color: {p.text_primary};
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {p.border_default};
            border-radius: 4px;
            background-color: {p.bg_input};
        }}
        QCheckBox::indicator:hover {{
            border-color: {p.accent};
        }}
        QCheckBox::indicator:checked {{
            background-color: {p.accent};
            border-color: {p.accent};
        }}
        """

class ThemeManager:
    """Singleton-like manager for dynamic theme updates."""
    _current_palette: ThemePalette = None

    @classmethod
    def get_palette(cls) -> ThemePalette:
        if cls._current_palette is None:
            cls._current_palette = ThemePalette(is_dark=True)
        return cls._current_palette

    @classmethod
    def apply_theme(cls, app, is_dark: bool):
        cls._current_palette = ThemePalette(is_dark=is_dark)
        qss = cls._current_palette.generate_qss()
        app.setStyleSheet(qss)
