"""
Theme management, color tokens, and comprehensive tactile QSS stylesheets for SignFlow Studio.
Features an editorial aesthetic:
- Light Mode: Warm Cream (#FAF7F0), Soft Butter Yellow (#FEF6D8 / #F3D580), Warm Oat (#F3EEE4), Deep Espresso (#1C1917).
- Dark Mode: Deep Warm Charcoal (#161513), Warm Dark Slate (#262320), Espresso Surface (#1E1C19), Amber Gold (#F5B419 / #FCD34D).
"""

class ThemePalette:
    def __init__(self, is_dark: bool = False):
        self.is_dark = is_dark
        if is_dark:
            # Dark Mode Specifications
            self.bg_window = "#161513"         # Deep warm charcoal canvas
            self.bg_sidebar = "#1E1C19"        # Espresso surface
            self.bg_card = "#262320"           # Warm dark slate elevated cards
            self.bg_card_elevated = "#2E2A26"  # Slightly elevated card
            self.bg_input = "#1E1C19"          # Input background
            self.bg_hover = "#2C2824"          # Subtle hover state
            self.bg_active = "#382C17"         # Dark amber tint active state

            self.border_subtle = "#332F2A"     # Subtle dividers
            self.border_default = "#3D3833"    # 1px border
            self.border_active = "#7A5E2A"     # Border for active amber elements
            self.border_focus = "#F5B419"      # Warm golden amber focus

            self.text_primary = "#FDFBF7"      # Warm cream white headers & readouts
            self.text_secondary = "#D6D0C5"    # Secondary text
            self.text_muted = "#8F887F"        # Muted metadata
            self.text_inverse = "#161513"

            self.accent = "#F5B419"            # Warm golden amber
            self.accent_hover = "#FCD34D"      # Bright golden highlight
            self.accent_pressed = "#D9820F"    # Deep amber pressed
            self.accent_subtle = "#382C17"     # Dark amber background tint
            self.accent_border = "#7A5E2A"     # Dark amber border
            self.accent_label = "#FCD34D"      # High contrast amber label

            self.success = "#34D399"
            self.success_subtle = "rgba(52, 211, 153, 0.16)"
            self.warning = "#F59E0B"
            self.warning_subtle = "rgba(245, 158, 11, 0.16)"
            self.danger = "#F87171"
            self.danger_subtle = "rgba(248, 113, 113, 0.16)"

            self.video_bg = "#0E0D0C"
            self.code_bg = "#191715"
            self.scrollbar_thumb = "#3D3833"
            self.scrollbar_thumb_hover = "#524B44"
        else:
            # Light Mode Specifications
            self.bg_window = "#FAF7F0"         # Warm cream / alabaster canvas
            self.bg_sidebar = "#F3EEE4"        # Warm oat sidebars & panels
            self.bg_card = "#FFFFFF"           # Elevated crisp cards
            self.bg_card_elevated = "#F8F5EE"  # Subtle warm elevated surface
            self.bg_input = "#FFFFFF"          # Input surface
            self.bg_hover = "#EFE9DD"          # Warm oat hover
            self.bg_active = "#FEF6D8"         # Soft butter yellow active highlight

            self.border_subtle = "#EEE9DF"     # Hairlines & subtle dividers
            self.border_default = "#E4DDD1"    # Warm stone 1px borders
            self.border_active = "#F3D580"     # Soft butter yellow border
            self.border_focus = "#F5B419"      # Warm gold focus ring

            self.text_primary = "#1C1917"      # Deep espresso charcoal headers
            self.text_secondary = "#57534E"    # Warm slate subtext
            self.text_muted = "#8A837A"        # Muted tactile details
            self.text_inverse = "#FAF7F0"

            self.accent = "#F5B419"            # Warm gold
            self.accent_hover = "#E5A30D"      # Hover gold
            self.accent_pressed = "#D9820F"    # Deep amber pressed
            self.accent_subtle = "#FEF6D8"     # Soft butter yellow fill
            self.accent_border = "#F3D580"     # Soft butter yellow border
            self.accent_label = "#D9820F"      # High-contrast deep amber label

            self.success = "#15803D"
            self.success_subtle = "#DCFCE7"
            self.warning = "#B45309"
            self.warning_subtle = "#FEF3C7"
            self.danger = "#B91C1C"
            self.danger_subtle = "#FEE2E2"

            self.video_bg = "#1C1917"
            self.code_bg = "#F3EEE4"
            self.scrollbar_thumb = "#D6CFC4"
            self.scrollbar_thumb_hover = "#BDB4A6"

    def generate_qss(self) -> str:
        p = self
        btn_text_color = "#1C1917" if not self.is_dark else "#161513"
        return f"""
        /* Global Reset & Editorial Typography */
        * {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", Roboto, "Helvetica Neue", sans-serif;
            font-size: 13px;
            color: {p.text_primary};
        }}

        QMainWindow, QDialog {{
            background-color: {p.bg_window};
        }}

        QWidget#centralWidget {{
            background-color: {p.bg_window};
        }}

        /* Sidebar Styling (Warm Oat / Espresso Surface) */
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

        /* Editorial Card Panels */
        QFrame.cardPanel {{
            background-color: {p.bg_card};
            border: 1px solid {p.border_default};
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
            padding: 9px 14px;
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
            border: 1px solid {p.accent_border};
            color: {p.text_primary};
            font-weight: 600;
        }}

        /* Primary Action Nav Button (Translate) */
        QPushButton.navBtnPredict {{
            text-align: left;
            padding: 10px 14px;
            border: 1px solid {p.accent_border};
            border-radius: 6px;
            background-color: {p.accent_subtle};
            color: {p.text_primary};
            font-weight: 700;
            font-size: 13px;
        }}
        QPushButton.navBtnPredict:hover {{
            background-color: {p.accent};
            color: {btn_text_color};
            border-color: {p.accent_hover};
        }}
        QPushButton.navBtnPredict:checked, QPushButton.navBtnPredict.active {{
            background-color: {p.accent};
            color: {btn_text_color};
            border-color: {p.accent_pressed};
        }}

        /* Tactile Warm Accent Buttons */
        QPushButton.primaryBtn {{
            background-color: {p.accent};
            color: {btn_text_color};
            border: 1px solid {p.accent_pressed};
            border-radius: 6px;
            padding: 8px 18px;
            font-weight: 700;
            font-size: 13px;
        }}
        QPushButton.primaryBtn:hover {{
            background-color: {p.accent_hover};
        }}
        QPushButton.primaryBtn:pressed {{
            background-color: {p.accent_pressed};
            color: #FFFFFF;
        }}
        QPushButton.primaryBtn:disabled {{
            background-color: {p.bg_hover};
            color: {p.text_muted};
            border-color: {p.border_subtle};
        }}

        QPushButton.secondaryBtn {{
            background-color: {p.bg_card};
            color: {p.text_primary};
            border: 1px solid {p.border_default};
            border-radius: 6px;
            padding: 7px 15px;
            font-weight: 500;
            font-size: 13px;
        }}
        QPushButton.secondaryBtn:hover {{
            background-color: {p.bg_hover};
            border-color: {p.text_secondary};
        }}
        QPushButton.secondaryBtn:pressed {{
            background-color: {p.bg_active};
            border-color: {p.accent_border};
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
            padding: 6px 12px;
            font-weight: 500;
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
            padding: 7px 15px;
            font-weight: 600;
        }}
        QPushButton.dangerBtn:hover {{
            background-color: {p.danger};
            color: #FFFFFF;
        }}

        /* Inputs, Text Boxes & Comboboxes */
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background-color: {p.bg_input};
            border: 1px solid {p.border_default};
            border-radius: 6px;
            padding: 7px 12px;
            color: {p.text_primary};
            selection-background-color: {p.accent};
            selection-color: {btn_text_color};
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px solid {p.border_focus};
        }}
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{
            background-color: {p.bg_hover};
            color: {p.text_muted};
            border-color: {p.border_subtle};
        }}

        QComboBox {{
            padding-right: 28px;
        }}
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 24px;
            border-left: none;
        }}
        QComboBox QAbstractItemView {{
            background-color: {p.bg_card};
            border: 1px solid {p.border_default};
            border-radius: 6px;
            selection-background-color: {p.accent_subtle};
            selection-color: {p.text_primary};
            color: {p.text_primary};
            padding: 4px;
            outline: none;
        }}
        QComboBox QAbstractItemView::item {{
            padding: 7px 12px;
            border-radius: 4px;
            min-height: 24px;
        }}
        QComboBox QAbstractItemView::item:hover {{
            background-color: {p.bg_hover};
        }}
        QComboBox QAbstractItemView::item:selected {{
            background-color: {p.accent_subtle};
            color: {p.text_primary};
            font-weight: 600;
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
            background: {p.bg_card};
            border: 2px solid {p.accent};
            width: 14px;
            margin-top: -5px;
            margin-bottom: -5px;
            border-radius: 7px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {p.accent};
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
            padding: 10px;
            line-height: 1.5;
        }}

        /* Thin Slender Tactile Scrollbars */
        QScrollBar:vertical {{
            border: none;
            background: transparent;
            width: 8px;
            margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background: {p.scrollbar_thumb};
            min-height: 24px;
            border-radius: 4px;
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
            height: 8px;
            margin: 0px;
        }}
        QScrollBar::handle:horizontal {{
            background: {p.scrollbar_thumb};
            min-width: 24px;
            border-radius: 4px;
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

        /* Badges & Status Pills */
        QLabel.badge {{
            padding: 3px 9px;
            border-radius: 12px;
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
            color: {p.accent_label};
            border: 1px solid {p.accent_border};
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
            border-color: {p.border_focus};
        }}
        QCheckBox::indicator:checked {{
            background-color: {p.accent};
            border-color: {p.accent_pressed};
        }}
        """


class ThemeManager:
    """Singleton-like manager for dynamic theme updates."""
    _current_palette: ThemePalette = None

    @classmethod
    def get_palette(cls) -> ThemePalette:
        if cls._current_palette is None:
            cls._current_palette = ThemePalette(is_dark=False)
        return cls._current_palette

    @classmethod
    def apply_theme(cls, app, is_dark: bool):
        cls._current_palette = ThemePalette(is_dark=is_dark)
        qss = cls._current_palette.generate_qss()
        app.setStyleSheet(qss)
