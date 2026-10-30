"""
About the Project Page.
Editorial overview panel presenting SignFlow Studio and the open-source mission.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, Signal

from app.core.theme import ThemeManager


class EditorialSeparator(QFrame):
    """Subtle, understated horizontal divider rule."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Plain)
        palette = ThemeManager.get_palette()
        self.setStyleSheet(f"""
            QFrame {{
                border: none;
                height: 1px;
                background-color: {palette.border_subtle};
                margin-top: 6px;
                margin-bottom: 6px;
            }}
        """)


class OverviewPage(QWidget):
    """
    Editorial information panel for SignFlow Studio.
    Presents the creators, purpose, project architecture, and real-world impact.
    """
    navigate_to = Signal(int)

    def __init__(self, camera_manager=None, parent=None):
        super().__init__(parent)
        self._camera_manager = camera_manager

        palette = ThemeManager.get_palette()

        # Responsive Scroll Area
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName("contentScrollArea")

        # Outer viewport container to center content with restrained width
        outer_container = QWidget()
        outer_container.setObjectName("contentContainer")
        outer_layout = QHBoxLayout(outer_container)
        outer_layout.setContentsMargins(40, 36, 40, 48)
        outer_layout.setSpacing(0)

        # Restrained editorial column (max width 780px for optimal reading measure)
        content_column = QWidget()
        content_column.setMaximumWidth(780)
        content_column.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(content_column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(28)

        # =====================================================================
        # 1. MAIN HEADING & SUBTITLE
        # =====================================================================
        header_block = QVBoxLayout()
        header_block.setSpacing(10)

        main_heading = QLabel("Technology that understands a gesture.")
        main_heading.setStyleSheet(f"""
            font-size: 32px;
            font-weight: 800;
            color: {palette.text_primary};
            letter-spacing: -0.6px;
            line-height: 1.2;
        """)
        main_heading.setWordWrap(True)

        subtitle = QLabel("An exploration of sign language recognition through computer vision and machine learning.")
        subtitle.setStyleSheet(f"""
            font-size: 16px;
            font-style: italic;
            color: {palette.text_secondary};
            line-height: 1.45;
        """)
        subtitle.setWordWrap(True)

        header_block.addWidget(main_heading)
        header_block.addWidget(subtitle)
        layout.addLayout(header_block)

        layout.addWidget(EditorialSeparator())

        # =====================================================================
        # 2. MEET THE CREATORS
        # =====================================================================
        creators_block = QVBoxLayout()
        creators_block.setSpacing(6)

        creators_section_lbl = QLabel("MEET THE CREATORS")
        creators_section_lbl.setStyleSheet(f"""
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.8px;
            color: {palette.text_muted};
        """)

        creators_names = QLabel("Ayush Pandey & Rajat")
        creators_names.setStyleSheet(f"""
            font-size: 19px;
            font-weight: 700;
            color: {palette.text_primary};
            letter-spacing: -0.2px;
        """)

        creators_school = QLabel("Student Developers & AI Enthusiasts\nOpen-Source Accessibility Project")
        creators_school.setStyleSheet(f"""
            font-size: 14px;
            color: {palette.text_secondary};
            line-height: 1.4;
        """)

        creators_block.addWidget(creators_section_lbl)
        creators_block.addWidget(creators_names)
        creators_block.addWidget(creators_school)
        layout.addLayout(creators_block)

        layout.addWidget(EditorialSeparator())

        # =====================================================================
        # 3. 01 — WHO ARE WE?
        # =====================================================================
        q1_block = QVBoxLayout()
        q1_block.setSpacing(12)

        q1_title = QLabel("01 — Who are we?")
        q1_title.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 700;
            color: {palette.text_primary};
            letter-spacing: -0.2px;
        """)

        q1_p1 = QLabel(
            "We are two friends with a shared interest in technology and computer vision. Although we're still "
            "learning the fundamentals of machine learning, we wanted to challenge ourselves by building a "
            "modular open-source harness for real-time sign language recognition."
        )
        q1_p1.setStyleSheet(f"font-size: 14px; color: {palette.text_secondary}; line-height: 1.6;")
        q1_p1.setWordWrap(True)

        q1_p2 = QLabel("SignFlow Studio is the result of that curiosity, teamwork, and experimentation.")
        q1_p2.setStyleSheet(f"font-size: 14px; color: {palette.text_secondary}; line-height: 1.6;")
        q1_p2.setWordWrap(True)

        q1_block.addWidget(q1_title)
        q1_block.addWidget(q1_p1)
        q1_block.addWidget(q1_p2)
        layout.addLayout(q1_block)

        # =====================================================================
        # 4. 02 — WHY DID WE BUILD SIGNFLOW STUDIO?
        # =====================================================================
        q2_block = QVBoxLayout()
        q2_block.setSpacing(12)

        q2_title = QLabel("02 — Why did we build SignFlow Studio?")
        q2_title.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 700;
            color: {palette.text_primary};
            letter-spacing: -0.2px;
        """)

        q2_p1 = QLabel(
            "We wanted to explore whether a computer could recognize hand signs through a camera and "
            "identify them using machine learning."
        )
        q2_p1.setStyleSheet(f"font-size: 14px; color: {palette.text_secondary}; line-height: 1.6;")
        q2_p1.setWordWrap(True)

        q2_p2 = QLabel(
            "More importantly, we wanted to work on an open-source tool that could have a meaningful purpose "
            "in real-time communication accessibility."
        )
        q2_p2.setStyleSheet(f"font-size: 14px; color: {palette.text_secondary}; line-height: 1.6;")
        q2_p2.setWordWrap(True)

        q2_block.addWidget(q2_title)
        q2_block.addWidget(q2_p1)
        q2_block.addWidget(q2_p2)
        layout.addLayout(q2_block)

        # =====================================================================
        # 5. 03 — WHAT DOES OUR PROJECT ACTUALLY DO?
        # =====================================================================
        q3_block = QVBoxLayout()
        q3_block.setSpacing(12)

        q3_title = QLabel("03 — What does our project actually do?")
        q3_title.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 700;
            color: {palette.text_primary};
            letter-spacing: -0.2px;
        """)

        q3_p1 = QLabel(
            "SignFlow Studio is a Windows desktop application designed to bring our hand-sign "
            "recognition experiments into one place."
        )
        q3_p1.setStyleSheet(f"font-size: 14px; color: {palette.text_secondary}; line-height: 1.6;")
        q3_p1.setWordWrap(True)

        q3_p2 = QLabel(
            "Our project uses computer vision to detect hand landmarks and machine learning to classify "
            "supported signs. We are also exploring recognition of dynamic gestures that involve movement."
        )
        q3_p2.setStyleSheet(f"font-size: 14px; color: {palette.text_secondary}; line-height: 1.6;")
        q3_p2.setWordWrap(True)

        q3_p3 = QLabel(
            "It is an experimental prototype, and we are continuing to improve its recognition capabilities."
        )
        q3_p3.setStyleSheet(f"font-size: 14px; color: {palette.text_secondary}; line-height: 1.6;")
        q3_p3.setWordWrap(True)

        q3_block.addWidget(q3_title)
        q3_block.addWidget(q3_p1)
        q3_block.addWidget(q3_p2)
        q3_block.addWidget(q3_p3)
        layout.addLayout(q3_block)

        # =====================================================================
        # 6. 04 — WHERE CAN THIS PROJECT BE USED IN REAL LIFE?
        # =====================================================================
        q4_block = QVBoxLayout()
        q4_block.setSpacing(14)

        q4_title = QLabel("04 — Where can this project be used in real life?")
        q4_title.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 700;
            color: {palette.text_primary};
            letter-spacing: -0.2px;
        """)

        q4_intro = QLabel("We believe technologies like SignFlow Studio could eventually contribute to:")
        q4_intro.setStyleSheet(f"font-size: 14px; color: {palette.text_secondary}; line-height: 1.6;")
        q4_intro.setWordWrap(True)

        q4_block.addWidget(q4_title)
        q4_block.addWidget(q4_intro)

        # The 4 bullet points with bold titles
        use_cases = [
            (
                "Everyday communication",
                "Helping explore new ways for people who use sign language and people unfamiliar with it to communicate."
            ),
            (
                "Education",
                "Supporting interactive learning tools that help students practise and understand hand signs."
            ),
            (
                "Public-facing environments",
                "Exploring accessibility applications for reception desks, information counters, and similar spaces."
            ),
            (
                "AI research and development",
                "Providing a foundation for experimenting with gesture-recognition techniques and responsibly collected training datasets."
            ),
        ]

        points_vbox = QVBoxLayout()
        points_vbox.setSpacing(10)
        points_vbox.setContentsMargins(8, 0, 0, 0)

        for heading, body in use_cases:
            point_lbl = QLabel(f"<b>{heading}</b> — {body}")
            point_lbl.setTextFormat(Qt.TextFormat.RichText)
            point_lbl.setStyleSheet(f"font-size: 14px; color: {palette.text_secondary}; line-height: 1.55;")
            point_lbl.setWordWrap(True)
            points_vbox.addWidget(point_lbl)

        q4_block.addLayout(points_vbox)

        # Concluding disclaimer statement
        q4_conclusion = QLabel(
            "These are potential applications. Our current project is an open-source model harness and prototype "
            "designed for community experimentation and real-time accessibility."
        )
        q4_conclusion.setStyleSheet(f"font-size: 13.5px; color: {palette.text_muted}; line-height: 1.55; font-style: italic;")
        q4_conclusion.setWordWrap(True)
        q4_block.addWidget(q4_conclusion)

        layout.addLayout(q4_block)

        layout.addWidget(EditorialSeparator())

        # =====================================================================
        # 7. CLOSING STATEMENT
        # =====================================================================
        closing_lbl = QLabel("Built with curiosity. Improved through experimentation.")
        closing_lbl.setStyleSheet(f"""
            font-size: 14.5px;
            font-style: italic;
            color: {palette.text_secondary};
            letter-spacing: 0.2px;
            padding-bottom: 24px;
        """)
        closing_lbl.setWordWrap(True)
        layout.addWidget(closing_lbl)

        layout.addStretch()

        # Center the restrained content column in the viewport
        outer_layout.addStretch(1)
        outer_layout.addWidget(content_column, 0)
        outer_layout.addStretch(1)

        scroll.setWidget(outer_container)

        outer_page_layout = QVBoxLayout(self)
        outer_page_layout.setContentsMargins(0, 0, 0, 0)
        outer_page_layout.addWidget(scroll)


# Backward-compatible alias
AboutProjectPage = OverviewPage
