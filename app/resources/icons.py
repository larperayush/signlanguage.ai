"""
High-DPI SVG vector icon generator for PySide6.
Provides crisp icons in any resolution and custom theme colors.
"""
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtCore import QByteArray, QSize, Qt

# Crisp Lucide / Feather SVG paths (viewBox 0 0 24 24)
SVG_PATHS = {
    "overview": (
        '<rect width="7" height="9" x="3" y="3" rx="1.5"/>'
        '<rect width="7" height="5" x="14" y="3" rx="1.5"/>'
        '<rect width="7" height="9" x="14" y="12" rx="1.5"/>'
        '<rect width="7" height="5" x="3" y="16" rx="1.5"/>'
    ),
    "predict": (
        '<path d="M2 12h3"/>'
        '<path d="M19 12h3"/>'
        '<path d="M12 2v3"/>'
        '<path d="M12 19v3"/>'
        '<circle cx="12" cy="12" r="7"/>'
        '<circle cx="12" cy="12" r="2"/>'
    ),
    "static_training": (
        '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>'
        '<polyline points="3.27 6.96 12 12.01 20.73 6.96"/>'
        '<line x1="12" y1="22.08" x2="12" y2="12"/>'
    ),
    "dynamic_training": (
        '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>'
    ),
    "settings": (
        '<line x1="4" y1="21" x2="4" y2="14"/>'
        '<line x1="4" y1="10" x2="4" y2="3"/>'
        '<line x1="12" y1="21" x2="12" y2="12"/>'
        '<line x1="12" y1="8" x2="12" y2="3"/>'
        '<line x1="20" y1="21" x2="20" y2="16"/>'
        '<line x1="20" y1="12" x2="20" y2="3"/>'
        '<line x1="1" y1="14" x2="7" y2="14"/>'
        '<line x1="9" y1="8" x2="15" y2="8"/>'
        '<line x1="17" y1="16" x2="23" y2="16"/>'
    ),
    "camera": (
        '<path d="m22 8-6 4 6 4V8Z"/>'
        '<rect width="14" height="12" x="2" y="6" rx="2" ry="2"/>'
    ),
    "camera_off": (
        '<line x1="2" y1="2" x2="22" y2="22"/>'
        '<path d="m22 8-6 4 6 4V8Z"/>'
        '<path d="M10 6h6a2 2 0 0 1 2 2v6"/>'
        '<path d="M2 9v7a2 2 0 0 0 2 2h11"/>'
    ),
    "play": (
        '<polygon points="6 3 20 12 6 21 6 3" fill="currentColor"/>'
    ),
    "stop": (
        '<rect width="14" height="14" x="5" y="5" rx="2" fill="currentColor"/>'
    ),
    "refresh": (
        '<path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>'
        '<path d="M3 3v5h5"/>'
        '<path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/>'
        '<path d="M16 21h5v-5"/>'
    ),
    "trash": (
        '<path d="M3 6h18"/>'
        '<path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/>'
        '<path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>'
    ),
    "copy": (
        '<rect width="13" height="13" x="9" y="9" rx="2" ry="2"/>'
        '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>'
    ),
    "check": (
        '<polyline points="20 6 9 17 4 12"/>'
    ),
    "alert": (
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/>'
        '<line x1="12" y1="17" x2="12.01" y2="17"/>'
    ),
    "info": (
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="12" y1="16" x2="12" y2="12"/>'
        '<line x1="12" y1="8" x2="12.01" y2="8"/>'
    ),
    "plus": (
        '<line x1="12" y1="5" x2="12" y2="19"/>'
        '<line x1="5" y1="12" x2="19" y2="12"/>'
    ),
    "sun": (
        '<circle cx="12" cy="12" r="4"/>'
        '<path d="M12 2v2"/>'
        '<path d="M12 20v2"/>'
        '<path d="m4.93 4.93 1.41 1.41"/>'
        '<path d="m17.66 17.66 1.41 1.41"/>'
        '<path d="M2 12h2"/>'
        '<path d="M20 12h2"/>'
        '<path d="m6.34 17.66-1.41 1.41"/>'
        '<path d="m19.07 4.93-1.41 1.41"/>'
    ),
    "moon": (
        '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>'
    ),
    "apps": (
        '<rect width="7" height="7" x="3" y="3" rx="1"/>'
        '<rect width="7" height="7" x="14" y="3" rx="1"/>'
        '<rect width="7" height="7" x="14" y="14" rx="1"/>'
        '<rect width="7" height="7" x="3" y="14" rx="1"/>'
    ),
    "terminal": (
        '<polyline points="4 17 10 11 4 5"/>'
        '<line x1="12" y1="19" x2="20" y2="19"/>'
    ),
    "chevron_down": (
        '<polyline points="6 9 12 15 18 9"/>'
    ),
    "chevron_right": (
        '<polyline points="9 18 15 12 9 6"/>'
    ),
    "search": (
        '<circle cx="11" cy="11" r="8"/>'
        '<line x1="21" y1="21" x2="16.65" y2="16.65"/>'
    ),
    "layers": (
        '<polygon points="12 2 2 7 12 12 22 7 12 2"/>'
        '<polyline points="2 17 12 22 22 17"/>'
        '<polyline points="2 12 12 17 22 12"/>'
    ),
    "cpu": (
        '<rect width="16" height="16" x="4" y="4" rx="2"/>'
        '<rect width="6" height="6" x="9" y="9"/>'
        '<line x1="9" y1="1" x2="9" y2="4"/>'
        '<line x1="15" y1="1" x2="15" y2="4"/>'
        '<line x1="9" y1="20" x2="9" y2="23"/>'
        '<line x1="15" y1="20" x2="15" y2="23"/>'
        '<line x1="20" y1="9" x2="23" y2="9"/>'
        '<line x1="20" y1="14" x2="23" y2="14"/>'
        '<line x1="1" y1="9" x2="4" y2="9"/>'
        '<line x1="1" y1="14" x2="4" y2="14"/>'
    ),
    "sliders": (
        '<line x1="4" y1="21" x2="4" y2="14"/>'
        '<line x1="4" y1="10" x2="4" y2="3"/>'
        '<line x1="12" y1="21" x2="12" y2="12"/>'
        '<line x1="12" y1="8" x2="12" y2="3"/>'
        '<line x1="20" y1="21" x2="20" y2="16"/>'
        '<line x1="20" y1="12" x2="20" y2="3"/>'
        '<line x1="1" y1="14" x2="7" y2="14"/>'
        '<line x1="9" y1="8" x2="15" y2="8"/>'
        '<line x1="17" y1="16" x2="23" y2="16"/>'
    ),
    "database": (
        '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
        '<path d="M3 5V19A9 3 0 0 0 21 19V5"/>'
        '<path d="M3 12A9 3 0 0 0 21 12"/>'
    ),
    "box": (
        '<path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/>'
        '<path d="m3.3 7 8.7 5 8.7-5"/>'
        '<path d="M12 22V12"/>'
    ),
    "folder": (
        '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>'
    ),
    "undo": (
        '<path d="M3 7v6h6"/>'
        '<path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13"/>'
    ),
    "file_text": (
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
        '<path d="M14 2v4a2 2 0 0 0 2 2h4"/>'
        '<path d="M10 9H8"/>'
        '<path d="M16 13H8"/>'
        '<path d="M16 17H8"/>'
    ),
    "link": (
        '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>'
        '<path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>'
    ),
    "shield_check": (
        '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>'
        '<path d="m9 12 2 2 4-4"/>'
    ),
    "user": (
        '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/>'
        '<circle cx="12" cy="7" r="4"/>'
    ),
    "school": (
        '<path d="m4 6 8-4 8 4-8 4-8-4Z"/>'
        '<path d="M6 10v6c0 3 3 5 6 5s6-2 6-5v-6"/>'
        '<path d="M22 10v6"/>'
    ),
    "target": (
        '<circle cx="12" cy="12" r="10"/>'
        '<circle cx="12" cy="12" r="6"/>'
        '<circle cx="12" cy="12" r="2"/>'
    ),
    "sparkles": (
        '<path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3L12 3z"/>'
    ),
    "compass": (
        '<circle cx="12" cy="12" r="10"/>'
        '<polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/>'
    ),
    "arrow_right": (
        '<line x1="5" y1="12" x2="19" y2="12"/>'
        '<polyline points="12 5 19 12 12 19"/>'
    ),
    "folder_open": (
        '<path d="m6 14 1.45-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.55 6a2 2 0 0 1-1.94 1.5H4a2 2 0 0 1-2-2V5c0-1.1.9-2 2-2h3.93a2 2 0 0 1 1.66.9l.82 1.2a2 2 0 0 0 1.66.9H18a2 2 0 0 1 2 2v2"/>'
    ),
}

def get_svg_markup(name: str, color: str = "#94A3B8", stroke_width: float = 2.0) -> str:
    path_data = SVG_PATHS.get(name, SVG_PATHS["info"])
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" 
            width="24" height="24" fill="none" stroke="{color}" 
            stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">
            {path_data}
            </svg>"""

def get_pixmap(name: str, color: str = "#94A3B8", size: int = 20, stroke_width: float = 2.0) -> QPixmap:
    svg_data = get_svg_markup(name, color, stroke_width)
    renderer = QSvgRenderer(QByteArray(svg_data.encode("utf-8")))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return pixmap

def get_icon(name: str, color: str = "#94A3B8", active_color: str = "#5E6AD2", size: int = 20) -> QIcon:
    normal_pixmap = get_pixmap(name, color, size)
    icon = QIcon()
    icon.addPixmap(normal_pixmap, QIcon.Mode.Normal)
    
    if active_color:
        active_pixmap = get_pixmap(name, active_color, size)
        icon.addPixmap(active_pixmap, QIcon.Mode.Active)
        icon.addPixmap(active_pixmap, QIcon.Mode.Selected)
    return icon
