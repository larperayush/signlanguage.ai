# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Specification File for SignFlow Studio.
Builds standalone Windows Desktop Executable (.exe) with custom icon and assets.
"""
import sys
import os
from pathlib import Path

block_cipher = None

SPEC_ROOT = Path(os.path.abspath(SPECPATH))

# Include resources directory
datas = [
    (str(SPEC_ROOT / "app" / "resources"), "app/resources"),
]

if (SPEC_ROOT / "models").exists():
    datas.append((str(SPEC_ROOT / "models"), "models"))

a = Analysis(
    ['run_app.py'],
    pathex=[str(SPEC_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'PySide6.QtMultimedia',
        'PySide6.QtMultimediaWidgets',
        'PySide6.QtSvg',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'scipy', 'tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SignFlowStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # Windowed GUI application (no cmd terminal popup)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(SPEC_ROOT / 'app' / 'resources' / 'app_icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SignFlowStudio',
)
