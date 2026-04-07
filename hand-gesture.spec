import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

blockifier = 'PyInstaller.utils._bootstrap_internal.blockifier'

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # MediaPipe model files
        ('C:/Users/Aaron/AppData/Local/programs/Python/Python312/Lib/site-packages/mediapipe/python', 'mediapipe/python'),
        *collect_data_files('mediapipe'),
        *collect_data_files('opencv_python'),
    ],
    hiddenimports=[
        'mediapipe',
        'mediapipe.python',
        'mediapipe.python.solutions',
        'mediapipe.python.solutions.hands',
        'mediapipe.python.solutions.drawing_utils',
        'mediapipe.python.solutions.drawing_styles',
        'mediapipe.python._framework_bindings',
        'cv2',
        'pyautogui',
        'numpy',
        'numpy.core._multiarray_umath',
        'numpy.core.multiarray',
        'PIL',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HandGesture',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HandGesture',
)
