# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src', 'src'),
        ('ffmpeg_bin', 'ffmpeg_bin'),
        ('venv6/Scripts', 'venv_scripts'),
        *collect_data_files('glassesTools'),  # ⬅ data-bestanden van glassesTools meenemen
    ],
    hiddenimports=[
        'glassesTools',
        'typeguard',
        'watchfiles',
        *collect_submodules('glassesTools'),  # ⬅ alle submodules meenemen
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='dualgaze_app',
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
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='dualgaze_app',
)
