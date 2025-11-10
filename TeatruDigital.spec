# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Colectăm manual TOATE fișierele din pachetul speechbrain pentru a evita erorile
datas_speechbrain, binaries_speechbrain, hiddenimports_speechbrain = collect_all('speechbrain')

a = Analysis(
    ['main_app.py'],
    pathex=[],
    binaries=binaries_speechbrain,  # Adăugăm binarele de la speechbrain
    
    # Lista completă și corectă a tuturor resurselor externe
    datas=[
        ('assets', 'assets'),
        ('characters', 'characters'),
        ('curriculum', 'curriculum'),
        ('scenes', 'scenes'),
        ('managers', 'managers'),
        ('pretrained_models', 'pretrained_models'),
        ('Backgrounds', 'Backgrounds'),
        ('config.json', '.'),
        ('family.json', '.'),
        ('.env', '.'),
        ('bin', 'bin')
    ] + datas_speechbrain,  # Adăugăm și resursele interne ale speechbrain
    
    # Lista completă a modulelor pe care PyInstaller nu le detectează automat
    hiddenimports=[
        'pygame', 
        'sounddevice._ffi', 
        'soundfile',
        'pkg_resources.py2_warn'
    ] + hiddenimports_speechbrain,  # Adăugăm și importurile ascunse ale speechbrain

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='TeatruDigital',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # Setat pe False pentru versiunea finală, fără consolă
    # Setează pe True DOAR pentru depanare
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
    name='TeatruDigital',
)