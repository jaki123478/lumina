# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['lumina_desktop/lumina_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('voices/it_IT-paola-medium.onnx', 'voices'),
        ('artifacts/vosk_italiano/vosk-model-small-it-0.22', 'artifacts/vosk_italiano/vosk-model-small-it-0.22'),
        ('config/lumina_voice_settings.json', 'config'),
        ('tokenizer_lumina_next', 'tokenizer_lumina_next'),
        ('checkpoints_lumina_image/lumina_image_v1.pt', 'checkpoints_lumina_image'),
    ],
    hiddenimports=['customtkinter', 'lumina_voice_neural', 'piper', 'vosk', 'lumina_general', 'lumina_image_gen', 'lumina_image_experimental', 'PIL', 'PIL.Image'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LuminaDesktop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
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
    name='LuminaDesktop',
)
