@echo off
setlocal

echo ===================================================
echo FRIDAY Inference Engine - Real Model Pipeline
echo ===================================================

:: -----------------------------------------------------
:: SSH AYARLARI (Bir onceki script ile ayni)
:: -----------------------------------------------------
set SSH_USER=node-1
set SSH_HOST=192.168.1.21
set SSH_PORT=22
set REMOTE_DIR=/home/node-1
:: -----------------------------------------------------

echo.
echo [1/4] Fetch: Guncel convert_to_friday.py dosyasi uzak sunucudan cekiliyor...
REM scp -P %SSH_PORT% %SSH_USER%@%SSH_HOST%:%REMOTE_DIR%/convert_to_friday.py .
if %errorlevel% neq 0 (
    echo.
    echo [HATA] Dosya cekilemedi! SSH baglantinizi ve bilgileri kontrol edin.
    REM exit /b %errorlevel%
)
echo Basarili (Lokal kopya kullaniliyor).

echo.
echo [2/4] Environment Setup: Gerekli Python kutuphaneleri kuruluyor...
pip install huggingface_hub safetensors transformers numpy ml_dtypes
if %errorlevel% neq 0 (
    echo.
    echo [HATA] Python kutuphaneleri kurulamadi! Lutfen Python ve pip'in kurulu oldugundan emin olun.
    exit /b %errorlevel%
)
echo Basarili.

echo.
echo [3/4] Data Ingestion: Model indiriliyor ve 1.58-bit Ternary formatina cevriliyor...
echo (Lutfen bekleyin, bu islem internet hizina ve model boyutuna gore birkac dakika surebilir...)
python convert_to_friday.py
if %errorlevel% neq 0 (
    echo.
    echo [HATA] Model cevirme islemi (convert_to_friday.py) basarisiz oldu!
    exit /b %errorlevel%
)
echo Basarili. friday_model.bin ve tokenizer_config.json olusturuldu.

echo.
echo [4/4] Ignition: FRIDAY Motoru Atesleniyor...
if not exist "friday_engine.exe" (
    echo.
    echo [HATA] friday_engine.exe bulunamadi! Lutfen once deploy_and_build.bat ile projeyi derlediginize emin olun.
    exit /b 1
)

echo Baslatiliyor...
echo ===================================================
friday_engine.exe

endlocal
