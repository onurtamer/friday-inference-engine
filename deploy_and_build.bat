@echo off
setlocal

echo ===================================================
echo FRIDAY Inference Engine - Deploy and Build Script
echo ===================================================

:: -----------------------------------------------------
:: KULLANICI AYARLARI (Lutfen burayi kendinize gore doldurun)
:: -----------------------------------------------------
set SSH_USER=node-1
set SSH_HOST=192.168.1.21
set SSH_PORT=22
:: SSH anahtariniz varsa yolunu girin (orn: C:\Users\isim\.ssh\id_rsa). Sifre kullanacaksaniz bos birakin: set SSH_KEY_PATH=
set SSH_KEY_PATH=
set REMOTE_DIR=/home/node-1
:: -----------------------------------------------------

echo.
echo [ASAMA 1] Dosyalar Uzak Sunucudan (Node) Cekiliyor...

set SCP_CMD=scp -P %SSH_PORT%
if not "%SSH_KEY_PATH%"=="" set SCP_CMD=%SCP_CMD% -i "%SSH_KEY_PATH%"

echo Sunucuya baglaniliyor ve dosyalar indiriliyor (.cpp, .h, .cu, .py)...
REM %SCP_CMD% %SSH_USER%@%SSH_HOST%:%REMOTE_DIR%/*.cpp .
REM %SCP_CMD% %SSH_USER%@%SSH_HOST%:%REMOTE_DIR%/*.h .
REM %SCP_CMD% %SSH_USER%@%SSH_HOST%:%REMOTE_DIR%/*.cu .
REM %SCP_CMD% %SSH_USER%@%SSH_HOST%:%REMOTE_DIR%/*.py .

echo.
echo [ASAMA 2] Native Windows Derlemesi Basliyor...

:: CUDA Path ayarlamasi (Sistemde yoksa varsayilan yola ayarla)
if "%CUDA_PATH%"=="" set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

where cl >nul 2>nul
if %errorlevel% equ 0 goto check_nvcc

echo [Bilgi] cl.exe PATH uzerinde yok. Visual Studio 2022 ayarlari otomatik yukleniyor...
if not exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    echo HATA: Visual Studio kurulumu bulunamadi.
    exit /b 1
)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul

:check_nvcc
where nvcc >nul 2>nul
if %errorlevel% equ 0 goto start_build

echo [Bilgi] nvcc PATH uzerinde yok. CUDA Toolkit yolu otomatik ekleniyor...
set "PATH=%CUDA_PATH%\bin;%PATH%"
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo HATA: nvcc CUDA Toolkit bulunamadi. Lutfen CUDA kurulumunuzu kontrol edin.
    exit /b 1
)

:start_build
echo.
echo [1/3] CUDA Cekirdekleri derleniyor...
:: Hata 1 ve Hata 4 Cozumu: Joker (*) kaldirildi. 
:: transformer.cpp icinde CUDA kodlari oldugu icin NVCC ile -x cu (CUDA dosyasi gibi davran) flagiyle derliyoruz.
nvcc -O3 -Xcompiler "/EHsc" --std=c++17 -c cuda_math_kernels.cu -x cu transformer.cpp
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [2/3] C++ Modulleri derleniyor...
:: transformer.cpp haricindeki C++ dosyalarini cl.exe'ye acikca listeleyerek veriyoruz.
cl /nologo /EHsc /std:c++17 /O2 /I"%CUDA_PATH%\include" /c main.cpp io_manager.cpp memory_manager.cpp tokenizer.cpp
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [3/3] Objeler birlestiriliyor Link...
link /NOLOGO *.obj /OUT:friday_engine.exe /LIBPATH:"%CUDA_PATH%\lib\x64" cudart_static.lib cuda.lib kernel32.lib user32.lib advapi32.lib shell32.lib
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ===================================================
echo DEPLOYMENT VE BUILD BASARILI - friday_engine.exe hazir!
echo ===================================================
endlocal
