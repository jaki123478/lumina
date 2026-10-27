@echo off
chcp 65001 >nul
title Lumina AI - Gestione Consenso Internet e Privacy
color 0A

:MENU
cls
echo =====================================================================
echo       🛡️ LUMINA SENTINEL: GESTIONE CONSENSO INTERNET E PRIVACY
echo =====================================================================
echo.
echo Direttiva Sovrana:
echo "Lumina opera di default 100%% offline in locale su Visual Studio,
echo  terminali e strumenti di sviluppo. L'accesso a Internet viene attivato
echo  esclusivamente su esplicito consenso dell'utente."
echo.
echo =====================================================================
echo  STATO ATTUALE:
if exist .venv\Scripts\python.exe (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)
%PYTHON_EXEC% -c "from lumina_network_policy import get_network_policy; p = get_network_policy(); print('  • Consenso Attivo : ' + ('🟢 ABILITATO' if p.is_internet_allowed() else '🔴 DISABILITATO (100%% Locale / Offline)')); print('  • Ambito Consenso : ' + p.policy.get('consent_scope', 'none'))"
echo =====================================================================
echo.
echo  [1] ACCORDA Consenso Internet (Solo per la Sessione Corrente)
echo  [2] ACCORDA Consenso Internet (Permanente)
echo  [3] REVOCA Consenso (Ripristina Modalità 100%% OFFLINE / Blindata)
echo  [4] Mostra Dettagli Politica di Rete (config\lumina_policy.json)
echo  [5] Esci
echo.
set /p SCELTA="Scegli un'opzione (1-5): "

if "%SCELTA%"=="1" (
    %PYTHON_EXEC% -c "from lumina_network_policy import get_network_policy; get_network_policy().grant_consent('session'); print('\n[+] Consenso accordato per la sessione corrente!')"
    pause
    goto MENU
)
if "%SCELTA%"=="2" (
    %PYTHON_EXEC% -c "from lumina_network_policy import get_network_policy; get_network_policy().grant_consent('permanent'); print('\n[+] Consenso accordato in modo permanente!')"
    pause
    goto MENU
)
if "%SCELTA%"=="3" (
    %PYTHON_EXEC% -c "from lumina_network_policy import get_network_policy; get_network_policy().revoke_consent(); print('\n[+] Consenso revocato! Lumina opera ora al 100%% in modalità OFFLINE.')"
    pause
    goto MENU
)
if "%SCELTA%"=="4" (
    cls
    type config\lumina_policy.json
    echo.
    pause
    goto MENU
)
if "%SCELTA%"=="5" (
    exit /b 0
)

goto MENU
