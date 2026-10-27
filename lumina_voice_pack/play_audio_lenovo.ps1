param (
    [string]$File = ""
)

Add-Type -AssemblyName PresentationCore

$audioDir = Join-Path $PSScriptRoot "audio"

if ($File -and (Test-Path (Join-Path $audioDir $File))) {
    $target = Join-Path $audioDir $File
    Write-Host "🔊 Riproduzione audio: $File" -ForegroundColor Cyan
    $player = New-Object System.Windows.Media.MediaPlayer
    $player.Open([System.Uri]$target)
    $player.Play()
    Start-Sleep -Seconds 1
    while ($player.NaturalDuration.HasTimeSpan -and $player.Position -lt $player.NaturalDuration.TimeSpan) {
        Start-Sleep -Milliseconds 500
    }
    $player.Close()
    exit 0
}

Write-Host "=================================================================" -ForegroundColor Green
Write-Host "       AUDIO UFFICIALI GENERATI DALLA VOCE DI LUMINA" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "Tutti i file sono stati sintetizzati con Microsoft Elsa (it-IT)"
Write-Host ""

$files = Get-ChildItem $audioDir -Filter *.wav
$i = 1
foreach ($f in $files) {
    Write-Host "[$i] $($f.Name) ($([math]::Round($f.Length/1024, 1)) KB)" -ForegroundColor Cyan
    $i++
}
Write-Host "[A] Riproduci TUTTI i file in sequenza" -ForegroundColor Magenta
Write-Host "[Q] Esci" -ForegroundColor Red
Write-Host ""

$choice = Read-Host "Scegli un'opzione"

if ($choice -eq "A" -or $choice -eq "a") {
    foreach ($f in $files) {
        Write-Host "`n🔊 In riproduzione: $($f.Name)..." -ForegroundColor Yellow
        $player = New-Object System.Windows.Media.MediaPlayer
        $player.Open([System.Uri]$f.FullName)
        $player.Play()
        Start-Sleep -Seconds 1
        while ($player.NaturalDuration.HasTimeSpan -and $player.Position -lt $player.NaturalDuration.TimeSpan) {
            Start-Sleep -Milliseconds 500
        }
        $player.Close()
    }
} elseif ($choice -match '^\d+$' -and [int]$choice -ge 1 -and [int]$choice -le $files.Count) {
    $sel = $files[[int]$choice - 1]
    Write-Host "`n🔊 In riproduzione: $($sel.Name)..." -ForegroundColor Yellow
    $player = New-Object System.Windows.Media.MediaPlayer
    $player.Open([System.Uri]$sel.FullName)
    $player.Play()
    Start-Sleep -Seconds 1
    while ($player.NaturalDuration.HasTimeSpan -and $player.Position -lt $player.NaturalDuration.TimeSpan) {
        Start-Sleep -Milliseconds 500
    }
    $player.Close()
} else {
    Write-Host "Uscita."
}
