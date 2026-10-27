param (
    [Parameter(Mandatory=$true)]
    [string]$Text,
    [Parameter(Mandatory=$true)]
    [string]$OutputFile,
    [switch]$SpeakLive
)

try {
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    
    # Try selecting Italian voice if available
    try {
        $synth.SelectVoice('Microsoft Elsa Desktop')
    } catch {
        # Fallback to default
    }
    
    # Set speech rate and volume
    $synth.Rate = 0
    $synth.Volume = 100

    # Ensure parent dir exists
    $parent = [System.IO.Path]::GetDirectoryName($OutputFile)
    if (-not [string]::IsNullOrEmpty($parent) -and -not (Test-Path $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    # Save to WAV
    $synth.SetOutputToWaveFile($OutputFile)
    $synth.Speak($Text)
    $synth.SetOutputToDefaultAudioDevice()

    if ($SpeakLive) {
        $synth.Speak($Text)
    }

    $synth.Dispose()
    Write-Output "SUCCESS: Audio saved to $OutputFile"
} catch {
    Write-Error $_.Exception.Message
    exit 1
}
