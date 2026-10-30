# Pubblicazione Lumina

## Koyeb

Usare `hf_complete_server/Dockerfile` come servizio web, porta `7860`.
Impostare le variabili d'ambiente:

- `LUMINA_MODEL_REPO`: repository Hugging Face del modello
- `HF_TOKEN`: token Hugging Face con permesso di lettura, se il repository è privato

Il modello completo richiede risorse adeguate; il piano gratuito è adatto solo a prove leggere.

## Cloudflare Tunnel verso il PC fisso

1. Installare `cloudflared` sul PC che esegue Lumina.
2. Avviare il server locale sulla porta `8000`.
3. Eseguire `AVVIA_CLOUDFLARE_LUMINA.bat`.
4. Il comando mostrerà il link HTTPS temporaneo da condividere.

Non inserire mai il token del tunnel nei file versionati.
