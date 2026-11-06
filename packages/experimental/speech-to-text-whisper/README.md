---
description: "Local Whisper ONNX transcription provider with Brazilian Portuguese support."
kind: "package-reference"
---

# @deepseek-ai/dsh-experimental-speech-to-text-whisper

English | [Português](#português)

## Summary

Optional local speech recognizer for the [voice input bundle](../voice-input-bundle/README.md).
It registers a `host-local` provider backed by a multilingual Whisper export running through
`sherpa-onnx-node` on the Host CPU, so Brazilian Portuguese is recognized offline.

## Use this package

The provider expects the Whisper model files and a Silero VAD file on disk; it performs no
downloads. The bundle configures `dataRoot` under the DSH home and points `vadModelPath` at the
shared Silero model.

| Config | Meaning |
| --- | --- |
| `providerId` | Registered id; defaults to `whisper-local`. |
| `dataRoot` | Absolute directory containing `models/<model>/`. |
| `model` | File prefix inside `models/<model>/`; defaults to `base`. |
| `vadModelPath` | Absolute path to `silero_vad.onnx`. |
| `threads` | CPU threads used by ONNX Runtime. |

Expected files for `model: base`:

```
<dataRoot>/models/base/base-encoder.int8.onnx
<dataRoot>/models/base/base-decoder.int8.onnx
<dataRoot>/models/base/base-tokens.txt
```

They can be fetched from `csukuangfj/sherpa-onnx-whisper-base` on Hugging Face. Larger exports
(`small`, `medium`) improve Portuguese accuracy at a higher CPU and memory cost.

## Understand the implementation

`src/index.mjs` registers the provider and owns one warm `worker.mjs` process. The worker loads the
ONNX encoder/decoder, segments the recording with Silero VAD, decodes every segment, and returns the
joined transcript over JSON lines. The browser always sends canonical 16 kHz mono PCM16 WAV, which
the worker validates before inference. `auto` leaves the language unset so Whisper detects it.

## Known limitations

- No preparation UI or download step: missing files surface as inference errors.
- Complete-recording transcription only, no streaming or speech synthesis.
- CPU-only inference; a few seconds of audio take roughly one second on `base` int8.

## Português

Provedor local de reconhecimento de fala para o compositor do DSH, com suporte a português do
Brasil. O modelo Whisper é executado localmente via `sherpa-onnx-node`, sem enviar áudio para a
nuvem. Configure `dataRoot` e `vadModelPath` com caminhos absolutos; o worker valida WAV PCM16 mono
de 16 kHz e usa VAD Silero para fatiar gravações longas.
