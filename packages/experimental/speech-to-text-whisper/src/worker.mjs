/** Local Whisper transcription worker (stdio JSON lines, one warm model per process). */
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const sherpa = require('sherpa-onnx-node')

const [, , modelDir, vadPath, threadsArg, modelName = 'base'] = process.argv
const threads = Number(threadsArg ?? 4) || 4
const encoder = `${modelDir}/${modelName}-encoder.int8.onnx`
const decoder = `${modelDir}/${modelName}-decoder.int8.onnx`
const tokens = `${modelDir}/${modelName}-tokens.txt`

const nativeConfig = {
  featConfig: { sampleRate: 16000, featureDim: 80 },
  modelConfig: {
    whisper: { encoder, decoder, language: 'pt', task: 'transcribe', tailPaddings: -1 },
    tokens,
    numThreads: threads,
    provider: 'cpu',
    debug: 0,
  },
}

let recognizer
try {
  recognizer = new sherpa.OfflineRecognizer(nativeConfig)
} catch (error) {
  process.stdout.write(`${JSON.stringify({ ok: false, error: `model load failed: ${error?.message ?? error}` })}\n`)
  process.exit(1)
}

const vad = new sherpa.Vad({
  sileroVad: { model: vadPath, threshold: 0.5, minSilenceDuration: 0.4, minSpeechDuration: 0.25,
    maxSpeechDuration: 25, windowSize: 512 },
  sampleRate: 16000, numThreads: threads, provider: 'cpu', debug: 0,
}, 27)

/**
 * Transcribe one 16 kHz mono PCM16 WAV.
 * @param {Buffer} audio - complete WAV bytes.
 * @param {string} language - Whisper language code such as `pt`; empty detects it.
 */
function transcribe(audio, language) {
  const started = performance.now()
  const view = new DataView(audio.buffer, audio.byteOffset, audio.byteLength)
  if (audio.length < 44 || view.getUint32(0, true) !== 0x46464952) throw new Error('not a RIFF WAV')
  if (view.getUint16(22, true) !== 1) throw new Error('WAV must be mono PCM16')
  if (view.getUint32(24, true) !== 16000) throw new Error('WAV must be 16 kHz')
  const pcm = new DataView(audio.buffer, audio.byteOffset + 44, audio.byteLength - 44)
  const samples = Float32Array.from({ length: pcm.byteLength / 2 }, (_, i) => pcm.getInt16(i * 2, true) / 32768)
  nativeConfig.modelConfig.whisper.language = language
  recognizer.setConfig(nativeConfig)
  vad.reset()
  const texts = []
  const drain = () => {
    while (!vad.isEmpty()) {
      const segment = vad.front(false)
      const stream = recognizer.createStream()
      stream.acceptWaveform({ sampleRate: 16000, samples: segment.samples })
      recognizer.decode(stream)
      texts.push(recognizer.getResult(stream).text.trim())
      vad.pop()
    }
  }
  for (let offset = 0; offset < samples.length; offset += 512) {
    vad.acceptWaveform(samples.subarray(offset, offset + 512))
    drain()
  }
  vad.flush()
  drain()
  return { text: texts.filter(Boolean).join(' ').trim(), seconds: (performance.now() - started) / 1000 }
}

let queue = Promise.resolve()
process.stdin.setEncoding('utf8')
let pending = ''
process.stdin.on('data', (chunk) => {
  pending += chunk
  let index
  while ((index = pending.indexOf('\n')) >= 0) {
    const line = pending.slice(0, index).trim()
    pending = pending.slice(index + 1)
    if (!line) continue
    const request = JSON.parse(line)
    queue = queue.then(() => {
      try {
        const result = transcribe(Buffer.from(request.audio, 'base64'), request.language ?? 'pt')
        process.stdout.write(`${JSON.stringify({ ok: true, ...result })}\n`)
      } catch (error) {
        process.stdout.write(`${JSON.stringify({ ok: false, error: String(error?.message ?? error) })}\n`)
      }
    })
  }
})
process.stdin.on('end', () => { queue.finally(() => process.exit(0)) })
