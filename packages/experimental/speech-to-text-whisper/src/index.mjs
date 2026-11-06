/** Local Whisper speech-to-text provider (multilingual, including Brazilian Portuguese). */
import { isAbsolute, join } from 'node:path'
import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import z from '@deepseek-ai/schemastery'

export const name = 'experimental-speech-to-text-whisper'
export const inject = ['speechToText']

/** Language hints accepted by the Whisper exports and named by the voice UI. */
const LANGUAGES = ['auto', 'pt', 'en', 'es', 'fr', 'de', 'ja', 'zh', 'ko', 'ru']

/** Model files live under `${dataRoot}/models/${model}`; the Silero VAD is shared with SenseVoice. */
export const Config = z.object({
  providerId: z.string().min(1).default('whisper-local'),
  dataRoot: z.string().min(1).required(),
  model: z.string().min(1).default('base'),
  vadModelPath: z.string().min(1).required(),
  threads: z.natural().min(1).default(4),
})

/** One warm inference process; recordings are serialized so Whisper never decodes concurrently. */
function createWorker(config) {
  const workerPath = fileURLToPath(new URL('./worker.mjs', import.meta.url))
  const modelDir = join(config.dataRoot, 'models', config.model)
  let child
  let buffer = ''
  let pending = null
  let chain = Promise.resolve()

  const failPending = (message) => {
    const waiter = pending
    pending = null
    waiter?.reject(new Error(message))
  }

  const ensureWorker = () => {
    if (child && child.exitCode === null && !child.killed) return child
    buffer = ''
    pending = null
    const spawned = spawn(process.execPath, [workerPath, modelDir, config.vadModelPath,
      String(config.threads), config.model], { stdio: ['pipe', 'pipe', 'pipe'] })
    spawned.stdout.setEncoding('utf8')
    spawned.stdout.on('data', (chunk) => {
      buffer += chunk
      let index
      while ((index = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, index).trim()
        buffer = buffer.slice(index + 1)
        if (!line) continue
        let message
        try { message = JSON.parse(line) } catch { continue }
        const settle = pending
        pending = null
        if (!settle) continue
        if (message.ok) settle.resolve(message)
        else settle.reject(new Error(message.error ?? 'Whisper worker failed'))
      }
    })
    spawned.stderr.setEncoding('utf8')
    spawned.stderr.on('data', (data) => process.stderr.write(`[whisper-stt] ${data}`))
    spawned.stdin.on('error', (error) => failPending(`Whisper worker input failed: ${error.message}`))
    spawned.on('error', (error) => failPending(`Whisper worker failed to start: ${error.message}`))
    spawned.on('exit', (code) => {
      if (child === spawned) child = undefined
      failPending(`Whisper worker exited (code ${code})`)
    })
    child = spawned
    return spawned
  }

  const transcribe = async (audio, language, signal) => {
    signal.throwIfAborted()
    const run = chain.then(() => new Promise((resolve, reject) => {
      if (signal.aborted) return reject(new Error('cancelled'))
      const worker = ensureWorker()
      const onAbort = () => failPending('cancelled')
      signal.addEventListener('abort', onAbort, { once: true })
      pending = {
        resolve: (message) => { signal.removeEventListener('abort', onAbort); resolve(message) },
        reject: (error) => { signal.removeEventListener('abort', onAbort); reject(error) },
      }
      worker.stdin.write(`${JSON.stringify({
        audio: Buffer.from(audio).toString('base64'),
        language: language === 'auto' ? '' : language,
      })}\n`)
    }))
    chain = run.catch(() => {})
    const started = performance.now()
    const message = await run
    return {
      text: message.text ?? '',
      audioSeconds: Math.max(0, (audio.byteLength - 44) / 32000),
      inferenceSeconds: message.seconds ?? (performance.now() - started) / 1000,
    }
  }

  return { transcribe, dispose: () => { child?.kill(); child = undefined } }
}

export function apply(ctx, config) {
  for (const path of [config.dataRoot, config.vadModelPath]) {
    if (!isAbsolute(path)) throw new Error(`Whisper paths must be absolute: ${path}`)
  }
  const worker = createWorker(config)
  const estimatedBytes = 200_000_000
  ctx.effect(() => {
    const unregister = ctx.speechToText.register({
      info: {
        id: config.providerId,
        name: `Whisper ${config.model} (local)`,
        location: 'host-local',
        languages: LANGUAGES,
        setupEstimate: { recommendedDiskBytes: estimatedBytes, expectedMemoryBytes: estimatedBytes,
          minimumMinutes: 1, maximumMinutes: 5 },
      },
      transcribe: (input, signal) => worker.transcribe(input.audio, input.language, signal),
    })
    return async () => {
      await unregister()
      worker.dispose()
    }
  })
}
