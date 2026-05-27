import { useState } from 'react'

const MODES = ['general', 'rag']
const MODELS = ['haiku', 'sonnet', 'opus']
const VERSIONS = ['v1', 'v2', 'v3']

function toggle(list, item) {
  return list.includes(item) ? list.filter(x => x !== item) : [...list, item]
}

export default function RunButton({ onRun, isRunning }) {
  const [mode, setMode] = useState('general')
  const [models, setModels] = useState(['haiku'])
  const [versions, setVersions] = useState(['v1'])

  function handleSubmit() {
    if (!models.length || !versions.length) return
    onRun({ mode, model: models, prompt_versions: versions })
  }

  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 p-5 space-y-4">
      <div className="flex flex-wrap gap-6 items-end">
        <div className="space-y-1.5">
          <p className="text-xs text-zinc-400 uppercase tracking-wider">Mode</p>
          <div className="flex gap-2">
            {MODES.map(m => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                  mode === m ? 'bg-indigo-600 text-white' : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-1.5">
          <p className="text-xs text-zinc-400 uppercase tracking-wider">Model</p>
          <div className="flex gap-2">
            {MODELS.map(m => (
              <button
                key={m}
                onClick={() => setModels(toggle(models, m))}
                className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                  models.includes(m) ? 'bg-indigo-600 text-white' : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-1.5">
          <p className="text-xs text-zinc-400 uppercase tracking-wider">Prompt Version</p>
          <div className="flex gap-2">
            {VERSIONS.map(v => (
              <button
                key={v}
                onClick={() => setVersions(toggle(versions, v))}
                className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                  versions.includes(v) ? 'bg-indigo-600 text-white' : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'
                }`}
              >
                {v}
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={handleSubmit}
          disabled={isRunning || !models.length || !versions.length}
          className="px-5 py-2 rounded-md bg-indigo-600 text-white text-sm font-semibold hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isRunning ? 'Running...' : 'Run Eval'}
        </button>
      </div>

      {isRunning && (
        <p className="text-xs text-zinc-500">
          Evaluation in progress — 20–60s per model/version combination...
        </p>
      )}
    </div>
  )
}
