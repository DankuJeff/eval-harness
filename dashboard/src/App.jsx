import { useState } from 'react'
import RunButton from './components/RunButton'
import ResultsTable from './components/ResultsTable'
import RegressionChart from './components/RegressionChart'
import ModelComparison from './components/ModelComparison'

const API_URL = 'http://localhost:8001/run'

export default function App() {
  const [runData, setRunData] = useState(null)
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState(null)

  async function handleRun(params) {
    setIsRunning(true)
    setError(null)
    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(`API error ${res.status}: ${text}`)
      }
      setRunData(await res.json())
    } catch (err) {
      setError(err.message)
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <header>
          <h1 className="text-2xl font-semibold tracking-tight text-white">Eval Harness</h1>
          <p className="text-sm text-zinc-400 mt-1">LLM evaluation dashboard — code understanding &amp; review tasks</p>
        </header>

        <RunButton onRun={handleRun} isRunning={isRunning} />

        {error && (
          <div className="rounded-md bg-red-900/30 border border-red-700 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        )}

        {runData && (
          <>
            <div className="grid grid-cols-4 gap-4">
              {[
                { label: 'Cases', value: runData.summary.total_cases },
                { label: 'Metric Checks', value: runData.summary.total_metric_checks },
                { label: 'Passed', value: runData.summary.passed },
                { label: 'Pass Rate', value: `${(runData.summary.pass_rate * 100).toFixed(1)}%` },
              ].map(({ label, value }) => (
                <div key={label} className="rounded-lg bg-zinc-900 border border-zinc-800 p-4">
                  <div className="text-xs text-zinc-400 uppercase tracking-wider">{label}</div>
                  <div className="text-2xl font-semibold text-white mt-1">{value}</div>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-2 gap-6">
              <RegressionChart results={runData.results} />
              <ModelComparison results={runData.results} />
            </div>

            <ResultsTable results={runData.results} />
          </>
        )}
      </div>
    </div>
  )
}
