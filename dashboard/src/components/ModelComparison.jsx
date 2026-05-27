import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const MODEL_COLORS = { haiku: '#6366f1', sonnet: '#22d3ee', opus: '#a78bfa' }

export default function ModelComparison({ results }) {
  const models = [...new Set(results.map(r => r.model))]

  if (models.length < 2) {
    return (
      <div className="rounded-lg bg-zinc-900 border border-zinc-800 p-5 flex flex-col justify-center items-center h-64">
        <p className="text-sm text-zinc-500">Run multiple models to see comparison</p>
      </div>
    )
  }

  const data = models.map(model => {
    const mr = results.filter(r => r.model === model)
    const passed = mr.reduce((sum, r) => sum + Object.values(r.metric_pass).filter(Boolean).length, 0)
    const total = mr.reduce((sum, r) => sum + Object.values(r.metric_pass).length, 0)
    return { model, pass_rate: total ? passed / total : 0 }
  })

  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 p-5">
      <h2 className="text-sm font-semibold text-white mb-0.5">Model Comparison</h2>
      <p className="text-xs text-zinc-400 mb-4">Pass rate by model</p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis dataKey="model" tick={{ fill: '#a1a1aa', fontSize: 11 }} />
          <YAxis domain={[0, 1]} tick={{ fill: '#a1a1aa', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', borderRadius: 6 }}
            labelStyle={{ color: '#f4f4f5', fontSize: 12 }}
            itemStyle={{ fontSize: 11 }}
            formatter={v => `${(v * 100).toFixed(1)}%`}
          />
          <Bar dataKey="pass_rate" radius={[2, 2, 0, 0]}>
            {data.map(entry => (
              <Cell key={entry.model} fill={MODEL_COLORS[entry.model] ?? '#6366f1'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
