import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const COLORS = ['#6366f1', '#22d3ee', '#a78bfa', '#34d399', '#f59e0b', '#f87171']

export default function RegressionChart({ results }) {
  const versions = [...new Set(results.map(r => r.prompt_version))]

  if (versions.length < 2) {
    return (
      <div className="rounded-lg bg-zinc-900 border border-zinc-800 p-5 flex flex-col justify-center items-center h-64">
        <p className="text-sm text-zinc-500">Run multiple prompt versions to see regression comparison</p>
      </div>
    )
  }

  const metricNames = [...new Set(results.flatMap(r => Object.keys(r.metric_scores)))]

  const data = versions.map(version => {
    const vr = results.filter(r => r.prompt_version === version)
    const entry = { version }
    for (const metric of metricNames) {
      const scores = vr.map(r => r.metric_scores[metric]).filter(s => s !== undefined)
      entry[metric] = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0
    }
    return entry
  })

  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 p-5">
      <h2 className="text-sm font-semibold text-white mb-0.5">Prompt Regression</h2>
      <p className="text-xs text-zinc-400 mb-4">Avg metric score by prompt version</p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis dataKey="version" tick={{ fill: '#a1a1aa', fontSize: 11 }} />
          <YAxis domain={[0, 1]} tick={{ fill: '#a1a1aa', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', borderRadius: 6 }}
            labelStyle={{ color: '#f4f4f5', fontSize: 12 }}
            itemStyle={{ fontSize: 11 }}
            formatter={v => v.toFixed(3)}
          />
          <Legend wrapperStyle={{ fontSize: 11, color: '#a1a1aa' }} />
          {metricNames.map((metric, i) => (
            <Bar key={metric} dataKey={metric} fill={COLORS[i % COLORS.length]} radius={[2, 2, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
