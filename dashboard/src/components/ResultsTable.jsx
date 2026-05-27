const TASK_COLORS = {
  bug_identification: 'text-yellow-400',
  code_explanation: 'text-blue-400',
  code_review: 'text-purple-400',
  security_flagging: 'text-red-400',
}

export default function ResultsTable({ results }) {
  const rows = results.flatMap(r =>
    Object.entries(r.metric_scores).map(([metric, score]) => ({
      case_id: r.case_id,
      task_type: r.task_type,
      model: r.model,
      prompt_version: r.prompt_version,
      metric,
      score,
      pass: r.metric_pass[metric],
    }))
  )

  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 overflow-hidden">
      <div className="px-5 py-3 border-b border-zinc-800">
        <h2 className="text-sm font-semibold text-white">Results</h2>
        <p className="text-xs text-zinc-400 mt-0.5">{rows.length} metric checks</p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800 text-xs text-zinc-400 uppercase tracking-wider">
              {['Case ID', 'Task Type', 'Model', 'Version', 'Metric', 'Score', 'Pass'].map(h => (
                <th key={h} className="px-4 py-2 text-left font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-b border-zinc-800/50 hover:bg-zinc-800/30">
                <td className="px-4 py-2 font-mono text-xs text-zinc-300">{row.case_id}</td>
                <td className={`px-4 py-2 text-xs ${TASK_COLORS[row.task_type] ?? 'text-zinc-300'}`}>
                  {row.task_type}
                </td>
                <td className="px-4 py-2 text-xs text-zinc-300">{row.model}</td>
                <td className="px-4 py-2 text-xs text-zinc-400">{row.prompt_version}</td>
                <td className="px-4 py-2 text-xs text-zinc-300">{row.metric}</td>
                <td className="px-4 py-2 font-mono text-xs text-zinc-200">{row.score.toFixed(3)}</td>
                <td className="px-4 py-2">
                  <span className={`text-xs font-semibold ${row.pass ? 'text-green-400' : 'text-red-400'}`}>
                    {row.pass ? 'PASS' : 'FAIL'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
