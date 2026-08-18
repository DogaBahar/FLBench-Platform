import React, { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function Results() {
  const [runs, setRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState(null);
  const [runData, setRunData] = useState(null);
  const [isConfigOpen, setIsConfigOpen] = useState(false);

  useEffect(() => {
    fetch('http://localhost:5001/api/runs')
      .then(res => res.json())
      .then(data => setRuns(data))
      .catch(err => console.error("Failed to fetch runs:", err));
  }, []);

  const loadRunDetails = (runId) => {
    setSelectedRun(runId);
    fetch(`http://localhost:5001/api/runs/${runId}`)
      .then(res => res.json())
      .then(data => {
        // --- 0. Initialize Safe Defaults ---
        data.chartData = [];
        data.sysChartData = [];
        data.serverChartData = [];
        data.distChartData = [];
        data.distKeys = [];

        // --- 1. Process Global ML Metrics (Accuracy & Loss) ---
        if (data.global_metrics && data.global_metrics.metrics_distributed) {
          const accData = data.global_metrics.metrics_distributed.accuracy || [];
          const lossData = data.global_metrics.losses_distributed || [];
          const stdData = data.global_metrics.fairness_metrics?.accuracy_std || [];
          const worstData = data.global_metrics.fairness_metrics?.worst_client_accuracy || [];
          const stdByRound = Object.fromEntries(stdData.map(([r, v]) => [r, v]));
          const worstByRound = Object.fromEntries(worstData.map(([r, v]) => [r, v]));
          data.chartData = accData.map((item, index) => ({
            round: item[0],
            accuracy: item[1] * 100, // Convert to percentage
            loss: lossData[index] ? lossData[index][1] : 0,
            accuracyStd: (stdByRound[item[0]] || 0) * 100,
            worstClientAccuracy: item[0] in worstByRound ? worstByRound[item[0]] * 100 : null
          }));
        }

        // --- 2. Process Client Hardware Telemetry per Round ---
        if (data.client_logs && Array.isArray(data.client_logs)) {
          const sysByRound = {};
          
          data.client_logs.forEach(client => {
            if (!client.logs) return;
            
            client.logs.forEach(log => {
              if (log.action === 'fit') {
                const r = log.round || 1;
                if (!sysByRound[r]) {
                  sysByRound[r] = { round: r, count: 0, cpu: 0, ram: 0, time: 0, comm: 0, iowait: 0 };
                }
                sysByRound[r].cpu += log.cpu_usage_percent || 0;
                sysByRound[r].ram += log.peak_memory_mb || 0;
                sysByRound[r].time += log.compute_time_seconds || 0;
                sysByRound[r].comm += log.comm_size_mb || 0;
                sysByRound[r].iowait += log.iowait || 0;
                sysByRound[r].count += 1;
              }
            });
          });

          let runningNetworkTotal = 0; 
          
          data.sysChartData = Object.values(sysByRound)
            .sort((a, b) => a.round - b.round) // Ensure rounds are in order
            .map(d => {
              const avgCommForRound = parseFloat((d.comm / d.count).toFixed(2));
              runningNetworkTotal += avgCommForRound; 
              
              return {
                round: d.round,
                avgCpu: parseFloat((d.cpu / d.count).toFixed(2)),
                avgRam: parseFloat((d.ram / d.count).toFixed(2)),
                avgTime: parseFloat((d.time / d.count).toFixed(2)),
                avgComm: avgCommForRound,
                cumulativeComm: parseFloat(runningNetworkTotal.toFixed(2)),
                avgIoWait: parseFloat((d.iowait / d.count).toFixed(4)), 
              };
            });
        }

        // --- 3. Process Server Telemetry ---
        if (data.server_logs && Array.isArray(data.server_logs) && data.server_logs.length > 0) {
          data.serverChartData = data.server_logs.map(log => ({
            round: log.round,
            tcpEst: log.tcp_established || 0,
            tcpWait: log.tcp_time_wait || 0,
            aggTime: parseFloat((log.aggregation_time_sec || 0).toFixed(2)),
            ioWait: parseFloat((log.iowait_time || 0).toFixed(4))
          })).sort((a, b) => a.round - b.round);
        }
        
        // --- 4. Process Data Distributions ---
        if (data.distributions && Object.keys(data.distributions).length > 0) {
          data.distChartData = Object.keys(data.distributions).map(clientId => {
            const counts = data.distributions[clientId];
            const formatted = { client: clientId.substring(0, 8) + "..." }; 
            Object.keys(counts).forEach(key => {
              formatted[`class_${key}`] = counts[key];
            });
            return formatted;
          });
          
          const allKeys = new Set();
          Object.values(data.distributions).forEach(clientObj => {
            Object.keys(clientObj).forEach(k => allKeys.add(`class_${k}`));
          });
          data.distKeys = Array.from(allKeys);
        }
        
        setRunData(data);
      })
      .catch(err => console.error("Failed to load run details:", err));
  };

  // --- NEW: Delete Run Function ---
  const handleDelete = async (e, runId) => {
    e.stopPropagation(); // Prevents loading the run when clicking delete
    
    if (!window.confirm(`Are you sure you want to permanently delete run: ${runId}?`)) return;

    try {
      const response = await fetch(`http://localhost:5001/api/runs/${runId}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        // Remove from list
        setRuns(runs.filter((run) => run !== runId));
        
        // If the user deleted the run they were currently viewing, clear the dashboard
        if (selectedRun === runId) {
          setSelectedRun(null);
          setRunData(null);
        }
      } else {
        console.error('Failed to delete the run');
        alert("Failed to delete the run from the server.");
      }
    } catch (error) {
      console.error('Error deleting run:', error);
    }
  };

  const getAvgClientMetric = (logs, key) => {
    if (!logs || !Array.isArray(logs) || logs.length === 0) return "0.00";
    const allValues = logs.flatMap(client => (client.logs || []).map(l => l[key] || 0));
    if (allValues.length === 0) return "0.00";
    return (allValues.reduce((a, b) => a + b, 0) / allValues.length).toFixed(2);
  };

  // Gap between the global (weighted-avg) accuracy and the single worst
  // client's accuracy on the final round -- a quick fairness-under-non-IID
  // read that a single averaged accuracy number can't show.
  const getFairnessGap = () => {
    const rounds = runData?.chartData;
    if (!rounds || rounds.length === 0) return null;
    const last = rounds[rounds.length - 1];
    if (last.worstClientAccuracy === null || last.worstClientAccuracy === undefined) return null;
    return (last.accuracy - last.worstClientAccuracy).toFixed(2);
  };

  return (
    <div>
      <h2>Benchmark History</h2>
      <div style={{ display: 'flex', gap: '2rem' }}>
        
        {/* Sidebar (UPDATED: Scrollable and Delete Buttons) */}
        <div style={{ width: '280px', borderRight: '1px solid #ccc', paddingRight: '1rem', maxHeight: '85vh', overflowY: 'auto' }}>
          {runs.length === 0 ? <p>No runs found.</p> : runs.map(run => (
            <div key={run} style={{ display: 'flex', marginBottom: '0.5rem', gap: '0.5rem' }}>
              <button 
                onClick={() => loadRunDetails(run)}
                style={{ 
                  flex: 1, padding: '0.5rem', cursor: 'pointer', textAlign: 'left',
                  background: selectedRun === run ? '#0070f3' : '#f0f0f0', 
                  color: selectedRun === run ? 'white' : 'black', 
                  border: 'none', borderRadius: '4px', overflow: 'hidden', textOverflow: 'ellipsis'
                }}
                title={run}
              >
                {run}
              </button>
              <button
                onClick={(e) => handleDelete(e, run)}
                style={{
                  padding: '0.5rem 0.75rem', cursor: 'pointer', background: '#ff4d4f', 
                  color: 'white', border: 'none', borderRadius: '4px', fontWeight: 'bold'
                }}
                title="Delete Run"
              >
                ✕
              </button>
            </div>
          ))}
        </div>

        {/* Main Content */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {!runData ? (
            <p>Select a run to view results.</p>
          ) : (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '2px solid #eee', paddingBottom: '0.5rem', marginBottom: '1.5rem' }}>
                <h3 style={{ margin: 0 }}>Run Details: {selectedRun}</h3>
                <div style={{ display: 'flex', gap: '1.25rem' }}>
                  <a
                    href={`http://localhost:5001/api/runs/${selectedRun}/export`}
                    style={{ color: '#10b981', textDecoration: 'none', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.3rem', fontWeight: 'bold' }}
                    title="Download this run as a results/ submission file to contribute via PR"
                  >
                    <span>Export for results/ ↓</span>
                  </a>
                  <a
                    href={`http://localhost:5001/api/runs/${selectedRun}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: '#0070f3', textDecoration: 'none', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.3rem', fontWeight: 'bold' }}
                  >
                    <span>View Raw JSON Payload ↗</span>
                  </a>
                </div>
              </div>
              
              {/* Top Level Summary Cards */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
                <div style={{ background: '#f8f9fa', padding: '1rem', borderRadius: '8px', borderLeft: '4px solid #0070f3' }}>
                  <div style={{ fontSize: '0.8rem', color: '#666' }}>Total Wall-Clock Time</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{runData.global_metrics?.wall_clock_time_seconds || 0}s</div>
                </div>
                <div style={{ background: '#f8f9fa', padding: '1rem', borderRadius: '8px', borderLeft: '4px solid #10b981' }}>
                  <div style={{ fontSize: '0.8rem', color: '#666' }}>Avg Client Payload</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{getAvgClientMetric(runData.client_logs, 'comm_size_mb')} MB</div>
                </div>
                <div style={{ background: '#f8f9fa', padding: '1rem', borderRadius: '8px', borderLeft: '4px solid #f59e0b' }}>
                  <div style={{ fontSize: '0.8rem', color: '#666' }}>Avg Client RAM Peak</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{getAvgClientMetric(runData.client_logs, 'peak_memory_mb')} MB</div>
                </div>
                <div style={{ background: '#f8f9fa', padding: '1rem', borderRadius: '8px', borderLeft: '4px solid #ef4444' }}>
                  <div style={{ fontSize: '0.8rem', color: '#666' }}>Avg CPU Usage</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{getAvgClientMetric(runData.client_logs, 'cpu_usage_percent')}%</div>
                </div>
                {getFairnessGap() !== null && (
                  <div style={{ background: '#f8f9fa', padding: '1rem', borderRadius: '8px', borderLeft: '4px solid #ec4899' }}>
                    <div style={{ fontSize: '0.8rem', color: '#666' }}>Accuracy Fairness Gap (Last Round)</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{getFairnessGap()} pp</div>
                  </div>
                )}
              </div>

              {/* Chart Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
                
                <div style={{ height: '300px' }}>
                  <h4 style={{ margin: '0 0 1rem 0' }}>Global Accuracy (%)</h4>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={runData.chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis domain={[0, 100]}/>
                      <Tooltip />
                      <Line type="monotone" dataKey="accuracy" stroke="#0070f3" strokeWidth={3} activeDot={{ r: 6 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div style={{ height: '300px' }}>
                  <h4 style={{ margin: '0 0 1rem 0' }}>Client Accuracy Fairness (%)</h4>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={runData.chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis domain={[0, 100]}/>
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" name="Global Avg" dataKey="accuracy" stroke="#0070f3" strokeWidth={2} />
                      <Line type="monotone" name="Worst Client" dataKey="worstClientAccuracy" stroke="#ec4899" strokeWidth={2} strokeDasharray="5 5" connectNulls />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div style={{ height: '300px' }}>
                  <h4 style={{ margin: '0 0 1rem 0' }}>Avg Client Compute Time (s)</h4>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={runData.sysChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="avgTime" fill="#8884d8" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div style={{ height: '300px' }}>
                  <h4 style={{ margin: '0 0 1rem 0' }}>Network Payload per Round (MB)</h4>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={runData.sysChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis />
                      <Tooltip />
                      <Area type="monotone" dataKey="avgComm" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                <div style={{ height: '300px' }}>
                  <h4 style={{ margin: '0 0 1rem 0' }}>Avg Peak RAM (MB)</h4>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={runData.sysChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="avgRam" stroke="#f59e0b" strokeWidth={3} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                
                <div style={{ height: '300px' }}>
                  <h4 style={{ margin: '0 0 1rem 0' }}>Server TCP Sockets (Concurrency)</h4>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={runData.serverChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Area type="monotone" name="Established" dataKey="tcpEst" stackId="1" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
                      <Area type="monotone" name="Time Wait" dataKey="tcpWait" stackId="1" stroke="#ef4444" fill="#ef4444" fillOpacity={0.6} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                <div style={{ height: '300px' }}>
                  <h4 style={{ margin: '0 0 1rem 0' }}>Server Aggregation Time (s)</h4>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={runData.serverChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis />
                      <Tooltip />
                      <Bar name="Aggregation Math + I/O" dataKey="aggTime" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

              </div>

              {/* Data Distribution Topology Chart */}
              {runData.distChartData.length > 0 && (
                <div style={{ height: '300px', marginBottom: '2rem' }}> 
                  <h4 style={{ margin: '0 0 1rem 0' }}>Data Distribution Topology (Samples per Class)</h4>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={runData.distChartData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis dataKey="client" type="category" width={80} />
                      <Tooltip />
                      {runData.distKeys?.map((key, index) => (
                        <Bar 
                          key={key} 
                          dataKey={key} 
                          stackId="a" 
                          fill={`hsl(${(index * 137.5) % 360}, 70%, 50%)`} 
                        />
                      ))}
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Collapsible Configuration Panel */}
              <div style={{ border: '1px solid #e5e7eb', borderRadius: '8px', overflow: 'hidden', backgroundColor: '#ffffff' }}>
                <button 
                  onClick={() => setIsConfigOpen(!isConfigOpen)}
                  style={{ 
                    width: '100%', padding: '1rem 1.5rem', background: '#f9fafb', border: 'none', 
                    textAlign: 'left', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', 
                    alignItems: 'center', fontSize: '1.1rem', fontWeight: 'bold', color: '#374151'
                  }}
                >
                  <span>Raw Configuration Parameters</span>
                  <span style={{ transition: 'transform 0.2s', transform: isConfigOpen ? 'rotate(90deg)' : 'rotate(0deg)' }}>
                    ▶
                  </span>
                </button>

                {isConfigOpen && (
                  <div style={{ padding: '1.5rem', background: '#1e1e1e', color: '#d4d4d4', margin: 0, borderTop: '1px solid #e5e7eb', overflowX: 'auto' }}>
                    <pre style={{ margin: 0, fontFamily: 'monospace', fontSize: '0.9rem' }}>
                      {JSON.stringify(runData.full_config || runData.config, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
              
            </div>
          )}
        </div>
      </div>
    </div>
  );
}