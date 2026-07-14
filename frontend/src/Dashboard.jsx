import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function Dashboard() {
  // 1. Core Platform State
  const [status, setStatus] = useState('Idle');
  const [isDeploying, setIsDeploying] = useState(false); // NEW: Prevents double-clicks
  const [numClients, setNumClients] = useState(3);
  const [framework, setFramework] = useState('flower');
  const navigate = useNavigate();

  // 2. The Unified Configuration Matrix
  const [config, setConfig] = useState({
    ml_hyperparameters: {
      epochs: 2,
      batch_size: 32,
      learning_rate: 0.001,
      optimizer: 'adam'
    },
    federated_settings: {
      rounds: 3,
      strategy: 'FedAvg',
      fraction_fit: 1.0
    },
    data_simulation: {
      dataset: 'cifar100',
      samples_per_client: 500,
      partition_strategy: 'iid',
      alpha: 0.5,
      shards_per_client: 2
    }
  });

  // Helper to cleanly update nested configuration state
  const updateConfig = (category, field, value) => {
    setConfig(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [field]: value
      }
    }));
  };

  // 3. Trigger the Orchestrator
  const startBenchmark = async () => {
    if (isDeploying) return; // Block multiple rapid clicks
    
    setIsDeploying(true);
    setStatus('Deploying Tasks...');

    // 1. Create a deep copy of the config to protect the React UI state
    const cleanConfig = JSON.parse(JSON.stringify(config));
    const strategy = cleanConfig.data_simulation.partition_strategy;

    cleanConfig.federated_settings.target_clients = numClients; 
    // 2. Aggressively strip out irrelevant parameters based on the chosen strategy
    if (strategy === 'iid') {
      delete cleanConfig.data_simulation.alpha;
      delete cleanConfig.data_simulation.shards_per_client;
    } else if (strategy === 'dirichlet') {
      delete cleanConfig.data_simulation.shards_per_client;
    } else if (strategy === 'shard') {
      delete cleanConfig.data_simulation.alpha;
    }
    
    try {
      const response = await fetch('http://localhost:5001/api/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          framework,
          clients: numClients,
          config: cleanConfig 
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        const runId = data.run_id;
        
        setStatus('Training Running... Please wait.');
        
        // 3. The Polling Mechanism (UPDATED)
        const pollInterval = setInterval(async () => {
          try {
            // Check the status of this specific run
            const checkRes = await fetch(`http://localhost:5001/api/runs/${runId}`);
            if (checkRes.ok) {
              const checkData = await checkRes.json();
              
              // Only navigate when the backend explicitly says it is DONE
              if (checkData.status === 'COMPLETED') {
                clearInterval(pollInterval); // Stop asking
                setStatus('Finished! Redirecting...');
                setIsDeploying(false); // Reset button
                
                setTimeout(() => {
                  navigate('/results');
                }, 1500);
              } 
              // Handle backend failures gracefully
              else if (checkData.status === 'FAILED') {
                clearInterval(pollInterval);
                setStatus('Benchmark Failed. Check Docker Logs.');
                setIsDeploying(false);
                alert("The benchmark failed during execution. Please check your backend terminal logs.");
              }
            }
          } catch (err) {
            console.error("Polling error:", err);
          }
        }, 3000); // Check every 3 seconds

      } else {
        let message = 'Error deploying tasks.';
        try {
          const errData = await response.json();
          if (errData?.error?.message) {
            message = errData.error.message;
          }
        } catch (_) {
          // response body wasn't JSON -- keep the generic message
        }
        setStatus(`Error: ${message}`);
        setIsDeploying(false);
      }
    } catch (error) {
      console.error("Failed to start benchmark:", error);
      setStatus('Connection Failed. Is Flask running?');
      setIsDeploying(false);
    }
  };

  // 4. UI Rendering
  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: '1000px', margin: '0 auto', padding: '2rem' }}>
      <header style={{ borderBottom: '2px solid #eaeaea', paddingBottom: '1rem', marginBottom: '2rem' }}>
        <h1 style={{ margin: 0, color: '#333' }}>Federated Learning Benchmark Platform</h1>
        <p style={{ margin: '0.5rem 0 0 0', color: '#666' }}>Distributed Task Orchestrator Control Panel</p>
      </header>

      <div style={{ display: 'grid', gap: '2rem' }}>
        
        {/* === CORE INFRASTRUCTURE === */}
        <section style={{ padding: '1.5rem', background: '#f8f9fa', borderRadius: '8px' }}>
          <h2 style={{ marginTop: 0, fontSize: '1.2rem' }}>1. Infrastructure</h2>
          <div style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
            <label>
              <strong>Framework:</strong> <br/>
              <select value={framework} onChange={e => setFramework(e.target.value)} disabled={isDeploying} style={{ padding: '0.5rem', marginTop: '0.25rem' }}>
                <option value="flower">Flower (flwr)</option>
                <option value="nvflare">NVIDIA FLARE </option>
                <option value="fedml">FedML</option>
              </select>
            </label>
            <label>
              <strong>Target Clients:</strong> <br/>
              <input 
                type="number" 
                min="1" 
                max="10" 
                value={numClients} 
                onChange={e => setNumClients(parseInt(e.target.value))} 
                disabled={isDeploying}
                style={{ padding: '0.5rem', marginTop: '0.25rem', width: '80px' }}
              />
            </label>
          </div>
        </section>

        {/* === FEDERATED SETTINGS === */}
        <section style={{ padding: '1.5rem', background: '#eef2ff', borderRadius: '8px' }}>
          <h2 style={{ marginTop: 0, fontSize: '1.2rem' }}>2. Federated Network Settings</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '2rem' }}>
            <label>
              <strong>Global Rounds:</strong> <br/>
              <input type="number" value={config.federated_settings.rounds} disabled={isDeploying} onChange={e => updateConfig('federated_settings', 'rounds', parseInt(e.target.value))} style={{ padding: '0.5rem', marginTop: '0.25rem', width: '100%' }}/>
            </label>
            <label>
              <strong>Strategy:</strong> <br/>
              <select value={config.federated_settings.strategy} disabled={isDeploying} onChange={e => updateConfig('federated_settings', 'strategy', e.target.value)} style={{ padding: '0.5rem', marginTop: '0.25rem', width: '100%' }}>
                <option value="FedAvg">FedAvg</option>
              </select>
            </label>
            <label>
              <strong>Fraction Fit:</strong> <br/>
              <input type="number" step="0.1" max="1" min="0.1" disabled={isDeploying} value={config.federated_settings.fraction_fit} onChange={e => updateConfig('federated_settings', 'fraction_fit', parseFloat(e.target.value))} style={{ padding: '0.5rem', marginTop: '0.25rem', width: '80%' }}/>
            </label>
          </div>
        </section>

        {/* === ML HYPERPARAMETERS === */}
        <section style={{ padding: '1.5rem', background: '#fdf4ff', borderRadius: '8px' }}>
          <h2 style={{ marginTop: 0, fontSize: '1.2rem' }}>3. Local ML Hyperparameters (PyTorch)</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '2rem' }}>
            <label>
              <strong>Local Epochs:</strong> <br/>
              <input type="number" value={config.ml_hyperparameters.epochs} disabled={isDeploying} onChange={e => updateConfig('ml_hyperparameters', 'epochs', parseInt(e.target.value))} style={{ padding: '0.5rem', marginTop: '0.25rem', width: '100%' }}/>
            </label>
            <label>
              <strong>Batch Size:</strong> <br/>
              <input type="number" value={config.ml_hyperparameters.batch_size} disabled={isDeploying} onChange={e => updateConfig('ml_hyperparameters', 'batch_size', parseInt(e.target.value))} style={{ padding: '0.5rem', marginTop: '0.25rem', width: '100%' }}/>
            </label>
            <label>
              <strong>Learning Rate:</strong> <br/>
              <input type="number" step="0.001" value={config.ml_hyperparameters.learning_rate} disabled={isDeploying} onChange={e => updateConfig('ml_hyperparameters', 'learning_rate', parseFloat(e.target.value))} style={{ padding: '0.5rem', marginTop: '0.25rem', width: '100%' }}/>
            </label>
            <label>
              <strong>Optimizer:</strong> <br/>
              <select value={config.ml_hyperparameters.optimizer} disabled={isDeploying} onChange={e => updateConfig('ml_hyperparameters', 'optimizer', e.target.value)} style={{ padding: '0.5rem', marginTop: '0.25rem', width: '100%' }}>
                <option value="adam">Adam</option>
                <option value="sgd">SGD (Momentum 0.9)</option>
              </select>
            </label>
          </div>
        </section>

        {/* === DATA SIMULATION === */}
        <section style={{ padding: '1.5rem', background: '#f0fdf4', borderRadius: '8px' }}>
          <h2 style={{ marginTop: 0, fontSize: '1.2rem' }}>4. Data Simulation</h2>
          <div style={{ display: 'flex', gap: '2rem' }}>
            <label style={{ width: '100%', maxWidth: '200px' }}>
              <strong>Dataset:</strong> <br/>
              <select 
                value={config.data_simulation.dataset} 
                onChange={e => updateConfig('data_simulation', 'dataset', e.target.value)} 
                disabled={isDeploying}
                style={{ padding: '0.5rem', marginTop: '0.25rem', width: '100%' }}
              >
                <option value="cifar100">CIFAR-100</option>
                <option value="femnist">FEMNIST</option>
              </select>
            </label>
            <label style={{ width: '100%', maxWidth: '200px' }}>
              <strong>Samples per Client:</strong> <br/>
              <input type="number" step="100" disabled={isDeploying} value={config.data_simulation.samples_per_client} onChange={e => updateConfig('data_simulation', 'samples_per_client', parseInt(e.target.value))} style={{ padding: '0.5rem', marginTop: '0.25rem', width: '100%' }}/>
            </label>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', paddingTop: '1rem' }}>
            <label>
              <strong>Partition Strategy:</strong> <br/>
              <select value={config.data_simulation.partition_strategy} disabled={isDeploying} onChange={e => updateConfig('data_simulation', 'partition_strategy', e.target.value)} style={{ padding: '0.5rem', marginTop: '0.25rem', width: '100%' }}>
                <option value="iid">IID (Uniform)</option>
                <option value="shard">Pathological (Shards)</option>
                <option value="dirichlet">Statistical (Dirichlet)</option>
              </select>
            </label>
            
            {config.data_simulation.partition_strategy === 'dirichlet' && (
              <label>
                <strong>Dirichlet Alpha:</strong> <br/>
                <input type="number" step="0.1" disabled={isDeploying} value={config.data_simulation.alpha} onChange={e => updateConfig('data_simulation', 'alpha', parseFloat(e.target.value))} style={{ padding: '0.5rem', marginTop: '0.25rem', width: '100%' }}/>
              </label>
            )}

            {config.data_simulation.partition_strategy === 'shard' && (
              <label>
                <strong>Shards per Client:</strong> <br/>
                <input type="number" step="1" disabled={isDeploying} value={config.data_simulation.shards_per_client} onChange={e => updateConfig('data_simulation', 'shards_per_client', parseInt(e.target.value))} style={{ padding: '0.5rem', marginTop: '0.25rem', width: '100%' }}/>
              </label>
            )}
          </div>
        </section>

        {/* === EXECUTION CONTROLS === */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', marginTop: '1rem', borderTop: '2px solid #eaeaea', paddingTop: '2rem' }}>
          <button 
            onClick={startBenchmark}
            disabled={isDeploying}
            style={{ 
              padding: '1rem 2rem', 
              fontSize: '1.1rem', 
              backgroundColor: isDeploying ? '#9ca3af' : '#0070f3', // Gray out if running
              color: 'white', 
              border: 'none', 
              borderRadius: '6px', 
              cursor: isDeploying ? 'not-allowed' : 'pointer', 
              fontWeight: 'bold',
              transition: 'background-color 0.2s'
            }}
          >
            {isDeploying ? 'Running Benchmark...' : 'Deploy Tasks & Start Benchmark'}
          </button>
          <div style={{ fontWeight: 'bold', color: status.includes('Error') || status.includes('Failed') ? '#ef4444' : '#0070f3' }}>
            Status: {status}
          </div>
        </div>

      </div>
    </div>
  );
}

export default Dashboard;