import React from 'react'
import ReactDOM from 'react-dom/client'
// This is the line that was missing! It tells main.jsx where to find 'App'
import App from './App.jsx' 

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)