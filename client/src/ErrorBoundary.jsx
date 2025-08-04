import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ 
          padding: '20px', 
          textAlign: 'center',
          backgroundColor: '#f8f8f8',
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center'
        }}>
          <h1 style={{ color: '#d32f2f' }}>🔧 Something went wrong</h1>
          <p style={{ marginBottom: '20px', fontSize: '16px' }}>
            There was an error loading the F&O Prediction Dashboard.
          </p>
          <button 
            onClick={() => window.location.reload()}
            style={{
              padding: '10px 20px',
              backgroundColor: '#1976d2',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '16px'
            }}
          >
            Reload Page
          </button>
          <details style={{ marginTop: '20px', maxWidth: '600px' }}>
            <summary style={{ cursor: 'pointer', marginBottom: '10px' }}>
              Show Technical Details
            </summary>
            <pre style={{ 
              backgroundColor: '#f0f0f0', 
              padding: '10px', 
              borderRadius: '5px',
              fontSize: '12px',
              overflow: 'auto',
              textAlign: 'left'
            }}>
              {this.state.error && this.state.error.toString()}
            </pre>
          </details>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
