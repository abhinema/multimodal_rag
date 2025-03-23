import yaml
import os
import uvicorn
from src.rag.retriever import MultimodalRAG
from src.api.endpoints import app

def load_config():
    """Load configuration from YAML file."""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def setup_directories():
    """Create necessary directories."""
    os.makedirs('temp', exist_ok=True)
    os.makedirs('temp/vectors', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)

def main():
    """Main entry point for the application."""
    # Load configuration
    config = load_config()
    
    # Setup directories
    setup_directories()
    
    # Initialize RAG system
    rag_system = MultimodalRAG(config)
    
    # Update the global variable in the API module
    import src.api.endpoints
    src.api.endpoints.rag_system = rag_system
    
    # Mount static files for UI
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
    
    # Start API server
    uvicorn.run(
        app,
        host=config['api']['host'],
        port=config['api']['port'],
        log_level="info"
    )

if __name__ == "__main__":
    main()