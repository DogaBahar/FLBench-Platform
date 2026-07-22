from flask import Flask
from flask_cors import CORS
from api.routes import bp as api_bp
from core.database import db_session

def create_app():
    app = Flask(__name__)

    # Enable CORS for all routes so the Vite React frontend can communicate with Flask
    CORS(app)

    # Register blueprints (mapped to /api to match Dashboard.jsx and Results.jsx)
    app.register_blueprint(api_bp, url_prefix='/api')

    # Schema creation/updates are now handled by Alembic migrations
    # (see scripts/migrate.py, run from docker-entrypoint.sh before this
    # process starts) rather than by Base.metadata.create_all() here.

    # Ensure database sessions are cleanly removed after each request
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        db_session.remove()

    return app

if __name__ == "__main__":
    app = create_app()
    # Run on port 5001 to match the fetch calls in the React frontend
    app.run(host="0.0.0.0", port=5001, debug=True)