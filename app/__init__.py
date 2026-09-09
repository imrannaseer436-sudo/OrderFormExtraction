"""Review-and-upload web app for the order-form extraction pipeline.

`server.py` is the entry point (see ../run_ui.py). This package is
deliberately thin: it wraps the existing pipeline (extract_ollama_cloud.py)
and the existing catalog matcher (brandlist_match.py) rather than
reimplementing any of their logic.
"""
