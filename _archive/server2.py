#!/usr/bin/env python3
"""
live_preview.py

A simple livereload server for Jupyter Notebook HTML files.
Runs a server on the specified port and reloads the browser
whenever the HTML file changes.
"""

import argparse
import os
from livereload import Server

def main():
    parser = argparse.ArgumentParser(
        description="Live-reload preview for Jupyter Notebook HTML"
    )
    parser.add_argument(
        "filename",
        help="Path to your notebook file (either .ipynb or .html)"
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8000,
        help="Port number to serve on (default: 8000)"
    )
    args = parser.parse_args()

    # Determine HTML file path
    notebook_path = args.filename
    if notebook_path.endswith(".ipynb"):
        html_path = notebook_path[:-6] + ".html"
    else:
        html_path = notebook_path

    if not os.path.isfile(html_path):
        print(f"Error: HTML file not found: {html_path}")
        exit(1)

    # Start livereload server
    server = Server()
    server.watch(html_path)
    serve_dir = os.path.dirname(os.path.abspath(html_path)) or "."
    print(f"Serving {html_path} on http://localhost:{args.port}")
    server.serve(root=serve_dir, port=args.port)

if __name__ == "__main__":
    main()

