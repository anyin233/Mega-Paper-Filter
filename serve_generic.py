#!/usr/bin/env python3
import http.server
import socketserver
import os
import webbrowser
import argparse
from threading import Timer

def open_browser(port, filename):
    webbrowser.open(f'http://localhost:{port}/{filename}')

def main():
    parser = argparse.ArgumentParser(description='Serve clustering visualization webpage')
    parser.add_argument('--port', type=int, default=8000, help='Port to serve on (default: 8000)')
    parser.add_argument('--dir', default='.', help='Directory to serve from (default: current)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    parser.add_argument('--html', default='clustering_visualization.html', help='HTML file to open')
    
    args = parser.parse_args()
    
    # Change to the specified directory
    if args.dir != '.':
        os.chdir(args.dir)
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    print(f"Starting server at http://localhost:{args.port}")
    print(f"Serving files from: {os.getcwd()}")
    print(f"Main page: {args.html}")
    
    # Check if required files exist
    if os.path.exists(args.html):
        print(f"✓ {args.html} found")
    else:
        print(f"⚠ {args.html} not found in current directory")
    
    if os.path.exists('clustering_data.json'):
        print("✓ clustering_data.json found (default dataset)")
    else:
        print("⚠ clustering_data.json not found (you can upload your own dataset)")
    
    print("\nPress Ctrl+C to stop the server")
    
    # Open browser after 1 second
    if not args.no_browser:
        Timer(1, lambda: open_browser(args.port, args.html)).start()
    
    try:
        with socketserver.TCPServer(("", args.port), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    main()