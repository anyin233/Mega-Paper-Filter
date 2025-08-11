#!/usr/bin/env python3
import http.server
import socketserver
import os
import webbrowser
from threading import Timer

def open_browser():
    webbrowser.open('http://localhost:8000/clustering_visualization.html')

def main():
    # Change to the project directory
    os.chdir('/Users/cydia2001/Project/paper-labeler')
    
    PORT = 8000
    Handler = http.server.SimpleHTTPRequestHandler
    
    print(f"Starting server at http://localhost:{PORT}")
    print("Files available:")
    print("  - clustering_visualization.html")
    print("  - clustering_data.json")
    print("\nPress Ctrl+C to stop the server")
    
    # Open browser after 1 second
    Timer(1, open_browser).start()
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    main()