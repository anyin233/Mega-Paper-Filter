#!/usr/bin/env python3
import http.server
import socketserver
import os
import webbrowser
from threading import Timer

def open_browser():
    webbrowser.open('http://localhost:8000/clustering_visualization.html')

def main():
    PORT = 8000
    Handler = http.server.SimpleHTTPRequestHandler
    
    print(f"🚀 Starting static clustering visualization server")
    print(f"📍 Server: http://localhost:{PORT}")
    print(f"📁 Serving from: {os.getcwd()}")
    print("-" * 50)
    
    # Check if required files exist
    if os.path.exists('clustering_visualization.html'):
        print("✓ clustering_visualization.html found")
    else:
        print("❌ clustering_visualization.html not found!")
        print("   Please make sure you're running from the correct directory")
        return
        
    if os.path.exists('clustering_data.json'):
        print("✓ clustering_data.json found (will auto-load)")
    else:
        print("⚠️  clustering_data.json not found")
        print("   You can upload your own file using the web interface")
    
    print("-" * 50)
    print("🌐 Opening http://localhost:{} in your browser".format(PORT))
    print("⌨️  Press Ctrl+C to stop the server\n")
    
    # Open browser after 1 second
    Timer(1, open_browser).start()
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")

if __name__ == "__main__":
    main()