#!/usr/bin/env python3
import http.server
import socketserver
import os
import webbrowser
import argparse
from threading import Timer

def open_browser(port, filename):
    webbrowser.open(f'http://localhost:{port}/{filename}')

def find_json_files():
    """Find JSON files that look like clustering data."""
    json_files = []
    for file in os.listdir('.'):
        if file.endswith('.json') and ('data' in file or 'cluster' in file):
            json_files.append(file)
    return json_files

def main():
    parser = argparse.ArgumentParser(description='Serve clustering visualization webpage')
    parser.add_argument('json_file', nargs='?', help='JSON clustering data file to serve')
    parser.add_argument('--port', type=int, default=8000, help='Port to serve on (default: 8000)')
    parser.add_argument('--dir', default='.', help='Directory to serve from (default: current)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    parser.add_argument('--html', default='clustering_visualization.html', help='HTML file to open')
    
    args = parser.parse_args()
    
    # Change to the specified directory
    if args.dir != '.':
        os.chdir(args.dir)
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    print(f"🚀 Starting clustering visualization server")
    print(f"📍 Server: http://localhost:{args.port}")
    print(f"📁 Serving from: {os.getcwd()}")
    print(f"🌐 Main page: {args.html}")
    print("-" * 50)
    
    # Check if required files exist
    if os.path.exists(args.html):
        print(f"✓ {args.html} found")
    else:
        print(f"❌ {args.html} not found - webpage will not load!")
        return
    
    # Handle JSON data file
    json_file_to_use = None
    if args.json_file:
        if os.path.exists(args.json_file):
            json_file_to_use = args.json_file
            print(f"✓ Using specified JSON file: {args.json_file}")
        else:
            print(f"❌ Specified JSON file not found: {args.json_file}")
            return
    else:
        # Look for JSON files automatically
        json_files = find_json_files()
        if json_files:
            json_file_to_use = json_files[0]
            print(f"✓ Found JSON data file: {json_file_to_use}")
            if len(json_files) > 1:
                print(f"ℹ️  Other JSON files found: {', '.join(json_files[1:])}")
        else:
            print("⚠️  No clustering JSON files found")
            print("   You can upload your own file using the web interface")
    
    # If we have a JSON file, copy it to the default name for auto-loading
    if json_file_to_use and json_file_to_use != 'clustering_data.json':
        try:
            with open(json_file_to_use, 'r') as src, open('clustering_data.json', 'w') as dst:
                dst.write(src.read())
            print(f"✓ Copied {json_file_to_use} to clustering_data.json for auto-loading")
        except Exception as e:
            print(f"⚠️  Could not copy JSON file: {e}")
    
    print("-" * 50)
    print("📊 Visualization Features:")
    print("   • Interactive scatter plot and network graph")
    print("   • Cluster filtering and search")
    print("   • Paper details on click")
    print("   • Drag & drop JSON file upload")
    print("-" * 50)
    print("\n🌐 Open http://localhost:{} in your browser".format(args.port))
    print("⌨️  Press Ctrl+C to stop the server\n")
    
    # Open browser after 1 second
    if not args.no_browser:
        Timer(1, lambda: open_browser(args.port, args.html)).start()
    
    try:
        with socketserver.TCPServer(("", args.port), Handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        # Clean up temp file
        if os.path.exists('clustering_data.json') and json_file_to_use and json_file_to_use != 'clustering_data.json':
            try:
                os.remove('clustering_data.json')
                print(f"\n🧹 Cleaned up temporary file: clustering_data.json")
            except:
                pass
        print("🛑 Server stopped.")

if __name__ == "__main__":
    main()