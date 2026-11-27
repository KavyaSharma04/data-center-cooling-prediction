"""
Data Center Cooling Intelligence - Interactive Dashboard Launcher
Starts a local web server and automatically opens dashboard.html in your default web browser.
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8050

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Prevent caching for live development
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    dashboard_file = "dashboard.html"
    
    if not os.path.exists(dashboard_file):
        print(f"Error: {dashboard_file} not found in the current directory.")
        sys.exit(1)
        
    url = f"http://localhost:{PORT}/{dashboard_file}"
    
    print("=" * 65)
    print(" DATA CENTER COOLING INTELLIGENCE DASHBOARD")
    print("=" * 65)
    print(f"Starting local dashboard server at: {url}")
    print("Press Ctrl+C at any time to stop the server.")
    print("=" * 65)
    
    # Open browser automatically
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Note: Could not open browser automatically ({e}). Please navigate to {url} manually.")
        
    try:
        with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
            print(f"Server is listening on port {PORT}...")
            httpd.serve_forever()
    except OSError:
        # If port 8050 is in use, try port 8051
        fallback_port = 8051
        url_fb = f"http://localhost:{fallback_port}/{dashboard_file}"
        print(f"Port {PORT} in use, trying {fallback_port}...")
        webbrowser.open(url_fb)
        with socketserver.TCPServer(("", fallback_port), CustomHandler) as httpd:
            print(f"Server is listening on port {fallback_port}...")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")

if __name__ == "__main__":
    main()
