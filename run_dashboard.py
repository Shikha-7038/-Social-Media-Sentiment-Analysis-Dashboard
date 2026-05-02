"""
Quick launcher for the sentiment analysis dashboard
"""

import subprocess
import webbrowser
import time
import sys

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     SOCIAL MEDIA SENTIMENT ANALYSIS DASHBOARD                ║
    ║                      Quick Launcher                          ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("Starting Streamlit dashboard...")
    print("Opening browser in a few seconds...")
    
    # Open browser after a delay
    def open_browser():
        time.sleep(2)
        webbrowser.open('http://localhost:8501')
    
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Launch streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/dashboard.py"])

if __name__ == "__main__":
    main()