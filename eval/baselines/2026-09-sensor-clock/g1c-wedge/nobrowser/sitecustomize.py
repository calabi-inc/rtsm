"""Put this directory on PYTHONPATH for the rtsm process under test: replaces webbrowser.open with a
no-op so `rtsm/utils/browser.open_browser` (fired 1.5 s after startup when visualization is on) does
not open Edge on the desktop. The dashboard client in G1-C is the scripted g1c_vizclient.py.
No effect on anything else."""
import webbrowser

webbrowser.open = lambda url, new=0, autoraise=True: True
