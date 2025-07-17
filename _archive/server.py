import sys
import os
from livereload import Server

if len(sys.argv) < 2:
    print("Usage: python live_preview.py <notebook.html>")
    sys.exit(1)

html_file = sys.argv[1]

if not os.path.isfile(html_file):
    print(f"Error: File '{html_file}' does not exist.")
    sys.exit(1)

# Use the directory containing the HTML file as the root
root_dir = os.path.dirname(os.path.abspath(html_file))
file_name = os.path.basename(html_file)

print(f"Serving {file_name} from {root_dir} with live reload...")

server = Server()
server.watch(html_file)
server.serve(root=root_dir, port=8000)

