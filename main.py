import sys
import os
from streamlit.web import cli as stcli

if __name__ == "__main__":
    # Add the current directory to Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Run Streamlit
    sys.argv = ["streamlit", "run", "src/app.py"]
    sys.exit(stcli.main())