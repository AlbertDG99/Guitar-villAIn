@echo off
echo "Activating virtual environment..."
call .\venv\Scripts\activate.bat
    
echo "Launching Guitar Hero visualizer..."
python -m color_pattern_approach.color_pattern_visualizer
    
echo "Script finished. Press any key to close."
pause