@echo off
echo "Activando entorno virtual..."
call .\venv\Scripts\activate.bat
    
echo "Lanzando el visualizador de Guitar Hero..."
python -m sloth_approach.polygon_visualizer
    
echo "Script finalizado. Presiona una tecla para cerrar."
pause