"""
Overlay Utilities - Utilidades para Superposiciones en Pantalla
==============================================================

Funciones para mostrar información en pantalla sin interrumpir la ventana principal.
"""

import tkinter as tk

def create_overlay_display(message: str, duration: float = 2.0):
    """
    Crea una ventana de superposición temporal para mostrar un mensaje.

    Utiliza tkinter para crear una ventana sin bordes que se destruye
    automáticamente después de una duración específica.

    Args:
        message (str): El mensaje a mostrar.
        duration (float): Duración en segundos que el mensaje será visible.
    """
    try:
        root = tk.Tk()
        root.overrideredirect(True)
        
        # Estilo de la ventana
        root.configure(bg='black')
        label = tk.Label(
            root, text=message, fg="white", bg="black",
            font=("Arial", 14, "bold"),
            padx=20, pady=10
        )
        label.pack()

        # Posicionar en la esquina superior izquierda
        root.geometry("+50+50")
        
        # Hacerla desaparecer después de la duración
        root.after(int(duration * 1000), root.destroy)
        
        root.mainloop()

    except (ImportError, tk.TclError) as e:
        # Fallback si tkinter no está disponible o falla
        print(f"INFO (Overlay): {message} (Error: {e})") 