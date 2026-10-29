import os

os.environ.setdefault("MPLBACKEND", "Agg")  # figuras sin ventana (el backend Tk falla en tests de Windows)
