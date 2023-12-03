import os

def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            # Zamień spacje i myślniki na podkreślenia w nazwie pliku
            new_filename = filename.replace(' ', '_').replace('-', '_')

            # Tworzymy nową ścieżkę docelową
            new_file_path = os.path.join(folder_path, new_filename)

            # Zmieniamy nazwę pliku
            os.rename(file_path, new_file_path)
            print(f"Zmieniono nazwę pliku: {filename} -> {new_filename}")

# Podaj ścieżkę do folderu
folder_path = "/Users/dawidmarszalik/Bootcamp Data Science/Projekt końcowy/photos/"

# Wywołaj funkcję do zmiany nazw plików
rename_files(folder_path)
