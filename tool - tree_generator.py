# tool - tree_generator_gui.py

import sys
import os
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QFileDialog, QLabel)
from PySide6.QtCore import Qt, QTimer # <-- ADAUGĂ QTimer AICI

# Constante pentru a păstra log-ul curat
IGNORE_DIRS = {
    "__pycache__", ".idea", ".git", "venv", ".venv", "Aarici_env",
    "build", "dist" # Ignorăm și folderele generate de PyInstaller
}
IGNORE_FILES = {
    ".gitignore", "tree_generator.py", "project_structure_log.txt",
    "TeatruDigital.spec", os.path.basename(__file__) # Ignorăm scriptul curent
}

class TreeGeneratorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generator Structură Folder")
        self.setGeometry(300, 300, 700, 500)
        self.init_ui()

    def init_ui(self):
        """Creează elementele interfeței grafice."""
        layout = QVBoxLayout(self)

        # Butonul de selecție
        self.select_button = QPushButton("📂 Selectează un Folder pentru Analiză...")
        self.select_button.setStyleSheet("font-size: 14px; padding: 10px;")
        self.select_button.clicked.connect(self.run_generation)
        layout.addWidget(self.select_button)
        
        # Eticheta care afișează calea selectată
        self.path_label = QLabel("Niciun folder selectat.")
        self.path_label.setStyleSheet("color: #555; margin-bottom: 10px;")
        layout.addWidget(self.path_label)

        # Caseta de text pentru rezultat
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("Structura folderului selectat va apărea aici...")
        self.result_text.setStyleSheet("font-family: Consolas, Courier New, monospace; font-size: 12px;")
        layout.addWidget(self.result_text)
        
        # Butonul de copiere
        self.copy_button = QPushButton("📋 Copiază în Clipboard")
        self.copy_button.setStyleSheet("font-size: 14px; padding: 10px;")
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.copy_button.setEnabled(False) # Se activează doar după generare
        layout.addWidget(self.copy_button)

    def run_generation(self):
        """Deschide dialogul de selecție și pornește generarea."""
        folder_path = QFileDialog.getExistingDirectory(self, "Selectează un folder de analizat")
        
        if folder_path:
            self.path_label.setText(f"Analizând: {folder_path}")
            self.result_text.setPlainText("Se generează structura, te rog așteaptă...")
            QApplication.processEvents() # Forțează actualizarea UI

            try:
                tree_output = self.generate_tree_string(folder_path)
                self.result_text.setPlainText(tree_output)
                self.copy_button.setEnabled(True)
                self.copy_button.setText("📋 Copiază în Clipboard")
            except Exception as e:
                self.result_text.setPlainText(f"A apărut o eroare:\n\n{e}")
                self.copy_button.setEnabled(False)

    def generate_tree_string(self, startpath):
        """
        Generează o reprezentare arborescentă sub formă de string.
        """
        tree_lines = []
        tree_lines.append(f"Structura folderului: {os.path.abspath(startpath)}")
        tree_lines.append("=" * 60)
        
        for root, dirs, files in os.walk(startpath, topdown=True):
            # Excludem directoarele din lista IGNORE_DIRS
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            # Nu afișăm folderul rădăcină în listă, ci doar subfolderele
            relative_root = os.path.relpath(root, startpath)
            if relative_root == ".":
                level = 0
            else:
                level = relative_root.count(os.sep) + 1

            if level > 0:
                indent = ' ' * 4 * (level - 1)
                tree_lines.append(f"{indent}└── {os.path.basename(root)}/")
            
            indent = ' ' * 4 * level
            for file_name in sorted(files):
                if file_name not in IGNORE_FILES:
                    tree_lines.append(f"{indent}├── {file_name}")
                    
        return "\n".join(tree_lines)
        
    def copy_to_clipboard(self):
        """Copiază conținutul casetei de text în clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.result_text.toPlainText())
        
        # Feedback vizual pentru utilizator
        self.copy_button.setText("✅ Copiat!")
        QTimer.singleShot(1500, lambda: self.copy_button.setText("📋 Copiază în Clipboard"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TreeGeneratorApp()
    window.show()
    sys.exit(app.exec())