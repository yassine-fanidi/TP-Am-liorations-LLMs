# TP: Amélioration et évaluation d'un chatbot basé sur des LLMS locaux

## Installation et exécution

### 1.	Installer Ollama (https://ollama.com/download), puis télécharger les modèles :
ollama pull llama3.2
ollama pull qwen:0.5b
ollama list

### 2.	Créer et activer l'environnement virtuel, puis installer les dépendances :
python -m venv .venv
.venv\Scripts\activate (cmd)
pip install -r requirements.txt

### 3.	Lancer l'application depuis le dossier contenant app.py et monitor.py :
streamlit run app.py

