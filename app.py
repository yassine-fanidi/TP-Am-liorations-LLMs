import json
import uuid
from datetime import datetime

import streamlit as st
import ollama

# Définir le titre de l'application
st.title("ENSET Chatbot")


# ───────── AMÉLIORATION 1 : gestion des conversations ─────────
def nouvelle_conversation():
    """Crée une conversation vide et la rend active."""
    cid = uuid.uuid4().hex[:8]
    st.session_state.conversations[cid] = {
        "id": cid,
        "titre": "Nouvelle conversation",
        "creee_le": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": [],
    }
    st.session_state.conv_courante = cid


def vers_txt(conv):
    lignes = [f"Conversation : {conv['titre']}", f"Créée le : {conv['creee_le']}", "=" * 50, ""]
    for m in conv["messages"]:
        auteur = "Utilisateur" if m["role"] == "user" else "Assistant"
        lignes += [f"[{auteur}]", m["content"], ""]
    return "\n".join(lignes)


def vers_json(conv):
    return json.dumps(conv, ensure_ascii=False, indent=2)


# Initialiser les conversations si elles n'existent pas
# (remplace l'ancien st.session_state.messages du code initial)
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
    nouvelle_conversation()

# --- Options de configuration (code initial, inchangé) ---
st.sidebar.header("Options de configuration")

# Sélection du modèle
available_models = ["llama3.2", "qwen:0.5b"]  # Liste des modèles disponibles
selected_model = st.sidebar.selectbox("Modèle", available_models, index=0)

# Température
temperature = st.sidebar.slider("Température", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

# Top-k
top_k = st.sidebar.slider("Top-k", min_value=1, max_value=100, value=40, step=1)

# Top-p
top_p = st.sidebar.slider("Top-p", min_value=0.0, max_value=1.0, value=0.9, step=0.05)

# Nombre maximal de tokens (num_predict)
max_tokens = st.sidebar.slider("Nombre maximal de tokens", min_value=1, max_value=2048, value=512, step=1)

# --- Fin des options de configuration ---

# --- Gestion des conversations (barre latérale) ---
st.sidebar.divider()
st.sidebar.header("Conversations")

if st.sidebar.button("➕ Nouvelle conversation"):
    nouvelle_conversation()
    st.rerun()

ids = list(st.session_state.conversations.keys())
etiquettes = [f"{n + 1}. {st.session_state.conversations[i]['titre']}" for n, i in enumerate(ids)]
etiquette = st.sidebar.selectbox(
    "Conversations de la session",
    etiquettes,
    index=ids.index(st.session_state.conv_courante),
)
choix = ids[etiquettes.index(etiquette)]
st.session_state.conv_courante = choix
conv = st.session_state.conversations[choix]

if st.sidebar.button("🗑️ Supprimer cette conversation"):
    del st.session_state.conversations[choix]
    if not st.session_state.conversations:
        nouvelle_conversation()
    else:
        st.session_state.conv_courante = list(st.session_state.conversations.keys())[-1]
    st.rerun()

st.sidebar.caption("Télécharger l'historique :")
col_txt, col_json = st.sidebar.columns(2)
col_txt.download_button("⬇️ TXT", vers_txt(conv), file_name=f"conversation_{choix}.txt",
                        disabled=not conv["messages"])
col_json.download_button("⬇️ JSON", vers_json(conv), file_name=f"conversation_{choix}.json",
                         mime="application/json", disabled=not conv["messages"])

# --- Afficher les messages de la conversation active ---
st.subheader(conv["titre"])
for message in conv["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accepter l'entrée de l'utilisateur
prompt = st.chat_input("Que voulez-vous me demander ?")
if prompt:
    premier_message = len(conv["messages"]) == 0

    # Afficher le message de l'utilisateur dans l'interface
    with st.chat_message("user"):
        st.markdown(prompt)
    # Ajouter le message de l'utilisateur à l'historique
    conv["messages"].append({"role": "user", "content": prompt})

    # Obtenir la réponse du modèle
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Appeler le modèle avec les options configurées
        for response in ollama.chat(
            model=selected_model,
            messages=[
                {
                    'role': m['role'],
                    'content': m['content'],
                }
                for m in conv["messages"]
            ],
            options={
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'num_predict': max_tokens,
            },
            stream=True
        ):
            full_response += response['message']['content']
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Ajouter la réponse du modèle à l'historique
    conv["messages"].append({"role": "assistant", "content": full_response})

    # Le titre de la conversation = début de la première question
    if premier_message:
        conv["titre"] = prompt[:40] + ("…" if len(prompt) > 40 else "")
        st.rerun()  # rafraîchit le titre dans la barre latérale