import json
import uuid
from datetime import datetime

import streamlit as st
import ollama

# Définir le titre de l'application
st.title("ENSET Chatbot")

MODE_CONV = "💬 Conversation"
MODE_COMP = "⚖️ Comparaison des modèles"
available_models = ["llama3.2", "qwen:0.5b"]  # Liste des modèles disponibles


# ───────── Génération en streaming (boucle du code initial, mise en fonction) ─────────
def generer(modele, messages, options, message_placeholder):
    """Appelle le modèle en streaming, affiche la réponse au fil de l'eau et la retourne."""
    full_response = ""
    for response in ollama.chat(
        model=modele,
        messages=[{'role': m['role'], 'content': m['content']} for m in messages],
        options=options,
        stream=True
    ):
        full_response += response['message']['content']
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    return full_response


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


def page_conversation(options):
    """Mode 1 : conversation avec historique (Amélioration 1)."""
    st.sidebar.divider()
    st.sidebar.header("Conversations")

    # Sélection du modèle (mode Conversation uniquement)
    selected_model = st.sidebar.selectbox("Modèle", available_models, index=0)

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

    # Afficher les messages de la conversation active
    st.subheader(conv["titre"])
    for message in conv["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accepter l'entrée de l'utilisateur
    prompt = st.chat_input("Que voulez-vous me demander ?")
    if prompt:
        premier_message = len(conv["messages"]) == 0

        with st.chat_message("user"):
            st.markdown(prompt)
        conv["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = generer(selected_model, conv["messages"], options, message_placeholder)

        conv["messages"].append({"role": "assistant", "content": full_response})

        if premier_message:
            conv["titre"] = prompt[:40] + ("…" if len(prompt) > 40 else "")
            st.rerun()  # rafraîchit le titre dans la barre latérale


# ───────── AMÉLIORATION 2 : comparaison de deux modèles ─────────
def afficher_comparaison(d):
    """Affiche les deux réponses côte à côte."""
    colonnes = st.columns(2)
    for col, m in zip(colonnes, d["modeles"]):
        with col:
            st.markdown(f"### {m}")
            st.markdown(d["reponses"][m])


def page_comparaison(options):
    """Mode 2 : la même question envoyée à deux modèles (Amélioration 2)."""
    st.subheader("⚖️ Comparaison des modèles")
    st.caption("La même question est envoyée aux deux modèles, avec des paramètres de génération identiques : "
               + ", ".join(f"{k} = {v}" for k, v in options.items()) + ".")

    # 1. Question unique
    question = st.text_area("Question", height=100, placeholder="Saisissez une question unique…")

    # 2. Sélection de deux modèles
    c1, c2 = st.columns(2)
    modele_a = c1.selectbox("Modèle A", available_models, index=0, key="cmp_a")
    modele_b = c2.selectbox("Modèle B", available_models, index=1, key="cmp_b")
    if modele_a == modele_b:
        st.warning("Choisissez deux modèles différents.")

    lancer = st.button("▶️ Comparer", type="primary",
                       disabled=(not question.strip() or modele_a == modele_b))

    if lancer:
        # 3. Même question, mêmes paramètres, pour les deux modèles (l'un après l'autre)
        messages = [{"role": "user", "content": question}]
        colonnes = st.columns(2)  # 4. Réponses côte à côte
        reponses = {}
        for col, modele in zip(colonnes, (modele_a, modele_b)):
            with col:
                st.markdown(f"### {modele}")
                placeholder = st.empty()
                reponses[modele] = generer(modele, messages, options, placeholder)
        st.session_state.derniere_comparaison = {
            "question": question, "modeles": (modele_a, modele_b), "reponses": reponses,
        }
    elif st.session_state.get("derniere_comparaison"):
        d = st.session_state.derniere_comparaison
        st.info(f"Dernière comparaison : « {d['question'][:80]} »")
        afficher_comparaison(d)


# ───────── Programme principal ─────────
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
    nouvelle_conversation()

# Choix du mode
mode = st.sidebar.radio("Mode", [MODE_CONV, MODE_COMP])

# --- Options de configuration (code initial) ---
st.sidebar.header("Options de configuration")

# Température
temperature = st.sidebar.slider("Température", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

# Top-k
top_k = st.sidebar.slider("Top-k", min_value=1, max_value=100, value=40, step=1)

# Top-p
top_p = st.sidebar.slider("Top-p", min_value=0.0, max_value=1.0, value=0.9, step=0.05)

# Nombre maximal de tokens (num_predict)
max_tokens = st.sidebar.slider("Nombre maximal de tokens", min_value=1, max_value=2048, value=512, step=1)

# Seed : même graine pour les deux modèles, afin d'obtenir des comparaisons reproductibles
seed = st.sidebar.number_input("Seed (reproductibilité)", value=42, step=1)

# --- Fin des options de configuration ---

options = {
    'temperature': temperature,
    'top_k': top_k,
    'top_p': top_p,
    'num_predict': max_tokens,
    'seed': int(seed),
}

if mode == MODE_CONV:
    page_conversation(options)
else:
    page_comparaison(options)