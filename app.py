import json
import time
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
import ollama

from monitor import ResourceSampler, hardware_info, memoire_modele, snapshot

# Définir le titre de l'application
st.title("ENSET Chatbot")

MODE_CONV = "💬 Conversation"
MODE_COMP = "⚖️ Comparaison des modèles"
available_models = []  # rempli au démarrage par la détection des modèles réellement installés

COLS_PERF = ["Modèle", "Temps total (s)", "TTFT (s)", "Tokens générés", "Vitesse (tok/s)",
             "Chargement modèle (s)", "Fin de génération"]
COLS_RES = ["Modèle", "CPU moy. (%)", "CPU max (%)", "RAM max (%)", "RAM utilisée max (Go)",
            "Mémoire du modèle (Mo)", "dont GPU (Mo)", "GPU moy. (%)", "GPU mémoire max (Mo)"]


# ───────── AMÉLIORATION 3 : génération en streaming avec mesure des performances ─────────
def lire(obj, cle, defaut=None):
    """Lit un champ d'une réponse Ollama (dict ou objet selon la version de la bibliothèque)."""
    if obj is None:
        return defaut
    if isinstance(obj, dict):
        return obj.get(cle, defaut)
    return getattr(obj, cle, defaut)


# ───────── AMÉLIORATION 5 : robustesse ─────────
def verifier_ollama():
    """Interroge Ollama. Retourne (service_accessible, modèles_installés, message_erreur)."""
    try:
        reponse = ollama.list()
        noms = []
        for m in lire(reponse, "models", []) or []:
            nom = lire(m, "model") or lire(m, "name")
            if nom:
                noms.append(nom)
        return True, sorted(noms), ""
    except Exception as e:
        return False, [], str(e)


def message_erreur(e, modele):
    """Transforme une exception en message explicite pour l'utilisateur."""
    if isinstance(e, ollama.ResponseError):
        if getattr(e, "status_code", None) == 404:
            return (f"Le modèle « {modele} » n'est pas installé. "
                    f"Installez-le avec la commande : `ollama pull {modele}`")
        return f"Ollama a renvoyé une erreur : {e}"
    if isinstance(e, OSError) or "connect" in str(e).lower():
        return ("Le service Ollama est inaccessible (il a peut-être été arrêté pendant la génération). "
                "Relancez l'application Ollama ou la commande `ollama serve`, puis réessayez.")
    return f"Erreur inattendue pendant la génération : {e}"


def generer(modele, messages, options, message_placeholder, zone_res=None):
    """Appelle le modèle en streaming, affiche la réponse au fil de l'eau.

    Retourne (réponse, métriques). Pendant la génération, un échantillonneur relève
    CPU / RAM / GPU en tâche de fond (Amélioration 4) ; zone_res les affiche en direct.
    """
    full_response, ttft, final, nb_morceaux, derniere_maj = "", None, None, 0, 0.0

    debut = time.perf_counter()
    with ResourceSampler() as echantillonneur:
        for response in ollama.chat(
            model=modele,
            messages=[{'role': m['role'], 'content': m['content']} for m in messages],
            options=options,
            stream=True
        ):
            texte = lire(lire(response, 'message'), 'content', '') or ''
            if texte:
                nb_morceaux += 1
                if ttft is None:
                    ttft = time.perf_counter() - debut      # temps jusqu'au 1er token
                full_response += texte
                message_placeholder.markdown(full_response + "▌")
            if lire(response, 'done'):
                final = response                             # dernier message : contient les statistiques
            if zone_res is not None and time.perf_counter() - derniere_maj > 0.5:
                derniere_maj = time.perf_counter()
                zone_res.caption(legende_direct(echantillonneur.dernier()))
    duree = time.perf_counter() - debut
    message_placeholder.markdown(full_response)
    if zone_res is not None:
        zone_res.empty()

    tokens = lire(final, 'eval_count') or nb_morceaux
    duree_gen_ns = lire(final, 'eval_duration') or 0
    charge_ns = lire(final, 'load_duration') or 0
    # vitesse de génération = tokens / durée de génération mesurée par Ollama
    vitesse = tokens / (duree_gen_ns / 1e9) if duree_gen_ns else (tokens / duree if duree else 0.0)
    raison = lire(final, 'done_reason')
    taille_mo, vram_mo = memoire_modele(modele)  # mémoire réellement occupée par le modèle (API Ollama)

    metriques = {
        "Modèle": modele,
        "Temps total (s)": round(duree, 3),
        "TTFT (s)": None if ttft is None else round(ttft, 3),
        "Tokens générés": int(tokens),
        "Vitesse (tok/s)": round(vitesse, 2),
        "Chargement modèle (s)": round(charge_ns / 1e9, 3),
        "Fin de génération": "tronquée (limite de tokens)" if raison == "length" else "terminée",
        **echantillonneur.resume(),                          # CPU, RAM, GPU
        "Mémoire du modèle (Mo)": taille_mo,
        "dont GPU (Mo)": vram_mo,
    }
    return full_response, metriques


def fmt(x, spec="{:.1f}"):
    return "–" if x is None else spec.format(x)


def legende_metriques(m):
    """Ligne de mesures affichée sous une réponse."""
    fin = " · ⚠️ tronquée" if m["Fin de génération"].startswith("tronquée") else ""
    res = (f" · CPU {fmt(m.get('CPU moy. (%)'), '{:.0f}')} % · RAM {fmt(m.get('RAM max (%)'), '{:.0f}')} %"
           f" · modèle {fmt(m.get('Mémoire du modèle (Mo)'), '{:.0f}')} Mo")
    if m.get("GPU moy. (%)") is not None:
        res += f" · GPU {fmt(m['GPU moy. (%)'], '{:.0f}')} %"
    return (f"⏱️ {fmt(m['Temps total (s)'])} s · TTFT {fmt(m['TTFT (s)'], '{:.2f}')} s · "
            f"{m['Tokens générés']} tokens · {fmt(m['Vitesse (tok/s)'])} tok/s{fin}{res}")


def legende_direct(s):
    """Ligne affichée en direct pendant la génération."""
    if s is None:
        return "🖥️ mesure en cours…"
    gpu = f" · GPU {s['gpu_util']:.0f} %" if s["gpu_util"] is not None else ""
    return f"🖥️ CPU {s['cpu']:.0f} % · RAM {s['ram_pct']:.0f} % ({s['ram_gb']:.1f} Go){gpu}"


def enregistrer_mesure(mode, question, metriques):
    """Garde toutes les mesures de la session (historique et export CSV)."""
    st.session_state.mesures.append({"Mode": mode, "Question": question[:40], **metriques})


def afficher_metriques(liste_metriques):
    """Tableaux comparatifs + graphiques (performances et ressources)."""
    df = pd.DataFrame(liste_metriques)
    st.markdown("#### 📊 Tableau comparatif des performances")
    st.dataframe(df[COLS_PERF], hide_index=True)
    colonnes = st.columns(3)
    for col, nom in zip(colonnes, ["Temps total (s)", "TTFT (s)", "Vitesse (tok/s)"]):
        col.caption(nom)
        col.bar_chart(df.set_index("Modèle")[nom])

    st.markdown("#### 🖥️ Ressources matérielles pendant la génération")
    st.dataframe(df[COLS_RES], hide_index=True)
    graphes = ["CPU moy. (%)", "Mémoire du modèle (Mo)"]
    if df["GPU moy. (%)"].notna().any():
        graphes.append("GPU moy. (%)")
    for col, nom in zip(st.columns(len(graphes)), graphes):
        col.caption(nom)
        col.bar_chart(df.set_index("Modèle")[nom])


# ───────── AMÉLIORATION 4 : panneau de surveillance (barre latérale) ─────────
def panneau_ressources():
    """Instantané CPU / RAM / GPU et modèles chargés en mémoire."""
    s = snapshot()
    st.sidebar.progress(min(s["cpu"] / 100, 1.0), text=f"CPU : {s['cpu']:.0f} %")
    st.sidebar.progress(min(s["ram_pct"] / 100, 1.0),
                        text=f"RAM : {s['ram_used_gb']:.1f} / {s['ram_total_gb']:.1f} Go")
    g = s["gpu"]
    if g:
        st.sidebar.progress(min(g["util"] / 100, 1.0), text=f"GPU : {g['util']:.0f} %")
        st.sidebar.progress(min(g["mem_used"] / g["mem_total"], 1.0),
                            text=f"Mémoire GPU : {g['mem_used']:.0f} / {g['mem_total']:.0f} Mo")
    else:
        st.sidebar.caption("GPU NVIDIA non détecté : mesures GPU indisponibles.")
    if s["modeles"]:
        for nom, taille, vram in s["modeles"]:
            st.sidebar.caption(f"Modèle chargé : {nom} — {taille} Mo (dont GPU : {vram} Mo)")
    else:
        st.sidebar.caption("Aucun modèle chargé en mémoire.")


@st.cache_data
def infos_materiel():
    return hardware_info()


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
        lignes += [f"[{auteur}]", m["content"]]
        if m.get("metriques"):
            lignes.append(f"({legende_metriques(m['metriques'])})")
        lignes.append("")
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
            if message.get("metriques"):
                st.caption(f"{message['modele']} — {legende_metriques(message['metriques'])}")

    # Accepter l'entrée de l'utilisateur
    prompt = st.chat_input("Que voulez-vous me demander ?")
    if prompt:
        premier_message = len(conv["messages"]) == 0

        with st.chat_message("user"):
            st.markdown(prompt)
        conv["messages"].append({"role": "user", "content": prompt})

        succes = False
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            zone_res = st.empty()
            try:
                # Indicateur visuel : génération en cours
                with st.spinner(f"⏳ {selected_model} génère une réponse…"):
                    full_response, metriques = generer(selected_model, conv["messages"], options,
                                                       message_placeholder, zone_res)
                st.caption(legende_metriques(metriques))
                succes = True
            except Exception as e:
                message_placeholder.empty()
                zone_res.empty()
                st.error(f"❌ {message_erreur(e, selected_model)}")

        if succes:
            conv["messages"].append({"role": "assistant", "content": full_response,
                                     "modele": selected_model, "metriques": metriques})
            enregistrer_mesure("Conversation", prompt, metriques)
            if premier_message:
                conv["titre"] = prompt[:40] + ("…" if len(prompt) > 40 else "")
                st.rerun()  # rafraîchit le titre dans la barre latérale
        else:
            conv["messages"].pop()  # on retire la question restée sans réponse


# ───────── AMÉLIORATION 2 : comparaison de deux modèles ─────────
def afficher_comparaison(d):
    """Affiche les deux réponses côte à côte, puis le tableau et les graphiques."""
    colonnes = st.columns(2)
    for col, m in zip(colonnes, d["modeles"]):
        with col:
            st.markdown(f"### {m}")
            if m in d["erreurs"]:
                st.error(f"❌ {d['erreurs'][m]}")
            else:
                st.markdown(d["reponses"][m])
                st.caption(legende_metriques(d["metriques"][m]))
    ok = [d["metriques"][m] for m in d["modeles"] if m in d["metriques"]]
    if ok:
        afficher_metriques(ok)


def page_comparaison(options):
    """Mode 2 : la même question envoyée à deux modèles (Amélioration 2)."""
    st.subheader("⚖️ Comparaison des modèles")
    st.caption("La même question est envoyée aux deux modèles, avec des paramètres de génération identiques : "
               + ", ".join(f"{k} = {v}" for k, v in options.items()) + ".")

    # 1. Question unique
    question = st.text_area("Question", height=100, placeholder="Saisissez une question unique…")

    # 2. Sélection de deux modèles parmi ceux installés
    c1, c2 = st.columns(2)
    modele_a = c1.selectbox("Modèle A", available_models, index=0, key="cmp_a")
    modele_b = c2.selectbox("Modèle B", available_models, index=min(1, len(available_models) - 1), key="cmp_b")
    if len(available_models) < 2:
        st.warning("Un seul modèle est installé : installez-en un second "
                   "(par exemple `ollama pull qwen:0.5b`) pour pouvoir comparer.")
    elif modele_a == modele_b:
        st.warning("Choisissez deux modèles différents.")

    lancer = st.button("▶️ Comparer", type="primary",
                       disabled=(not question.strip() or modele_a == modele_b))

    if lancer:
        # 3. Même question, mêmes paramètres, pour les deux modèles (l'un après l'autre)
        messages = [{"role": "user", "content": question}]
        colonnes = st.columns(2)  # 4. Réponses côte à côte
        reponses, metriques, erreurs = {}, {}, {}
        for col, modele in zip(colonnes, (modele_a, modele_b)):
            with col:
                st.markdown(f"### {modele}")
                placeholder = st.empty()
                zone_res = st.empty()
                try:
                    with st.spinner(f"⏳ {modele} génère une réponse…"):
                        reponses[modele], metriques[modele] = generer(modele, messages, options,
                                                                      placeholder, zone_res)
                    st.caption(legende_metriques(metriques[modele]))
                    enregistrer_mesure("Comparaison", question, metriques[modele])
                except Exception as e:
                    placeholder.empty()
                    zone_res.empty()
                    erreurs[modele] = message_erreur(e, modele)
                    st.error(f"❌ {erreurs[modele]}")
        st.session_state.derniere_comparaison = {
            "question": question, "modeles": (modele_a, modele_b),
            "reponses": reponses, "metriques": metriques, "erreurs": erreurs,
        }
        ok = [metriques[m] for m in (modele_a, modele_b) if m in metriques]
        if ok:
            afficher_metriques(ok)
    elif st.session_state.get("derniere_comparaison"):
        d = st.session_state.derniere_comparaison
        st.info(f"Dernière comparaison : « {d['question'][:80]} »")
        afficher_comparaison(d)

    # Historique de toutes les mesures de la session (export CSV)
    if st.session_state.mesures:
        with st.expander(f"📜 Historique de toutes les mesures ({len(st.session_state.mesures)})"):
            df = pd.DataFrame(st.session_state.mesures)
            st.dataframe(df, hide_index=True)
            st.download_button("⬇️ Exporter en CSV", df.to_csv(index=False),
                               file_name="mesures_session.csv", mime="text/csv")


# ───────── Programme principal ─────────
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
    nouvelle_conversation()
if "mesures" not in st.session_state:
    st.session_state.mesures = []

# Détection du service Ollama et des modèles réellement installés
service_ok, available_models, erreur_service = verifier_ollama()
if not service_ok:
    st.error("⚠️ Le service Ollama est inaccessible. Lancez l'application Ollama "
             "(ou la commande `ollama serve`), puis cliquez sur « Réessayer ».")
    st.code(erreur_service)
    st.button("🔄 Réessayer")
    st.stop()
if not available_models:
    st.warning("Ollama fonctionne, mais aucun modèle n'est installé. Exécutez par exemple "
               "`ollama pull llama3.2` et `ollama pull qwen:0.5b`, puis rechargez la page.")
    st.button("🔄 Recharger")
    st.stop()

st.sidebar.success(f"Ollama connecté · {len(available_models)} modèle(s) installé(s)")

# Choix du mode : conversation ou comparaison
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

# --- Surveillance des ressources (Amélioration 4) ---
st.sidebar.divider()
if st.sidebar.toggle("📈 Surveiller les ressources"):
    panneau_ressources()
    st.sidebar.button("🔄 Actualiser")
with st.sidebar.expander("🖥️ Matériel (pour le rapport)"):
    for cle, valeur in infos_materiel().items():
        st.write(f"**{cle}** : {valeur}")

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