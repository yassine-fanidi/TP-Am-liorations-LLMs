"""Surveillance des ressources matérielles : CPU, RAM, GPU et mémoire occupée par les modèles Ollama."""
import platform
import shutil
import subprocess
import threading

import ollama
import psutil


# ───────── GPU (NVIDIA uniquement, via nvidia-smi) ─────────
def gpu_snapshot():
    """Retourne l'état du premier GPU NVIDIA, ou None si indisponible."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        util, used, total, name = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",", 3)]
        return {"util": float(util), "mem_used": float(used), "mem_total": float(total), "name": name}
    except Exception:
        return None


# ───────── Mémoire réellement occupée par les modèles (API Ollama) ─────────
def _lire(obj, cle, defaut=None):
    if obj is None:
        return defaut
    if isinstance(obj, dict):
        return obj.get(cle, defaut)
    return getattr(obj, cle, defaut)


def _norm(nom):
    return nom if ":" in nom else nom + ":latest"


def modeles_charges():
    """Modèles actuellement chargés en mémoire : liste de (nom, taille_Mo, dont_GPU_Mo)."""
    try:
        resultat = []
        for m in _lire(ollama.ps(), "models", []) or []:
            nom = _lire(m, "model") or _lire(m, "name") or "?"
            taille = (_lire(m, "size", 0) or 0) / 1024**2
            vram = (_lire(m, "size_vram", 0) or 0) / 1024**2
            resultat.append((nom, round(taille), round(vram)))
        return resultat
    except Exception:
        return []


def memoire_modele(modele):
    """(taille_Mo, dont_GPU_Mo) du modèle indiqué s'il est chargé, sinon (None, None)."""
    for nom, taille, vram in modeles_charges():
        if _norm(nom) == _norm(modele):
            return taille, vram
    return None, None


# ───────── Instantané et matériel ─────────
def snapshot(cpu_interval=0.3):
    """Instantané complet (panneau latéral)."""
    vm = psutil.virtual_memory()
    return {
        "cpu": psutil.cpu_percent(interval=cpu_interval),
        "ram_pct": vm.percent,
        "ram_used_gb": vm.used / 1024**3,
        "ram_total_gb": vm.total / 1024**3,
        "gpu": gpu_snapshot(),
        "modeles": modeles_charges(),
    }


def cpu_name():
    """Nom commercial du processeur (Windows, Linux, macOS)."""
    try:
        if platform.system() == "Windows":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for ligne in f:
                    if ligne.startswith("model name"):
                        return ligne.split(":", 1)[1].strip()
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def gpu_names():
    """Noms de toutes les cartes graphiques (Windows : y compris le GPU intégré)."""
    noms = []
    g = gpu_snapshot()
    if g:
        noms.append(f"{g['name']} ({g['mem_total']:.0f} Mo de mémoire)")
    elif platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=8)
            noms += [l.strip() for l in out.stdout.splitlines() if l.strip()]
        except Exception:
            pass
    return noms


def hardware_info():
    """Caractéristiques matérielles à reporter dans le rapport."""
    noms = gpu_names()
    if gpu_snapshot():
        gpu = noms[0]
    elif noms:
        gpu = " ; ".join(noms) + " (mesures GPU indisponibles : seul NVIDIA est supporté)"
    else:
        gpu = "Aucun GPU détecté"
    return {
        "Système": f"{platform.system()} {platform.release()}",
        "Processeur": cpu_name(),
        "Cœurs (physiques / logiques)": f"{psutil.cpu_count(logical=False)} / {psutil.cpu_count()}",
        "RAM totale": f"{psutil.virtual_memory().total / 1024**3:.1f} Go",
        "GPU": gpu,
    }


# ───────── Échantillonnage pendant une génération ─────────
class ResourceSampler:
    """Relève CPU / RAM / GPU en tâche de fond pendant une génération."""

    def __init__(self, interval=0.3):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            vm = psutil.virtual_memory()
            g = gpu_snapshot()
            self.samples.append({
                "cpu": psutil.cpu_percent(interval=None),
                "ram_pct": vm.percent,
                "ram_gb": vm.used / 1024**3,
                "gpu_util": g["util"] if g else None,
                "gpu_mem": g["mem_used"] if g else None,
            })
            self._stop.wait(self.interval)

    def __enter__(self):
        psutil.cpu_percent(interval=None)  # amorçage
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)
        return False

    def dernier(self):
        """Dernier échantillon (affichage en direct)."""
        return self.samples[-1] if self.samples else None

    def resume(self):
        s = self.samples
        if not s:
            return {"CPU moy. (%)": None, "CPU max (%)": None, "RAM max (%)": None,
                    "RAM utilisée max (Go)": None, "GPU moy. (%)": None, "GPU mémoire max (Mo)": None}
        gu = [x["gpu_util"] for x in s if x["gpu_util"] is not None]
        gm = [x["gpu_mem"] for x in s if x["gpu_mem"] is not None]
        return {
            "CPU moy. (%)": round(sum(x["cpu"] for x in s) / len(s), 1),
            "CPU max (%)": round(max(x["cpu"] for x in s), 1),
            "RAM max (%)": round(max(x["ram_pct"] for x in s), 1),
            "RAM utilisée max (Go)": round(max(x["ram_gb"] for x in s), 2),
            "GPU moy. (%)": round(sum(gu) / len(gu), 1) if gu else None,
            "GPU mémoire max (Mo)": round(max(gm), 0) if gm else None,
        }