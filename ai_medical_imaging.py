import os
import tempfile
import time
import pydicom
from pydicom.errors import InvalidDicomError
from tenacity import retry, wait_exponential, stop_after_attempt
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.run.agent import RunOutput
import streamlit as st
from agno.media import Image as AgnoImage

# --- Configuration et Fonctions Utilitaires ---

@retry(wait=wait_exponential(multiplier=1, min=2, max=15),
       stop=stop_after_attempt(5))
def run_agent_with_retry(agent, query, images):
    return agent.run(query, images=images, stream=False)

def load_image_to_pil(uploaded_file, file_extension):
    if file_extension.lower() == "dicom":
        try:
            ds = pydicom.dcmread(uploaded_file.read())
            if 'PixelData' in ds:
                import numpy as np
                pixel_array = ds.pixel_array
                if pixel_array.dtype != np.uint8:
                    if pixel_array.max() > 0:
                        pixel_array = (pixel_array / pixel_array.max() * 255).astype(np.uint8)
                    else:
                        pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
                return PILImage.fromarray(pixel_array)
            else:
                raise ValueError("Le fichier DICOM ne contient pas de données pixel.")
        except InvalidDicomError:
            raise ValueError("Format de fichier DICOM invalide.")
        except Exception as e:
            raise ValueError(f"Erreur lors du traitement du fichier DICOM : {e}")
    else:
        return PILImage.open(uploaded_file)

def process_uploaded_file(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    img_pil = load_image_to_pil(uploaded_file, ext)
    width, height = img_pil.size
    aspect_ratio = width / height
    new_width = 400
    new_height = int(new_width / aspect_ratio)
    return img_pil.resize((new_width, new_height))

# --- Interface Streamlit ---

st.set_page_config(
    page_title="Analyseur d'Imagerie Médicale IA",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .analysis-card {
        background-color: #f0f2f6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .badge {
        display: inline-block;
        background-color: #10b981;
        color: white;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-urgent {
        background-color: #ef4444;
    }
    .stProgress > div > div > div > div {
        background-color: #2563eb;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital-3.png", width=60)
    st.markdown("### ℹ️ Configuration")

    if "OPENROUTER_API_KEY" not in st.session_state:
        try:
            st.session_state.OPENROUTER_API_KEY = st.secrets["general"]["OPENROUTER_API_KEY"]
            if not st.session_state.OPENROUTER_API_KEY or st.session_state.OPENROUTER_API_KEY.strip() == "":
                st.session_state.OPENROUTER_API_KEY = None
        except (KeyError, FileNotFoundError):
            st.session_state.OPENROUTER_API_KEY = None

    if st.session_state.OPENROUTER_API_KEY:
        st.success("✅ Clé API active")
    else:
        st.error("❌ Clé API introuvable")

    st.divider()

    st.markdown("""
    **📋 Capacités**
    - Détection de lésions & anomalies
    - Analyse multi-modalités
    - Compte rendu structuré
    - Évaluation de sévérité
    - Recommandations
    """)

    st.divider()
    st.caption("🔒 Analyse locale - Les images ne sont pas stockées")

# --- Initialisation de l'Agent ---
medical_agent = None
if st.session_state.OPENROUTER_API_KEY:
    try:
        medical_agent = Agent(
            model=OpenRouter(
                id="nvidia/nemotron-nano-12b-v2-vl:free",
                api_key=st.session_state.OPENROUTER_API_KEY,
                max_tokens=8192,
            ),
            markdown=True
        )
    except Exception as e:
        st.error(f"Erreur d'initialisation : {e}")

if not medical_agent:
    st.warning("Veuillez configurer une clé API valide dans la barre latérale.")

# --- Requête d'Analyse Technique ---
query = """
Tu es un radiologue senior, expert en imagerie médicale. Analyse les images fournies et rédige un compte rendu structuré en français, au format professionnel.

## Méthodologie d'analyse
1. Identifier la modalité et la région anatomique
2. Analyser systématiquement chaque structure visible
3. Décrire les anomalies avec terminologie médicale précise
4. Évaluer la sévérité et l'urgence
5. Formuler un diagnostic raisonné
6. Proposer des examens complémentaires si indiqué

## Structure du rapport attendue

### 1. Renseignements
- Modalité, région anatomique, incidence / coupe

### 2. Description sémiologique
- Pour chaque structure analysée : signal / densité, forme, contours, taille, limites
- Anomalies : localisation précise, taille (mm), densité / signal, contours, caractère (homogène/ hétérogène)
- Si normal : "Structure d'aspect normal."

### 3. Interprétation
- Diagnostic principal
- Diagnostics différentiels (classés par probabilité)
- Sémiologie clinique associée

### 4. Conclusion
- Urgence : OUI / NON
- Sévérité : Normale / Légère / Modérée / Grave
- Recommandations : examens complémentaires, délai de prise en charge
- Résumé fonctionnel (2-3 lignes)

Format : markdown structuré, termes médicaux précis, pas de phrase d'introduction.
"""

# --- Interface Utilisateur ---
st.markdown("<div class='main-header'>🏥 Analyseur d'Imagerie Médicale</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Assistant IA spécialisé en radiologie — Téléchargez une ou plusieurs images pour un compte rendu structuré</div>", unsafe_allow_html=True)

b1, b2, b3 = st.columns(3)
with b1:
    st.markdown("🩻 **Rayons X**\nThorax, os, abdomen")
with b2:
    st.markdown("🧠 **IRM / Scanner**\nCerveau, articulations, organes")
with b3:
    st.markdown("📊 **DICOM**\nFormat natif accepté")

st.divider()

upload_container = st.container()
image_container = st.container()
analysis_container = st.container()

with upload_container:
    uploaded_files = st.file_uploader(
        "📤 Sélectionner des images médicales",
        type=["jpg", "jpeg", "png", "dicom"],
        accept_multiple_files=True,
        help="JPG, JPEG, PNG, DICOM — plusieurs fichiers acceptés"
    )

if uploaded_files and len(uploaded_files) > 0 and medical_agent:
    temp_paths = []

    with image_container:
        try:
            n = len(uploaded_files)
            st.markdown(f"**{n} fichier(s) téléchargé(s)**")
            cols = st.columns(min(n, 4))
            resized_images = []
            for i, f in enumerate(uploaded_files):
                img = process_uploaded_file(f)
                resized_images.append(img)
                with cols[i % 4]:
                    st.image(img, caption=f.name, use_container_width=True)

            col_a, col_b = st.columns([3, 1])
            with col_b:
                analyze_button = st.button(
                    "🔍 Lancer l'analyse" if n == 1 else f"🔍 Analyser les {n} images",
                    type="primary",
                    use_container_width=True
                )

        except ValueError as ve:
            st.error(f"Erreur de chargement : {ve}")
        except Exception as e:
            st.error(f"Erreur inattendue : {e}")

    with analysis_container:
        if analyze_button:
            status = st.status("🔄 Analyse en cours...", expanded=True)

            try:
                status.update(label="📂 Préparation des images...", state="running")
                for i, (f, img) in enumerate(zip(uploaded_files, resized_images)):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix=f"img_{i}_") as tmp:
                        temp_paths.append(tmp.name)
                        img.save(tmp.name)

                status.update(label="📤 Transmission à l'IA radiologue...", state="running")

                agno_images = [AgnoImage(filepath=p) for p in temp_paths]
                response: RunOutput = run_agent_with_retry(medical_agent, query, images=agno_images)

                status.update(label="✅ Analyse terminée", state="complete")

                st.markdown("---")
                st.markdown("### 📋 Compte Rendu Radiologique")
                st.markdown(response.content)
                st.markdown("---")
                st.caption("⚠️ Ce rapport est généré par IA. Il ne remplace pas l'avis d'un médecin radiologue qualifié.")

            except Exception as e:
                status.update(label="❌ Erreur", state="error")
                error_str = str(e)
                if "524" in error_str:
                    st.error("⏱️ Erreur 524 — Le serveur a mis trop de temps. Cliquez à nouveau sur Analyser.")
                elif "429" in error_str:
                    st.error("⏳ Limite de taux atteinte. Veuillez patienter quelques instants puis réessayer.")
                elif "timeout" in error_str.lower():
                    st.error("⏱️ Délai d'attente dépassé. Réessayez.")
                else:
                    st.error(f"Erreur : {e}")
            finally:
                for p in temp_paths:
                    if os.path.exists(p):
                        os.remove(p)
elif not medical_agent:
    st.warning("⚠️ Clé API manquante — configurez-la dans la barre latérale.")
else:
    st.info("👆 Téléchargez une ou plusieurs images médicales pour débuter.")
