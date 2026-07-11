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

# --- Interface Streamlit ---

st.set_page_config(layout="wide")

with st.sidebar:
    st.title("ℹ️ Configuration")

    if "OPENROUTER_API_KEY" not in st.session_state:
        try:
            st.session_state.OPENROUTER_API_KEY = st.secrets["general"]["OPENROUTER_API_KEY"]
            if not st.session_state.OPENROUTER_API_KEY or st.session_state.OPENROUTER_API_KEY.strip() == "":
                st.session_state.OPENROUTER_API_KEY = None
        except (KeyError, FileNotFoundError):
            st.session_state.OPENROUTER_API_KEY = None

    if st.session_state.OPENROUTER_API_KEY:
        st.success("✅ Clé API (OpenRouter) configurée.")
    else:
        st.error("❌ Clé API OpenRouter introuvable dans `.streamlit/secrets.toml`.")

    st.info(
        "Cet outil fournit une analyse assistée par IA des données d'imagerie médicale "
        "en utilisant la vision par ordinateur et l'expertise radiologique."
    )

# --- Initialisation de l'Agent ---
medical_agent = None
if st.session_state.OPENROUTER_API_KEY:
    try:
        medical_agent = Agent(
            model=OpenRouter(
                id="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                api_key=st.session_state.OPENROUTER_API_KEY,
                max_tokens=8192,
            ),
            markdown=True
        )
    except Exception as e:
        st.error(f"Échec de l'initialisation de l'agent IA : {e}")
        st.warning("Veuillez vérifier votre clé API et votre connexion internet.")

if not medical_agent:
    st.warning("Veuillez configurer une clé API valide dans la barre latérale pour activer l'agent.")

# --- Requête d'Analyse ---
query = """
Vous êtes un expert en imagerie médicale hautement qualifié, possédant une connaissance approfondie en radiologie et en imagerie diagnostique. Analysez l'image médicale du patient et structurez votre réponse comme suit :

### 1. Type d'Image & Région
- Spécifiez la modalité d'imagerie (Rayon X/IRM/Scanners/Échographie/etc.)
- Identifiez la région anatomique du patient et le positionnement
- Commentez la qualité de l'image et son adéquation technique

### 2. Résultats Clés
- Listez les observations principales de manière systématique
- Notez toute anomalie dans l'imagerie du patient avec des descriptions précises
- Incluez les mesures et densités lorsque cela est pertinent
- Décrivez l'emplacement, la taille, la forme et les caractéristiques
- Évaluez la sévérité : Normal/Léger/Modéré/Grave

### 3. Évaluation Diagnostique
- Fournissez le diagnostic principal avec un niveau de confiance
- Listez les diagnostics différentiels par ordre de probabilité
- Soutenez chaque diagnostic avec les preuves observées sur l'imagerie du patient
- Notez tout résultat critique ou urgent

### 4. Explication pour le Patient
- Expliquez les résultats dans un langage simple et clair que le patient peut comprendre
- Évitez le jargon médical ou fournissez des définitions claires
- Incluez des analogies visuelles si cela peut aider
- Répondez aux préoccupations courantes des patients liées à ces résultats

Formatez votre réponse en utilisant des titres markdown clairs et des points. Soyez concis mais complet.
Répondez TOUJOURS en français.
"""

# --- Interface Utilisateur ---
st.title("🏥 Agent de Diagnostic en Imagerie Médicale")
st.write("Téléchargez une image médicale (JPG, JPEG, PNG, DICOM) pour une analyse assistée par IA.")

upload_container = st.container()
image_container = st.container()
analysis_container = st.container()

with upload_container:
    uploaded_file = st.file_uploader(
        "Télécharger une image médicale",
        type=["jpg", "jpeg", "png", "dicom"],
        help="Formats supportés : JPG, JPEG, PNG, DICOM"
    )

if uploaded_file is not None and medical_agent:
    file_extension = os.path.splitext(uploaded_file.name)[1][1:].lower()
    temp_path = None

    with image_container:
        try:
            img_pil = load_image_to_pil(uploaded_file, file_extension)
            width, height = img_pil.size
            aspect_ratio = width / height
            new_width = 500
            new_height = int(new_width / aspect_ratio)
            resized_img_pil = img_pil.resize((new_width, new_height))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix="img_") as temp_file:
                temp_path = temp_file.name
                resized_img_pil.save(temp_path)

            st.image(
                resized_img_pil,
                caption="Image médicale téléchargée",
                use_container_width=True
            )

            analyze_button = st.button(
                "🔍 Analyser l'image",
                type="primary",
                use_container_width=True
            )

        except ValueError as ve:
            st.error(f"Erreur de chargement de l'image : {ve}")
            uploaded_file = None
        except Exception as e:
            st.error(f"Une erreur inattendue s'est produite lors du chargement : {e}")
            uploaded_file = None

    with analysis_container:
        if analyze_button and uploaded_file and temp_path:
            progress_bar = st.progress(0, text="🔄 Préparation...")

            try:
                progress_bar.progress(15, text="📂 Image chargée")
                agno_image = AgnoImage(filepath=temp_path)

                progress_bar.progress(30, text="📤 Envoi à l'API...")

                response: RunOutput = run_agent_with_retry(medical_agent, query, images=[agno_image])

                progress_bar.progress(90, text="📝 Génération du rapport...")
                time.sleep(0.2)

                progress_bar.progress(100, text="✅ Analyse terminée !")
                time.sleep(0.2)
                progress_bar.empty()

                st.markdown("### 📋 Résultats de l'Analyse")
                st.markdown("---")
                st.markdown(response.content)
                st.markdown("---")
                st.caption(
                    "Note : Cette analyse est générée par IA et doit être examinée par "
                    "un professionnel de santé qualifié."
                )

            except Exception as e:
                error_str = str(e)
                if "524" in error_str:
                    st.error("⏱️ Erreur 524 : le serveur upstream a mis trop de temps à répondre. Clique à nouveau sur 'Analyser' pour réessayer.")
                elif "timeout" in error_str.lower():
                    st.error("⏱️ L'analyse a pris trop de temps. Réessaie, le modèle peut être lent par moment.")
                else:
                    st.error(f"Erreur d'analyse : {e}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
elif not medical_agent:
    st.warning("Veuillez configurer une clé API valide dans la barre latérale pour activer l'agent.")
else:
    st.info("👆 Veuillez télécharger une image médicale pour commencer l'analyse.")
