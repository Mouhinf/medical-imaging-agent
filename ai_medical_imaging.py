import os
import tempfile
import time
import numpy as np
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

def apply_voi_lut(pixel_array, ds):
    window_center = None
    window_width = None
    if hasattr(ds, 'VOILUTSequence') and ds.VOILUTSequence:
        return pixel_array
    if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
        try:
            wc = ds.WindowCenter
            ww = ds.WindowWidth
            if isinstance(wc, pydicom.multival.MultiValue):
                wc = float(wc[0])
            if isinstance(ww, pydicom.multival.MultiValue):
                ww = float(ww[0])
            window_center = float(wc)
            window_width = float(ww)
        except (TypeError, ValueError):
            pass
    if window_center is not None and window_width is not None and window_width > 0:
        low = window_center - window_width / 2
        high = window_center + window_width / 2
        pixel_array = np.clip(pixel_array, low, high)
        pixel_array = ((pixel_array - low) / (high - low) * 255).astype(np.uint8)
    else:
        pixel_array = pixel_array.astype(np.float64)
        mn, mx = pixel_array.min(), pixel_array.max()
        if mx > mn:
            pixel_array = ((pixel_array - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
    return pixel_array

def apply_rescale(pixel_array, ds):
    slope = 1.0
    intercept = 0.0
    if hasattr(ds, 'RescaleSlope'):
        try:
            slope = float(ds.RescaleSlope)
        except (TypeError, ValueError):
            pass
    if hasattr(ds, 'RescaleIntercept'):
        try:
            intercept = float(ds.RescaleIntercept)
        except (TypeError, ValueError):
            pass
    return pixel_array.astype(np.float64) * slope + intercept

def dicom_to_pil(uploaded_file):
    ds = pydicom.dcmread(uploaded_file.read())
    if 'PixelData' not in ds:
        raise ValueError("Le fichier DICOM ne contient pas de données pixel.")
    pixel_array = ds.pixel_array
    if pixel_array.ndim == 3 and pixel_array.shape[0] > 1:
        mid = pixel_array.shape[0] // 2
        pixel_array = pixel_array[mid]
    elif pixel_array.ndim == 3:
        pixel_array = pixel_array[0]
    photometric = str(getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2'))
    pixel_array = apply_rescale(pixel_array, ds)
    pixel_array = apply_voi_lut(pixel_array, ds)
    if photometric == 'MONOCHROME1':
        pixel_array = 255 - pixel_array
    if pixel_array.ndim == 2:
        return PILImage.fromarray(pixel_array.astype(np.uint8), mode='L')
    elif pixel_array.ndim == 3 and pixel_array.shape[2] >= 3:
        return PILImage.fromarray(pixel_array[:, :, :3].astype(np.uint8))
    else:
        return PILImage.fromarray(pixel_array.astype(np.uint8))

def extract_dicom_metadata(uploaded_file):
    ds = pydicom.dcmread(uploaded_file.read())
    tags = {
        "Patient": [],
        "Examen": [],
        "Image": [],
    }
    patient_fields = [
        ("PatientName", "Nom"),
        ("PatientID", "ID"),
        ("PatientBirthDate", "Date naissance"),
        ("PatientSex", "Sexe"),
        ("PatientAge", "Âge"),
    ]
    exam_fields = [
        ("StudyDate", "Date examen"),
        ("StudyTime", "Heure examen"),
        ("StudyDescription", "Description"),
        ("Modality", "Modalité"),
        ("BodyPartExamined", "Région"),
        ("SeriesDescription", "Description série"),
        ("ProtocolName", "Protocole"),
    ]
    image_fields = [
        ("Manufacturer", "Constructeur"),
        ("ManufacturerModelName", "Modèle"),
        ("InstitutionName", "Institution"),
        ("SliceThickness", "Épaisseur coupe"),
        ("PixelSpacing", "Espacement pixel"),
        ("KVP", "kVp"),
        ("XRayTubeCurrent", "mA"),
        ("Exposure", "Exposition (mAs)"),
        ("RepetitionTime", "TR (ms)"),
        ("EchoTime", "TE (ms)"),
        ("MagneticFieldStrength", "Champ (T)"),
        ("FlipAngle", "Angle de bascule"),
        ("SeriesNumber", "N° série"),
        ("InstanceNumber", "N° instance"),
    ]
    for attr, label in patient_fields:
        val = getattr(ds, attr, None)
        if val is not None:
            tags["Patient"].append((label, str(val)))
    for attr, label in exam_fields:
        val = getattr(ds, attr, None)
        if val is not None:
            tags["Examen"].append((label, str(val)))
    for attr, label in image_fields:
        val = getattr(ds, attr, None)
        if val is not None:
            tags["Image"].append((label, str(val)))
    return tags, ds

def load_image_to_pil(uploaded_file, file_extension):
    if file_extension.lower() in ("dicom", "dcm"):
        return dicom_to_pil(uploaded_file)
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
    st.markdown("📊 **DICOM** (.dcm)\nFenêtrage, métadonnées DICOM")

st.divider()

upload_container = st.container()
image_container = st.container()
analysis_container = st.container()

with upload_container:
    uploaded_files = st.file_uploader(
        "📤 Sélectionner des images médicales",
        type=["jpg", "jpeg", "png", "dicom", "dcm"],
        accept_multiple_files=True,
        help="JPG, JPEG, PNG, DICOM (.dcm) — plusieurs fichiers acceptés"
    )

if uploaded_files and len(uploaded_files) > 0 and medical_agent:
    temp_paths = []

    with image_container:
        try:
            n = len(uploaded_files)
            st.markdown(f"**{n} fichier(s) téléchargé(s)**")
            cols = st.columns(min(n, 4))
            resized_images = []
            dicom_metadatas = []
            for i, f in enumerate(uploaded_files):
                ext = os.path.splitext(f.name)[1][1:].lower()
                if ext in ("dicom", "dcm"):
                    meta, _ = extract_dicom_metadata(f)
                    f.seek(0)
                    dicom_metadatas.append((i, f.name, meta))
                img = process_uploaded_file(f)
                resized_images.append(img)
                with cols[i % 4]:
                    st.image(img, caption=f.name, use_container_width=True)

            for idx, fname, meta in dicom_metadatas:
                with st.expander(f"📋 Métadonnées DICOM — {fname}"):
                    for section, entries in meta.items():
                        if entries:
                            st.markdown(f"**{section}**")
                            for label, val in entries:
                                st.markdown(f"- {label}: `{val}`")
                            st.markdown("")

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

                dicom_context = ""
                for idx, fname, meta in dicom_metadatas:
                    modalite = ""
                    region = ""
                    for section, entries in meta.items():
                        for label, val in entries:
                            if "Modalité" in label:
                                modalite = val
                            if "Région" in label:
                                region = val
                    if modalite or region:
                        dicom_context += f"\n- {fname}: Modalité={modalite}, Région={region}"

                full_query = query
                if dicom_context:
                    full_query = (
                        "Informations DICOM extraites des fichiers :\n"
                        + dicom_context
                        + "\n\n---\n\n"
                        + query
                    )

                agno_images = [AgnoImage(filepath=p) for p in temp_paths]
                response: RunOutput = run_agent_with_retry(medical_agent, full_query, images=agno_images)

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
