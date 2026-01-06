"""
Interface Streamlit améliorée pour la Classification de Documents
Avec gestion des documents vides, illisibles et hors classes
"""
import streamlit as st
import sys
from pathlib import Path
import json
import tempfile
from PIL import Image
import io
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Ajouter le chemin src
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent))

# IMPORTANT: Utiliser le pipeline amélioré
from pipeline_improved import ImprovedDocumentClassificationPipeline

# Configuration de la page
st.set_page_config(
    page_title="Classification de Documents",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé (même style qu'avant + nouveaux cas)
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .confidence-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        margin: 0.5rem 0;
    }
    
    .confidence-high {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .confidence-low {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    
    /* NOUVEAU: Badges pour cas spéciaux */
    .badge-error {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        color: white;
    }
    
    .main-title {
        text-align: center;
        color: white;
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
    }
    
    .subtitle {
        text-align: center;
        color: white;
        font-size: 1.3rem;
        margin-bottom: 3rem;
    }
    
    .class-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .stat-box {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Configuration des classes (incluant cas spéciaux)
CLASS_CONFIG = {
    "piece_identite": {
        "icon": "🪪",
        "name": "Pièce d'Identité",
        "color": "#667eea",
        "description": "CNIE Recto/Verso"
    },
    "releve_notes": {
        "icon": "📊",
        "name": "Relevé de Notes",
        "color": "#f093fb",
        "description": "Relevés académiques"
    },
    "facture": {
        "icon": "🧾",
        "name": "Facture",
        "color": "#fa709a",
        "description": "Factures d'eau, électricité, etc."
    },
    # NOUVEAUX: Cas spéciaux
    "DOCUMENT_VIDE": {
        "icon": "📄❌",
        "name": "Document Vide",
        "color": "#ff6b6b",
        "description": "PDF sans contenu détectable"
    },
    "DOCUMENT_ILLISIBLE": {
        "icon": "👓❌",
        "name": "Document Illisible",
        "color": "#ee5a6f",
        "description": "Qualité trop faible pour l'OCR"
    },
    "DOCUMENT_NON_RECONNU": {
        "icon": "❓",
        "name": "Document Non Reconnu",
        "color": "#feca57",
        "description": "Ne correspond à aucune classe connue"
    }
}


@st.cache_resource
def load_pipeline():
    """Charge le pipeline de classification (mise en cache)"""
    with st.spinner("⚙️ Initialisation du système..."):
        pipeline = ImprovedDocumentClassificationPipeline()
    return pipeline


def get_confidence_badge(result):
    """Retourne un badge HTML stylé selon la confiance et le type de résultat"""
    predicted_class = result['predicted_class']
    confidence = result['confidence']
    
    # Cas spéciaux
    if predicted_class == "DOCUMENT_VIDE":
        return '<span class="confidence-badge badge-error">⚠️ Document Vide</span>'
    elif predicted_class == "DOCUMENT_ILLISIBLE":
        return '<span class="confidence-badge badge-error">⚠️ Document Illisible</span>'
    elif predicted_class == "DOCUMENT_NON_RECONNU":
        return '<span class="confidence-badge badge-warning">⚠️ Hors Classes</span>'
    
    # Cas normaux
    if confidence >= 0.9:
        badge_class = "confidence-high"
        label = "Haute Confiance"
    elif confidence >= 0.7:
        badge_class = "confidence-medium"
        label = "Confiance Moyenne"
    else:
        badge_class = "confidence-low"
        label = "Faible Confiance"
    
    return f'<span class="confidence-badge {badge_class}">{label} ({confidence:.1%})</span>'


def create_confidence_gauge(confidence, predicted_class):
    """Crée une jauge de confiance avec Plotly"""
    
    # Pour les cas spéciaux, afficher 0
    if predicted_class in ["DOCUMENT_VIDE", "DOCUMENT_ILLISIBLE", "DOCUMENT_NON_RECONNU"]:
        value = 0
        color = "#ff6b6b"
    else:
        value = confidence * 100
        color = "#667eea"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confiance", 'font': {'size': 24}},
        number = {'suffix': "%", 'font': {'size': 40}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#fee140'},
                {'range': [50, 70], 'color': '#f5576c'},
                {'range': [70, 90], 'color': '#f093fb'},
                {'range': [90, 100], 'color': '#667eea'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#333", 'family': "Arial"}
    )
    
    return fig


def create_scores_chart(scores):
    """Crée un graphique des scores par modèle"""
    models = list(scores.keys())
    values = list(scores.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=values,
            marker_color=['#667eea', '#f093fb', '#fa709a'],
            text=[f'{v:.1%}' for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Scores par Modèle",
        xaxis_title="Modèle",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={'color': "#333", 'family': "Arial"}
    )
    
    return fig


def display_result(result):
    """Affiche les résultats de classification de manière stylée"""
    predicted_class = result['predicted_class']
    confidence = result['confidence']
    
    class_info = CLASS_CONFIG.get(predicted_class, {
        "icon": "📄",
        "name": predicted_class,
        "color": "#666",
        "description": ""
    })
    
    # En-tête avec icône et classe
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown(f"<div class='class-icon'>{class_info['icon']}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"### {class_info['name']}")
        st.markdown(f"*{class_info['description']}*")
        st.markdown(get_confidence_badge(result), unsafe_allow_html=True)
        
        # Afficher la raison de rejet si présente
        if 'rejection_reason' in result:
            st.warning(f"⚠️ **Raison**: {result['rejection_reason']}")
    
    # Jauge de confiance
    st.plotly_chart(create_confidence_gauge(confidence, predicted_class), use_container_width=True)
    
    # Détails supplémentaires
    with st.expander("📊 Détails de la Classification", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Informations Générales**")
            st.write(f"- **Page**: {result.get('page_number', 1)}")
            st.write(f"- **Stratégie**: {result.get('strategy', 'N/A')}")
            st.write(f"- **À revoir**: {'Oui ⚠️' if result.get('should_review', False) else 'Non ✅'}")
            st.write(f"- **Confiance OCR**: {result.get('ocr_confidence', 0):.1%}")
        
        with col2:
            st.markdown("**Caractéristiques du Document**")
            gabarit_features = result.get('gabarit_features', {})
            st.write(f"- **Aspect Ratio**: {gabarit_features.get('aspect_ratio', 0):.2f}")
            st.write(f"- **A une photo**: {'Oui' if gabarit_features.get('has_photo', False) else 'Non'}")
            st.write(f"- **A un tableau**: {'Oui' if gabarit_features.get('has_table', False) else 'Non'}")
            st.write(f"- **Densité de texte**: {gabarit_features.get('text_density', 0):.2%}")
    
    # Scores détaillés (seulement si document reconnu)
    if predicted_class not in ["DOCUMENT_VIDE", "DOCUMENT_ILLISIBLE", "DOCUMENT_NON_RECONNU"]:
        with st.expander("🎯 Scores par Modèle", expanded=False):
            all_scores = result.get('all_scores', {})
            if all_scores:
                cv_score = all_scores.get('cv_scores', {}).get(predicted_class, 0)
                nlp_score = all_scores.get('nlp_scores', {}).get(predicted_class, 0)
                gabarit_score = all_scores.get('gabarit_scores', {}).get(predicted_class, 0)
                
                scores_dict = {
                    'Computer Vision': cv_score,
                    'NLP': nlp_score,
                    'Gabarit': gabarit_score
                }
                
                st.plotly_chart(create_scores_chart(scores_dict), use_container_width=True)
    
    # Aperçu du texte
    with st.expander("📝 Aperçu du Texte Extrait", expanded=False):
        text_preview = result.get('text_preview', 'Aucun texte extrait')
        if text_preview == 'Aucun texte extrait' or len(text_preview.strip()) < 10:
            st.warning("⚠️ Aucun texte n'a pu être extrait de ce document")
        else:
            st.text_area("Texte OCR", text_preview, height=200, disabled=True)


def main():
    """Fonction principale de l'application"""
    
    # En-tête
    st.markdown('<h1 class="main-title">📄 Classification Intelligente de Documents</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Système multimodal avec détection des documents vides et hors classes</p>', unsafe_allow_html=True)
    
    # Barre latérale
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Mode de traitement
        mode = st.radio(
            "Mode de traitement",
            ["📄 Document unique", "📚 Traitement par batch"],
            help="Choisissez si vous voulez traiter un seul document ou plusieurs"
        )
        
        st.markdown("---")
        
        # Informations sur les classes
        st.markdown("## 📋 Classes Reconnues")
        for class_key in ["piece_identite", "releve_notes", "facture"]:
            class_info = CLASS_CONFIG[class_key]
            st.markdown(f"{class_info['icon']} **{class_info['name']}**")
            st.caption(class_info['description'])
        
        st.markdown("---")
        
        # Cas spéciaux
        st.markdown("## ⚠️ Cas Spéciaux Détectés")
        st.info(
            "Le système peut détecter:\n"
            "- 📄 Documents vides\n"
            "- 👓 Documents illisibles\n"
            "- ❓ Documents hors classes"
        )
        
        st.markdown("---")
        
        # À propos
        st.markdown("## ℹ️ À propos")
        st.info(
            "**Version**: 2.0\n\n"
            "**Modèles**:\n"
            "- ResNet50 (CV)\n"
            "- CamemBERT (NLP)\n"
            "- Détection de gabarits\n\n"
            "**Nouveautés v2.0**:\n"
            "- Détection documents vides\n"
            "- Gestion documents illisibles\n"
            "- Rejet hors classes"
        )
    
    # Zone principale
    try:
        # Charger le pipeline
        pipeline = load_pipeline()
        st.success("✅ Système prêt !")
        
        if mode == "📄 Document unique":
            # Mode document unique
            st.markdown("### 📤 Téléversez votre document PDF")
            
            uploaded_file = st.file_uploader(
                "Glissez-déposez ou cliquez pour sélectionner",
                type=['pdf'],
                help="Formats acceptés: PDF uniquement"
            )
            
            if uploaded_file is not None:
                # Sauvegarder temporairement le fichier
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Afficher les informations du fichier
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                    st.markdown('<div class="stat-number">📄</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="stat-label">{uploaded_file.name}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    file_size = len(uploaded_file.getvalue()) / 1024  # KB
                    st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                    st.markdown(f'<div class="stat-number">{file_size:.1f}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="stat-label">KB</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                    st.markdown(f'<div class="stat-number">{datetime.now().strftime("%H:%M")}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="stat-label">Téléversé à</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Bouton de classification
                if st.button("🚀 Classifier le Document", type="primary", use_container_width=True):
                    with st.spinner("🔄 Classification en cours..."):
                        try:
                            # Traiter le document
                            results = pipeline.process_single_document(tmp_path)
                            
                            st.markdown("---")
                            st.markdown("## 🎯 Résultats de la Classification")
                            
                            # Afficher chaque page
                            for idx, result in enumerate(results):
                                st.markdown(f"### 📄 Page {idx + 1}")
                                display_result(result)
                                
                                if idx < len(results) - 1:
                                    st.markdown("---")
                            
                            # Bouton de téléchargement des résultats
                            st.markdown("---")
                            results_json = json.dumps(results, indent=2, ensure_ascii=False)
                            st.download_button(
                                label="📥 Télécharger les Résultats (JSON)",
                                data=results_json,
                                file_name=f"classification_{uploaded_file.name.replace('.pdf', '')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                            
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la classification: {str(e)}")
                            st.exception(e)
        
        else:
            # Mode batch
            st.markdown("### 📚 Traitement par Batch")
            st.info("🚧 Fonctionnalité en cours de développement")
            st.markdown(
                "Cette fonctionnalité permettra de traiter plusieurs documents PDF en une seule fois "
                "et de générer un rapport complet avec statistiques."
            )
    
    except Exception as e:
        st.error(f"❌ Erreur d'initialisation: {str(e)}")
        st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: white; padding: 2rem;'>"
        "Développé avec ❤️ | Powered by Streamlit, PyTorch & CamemBERT | v2.0 - Gestion cas spéciaux"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
