# Show HN: Neuralrad Mammo AI - Free Research Tool for Mammogram Analysis

## Title
Show HN: Free AI-powered mammogram analysis tool combining deep learning + vision LLM (research only)

## Post Content

I've built Neuralrad Mammo AI, a free research tool that combines deep learning object detection with vision language models to analyze mammograms. The goal is to provide researchers and medical professionals with a secondary analysis tool for investigation purposes.

**⚠️ Important Disclaimers:**
- **NOT FDA 510(k) cleared** - this is purely for research investigation
- **Not for clinical diagnosis** - results should only be used as a secondary opinion
- **Completely free** - no registration, no payment, no data retention

**Technology Stack:**
- **Deep Learning**: RetinaNet with ResNet backbone trained on mammography datasets
- **Vision LLM**: Advanced vision language model for detailed interpretation
- **Frontend**: Svelte web application with interactive zoom/pan tools
- **Backend**: Python Flask API with PyTorch inference

**What it does:**
1. Upload a mammogram image (JPEG/PNG)
2. AI identifies potential masses and calcifications
3. Vision LLM provides radiologist-style analysis
4. Results presented in medical report format
5. Interactive viewer with zoom/pan capabilities

**Key Features:**
- Detects and classifies masses (benign/malignant)
- Identifies calcifications (benign/malignant) 
- Provides confidence scores and size assessments
- Generates detailed analysis using vision LLM
- No data storage - images processed and discarded
- Open source codebase available

**Use Cases:**
- Medical research and education
- Second opinion for researchers
- Algorithm comparison studies
- Teaching tool for radiology training
- Academic research validation

The system is designed specifically for research investigation purposes and to complement (never replace) professional medical judgment. I'm hoping this can be useful for the medical AI research community and welcome feedback on the approach.

**Live Demo**: http://mammo.neuralrad.com:5300
**GitHub**: [Link to repository if public]

What do you think? Any suggestions for improving the analysis or making it more useful for research purposes?

---

*Built with PyTorch, Svelte, and a lot of caffeine. Always remember: this is research only and not for clinical use.*