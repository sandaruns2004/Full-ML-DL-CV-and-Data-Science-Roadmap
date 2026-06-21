import os

# Base directory
base_dir = r"c:\Users\ADMIN\Desktop\Full-ML-DL-CV-and-Data-Science-Roadmap"

# Structure mappings for the roadmap table
roadmap_data = [
    ("00", "00-Prerequisites", "Python, Math, Stats", "⭐☆☆☆☆", "20"),
    ("01", "01-Data-Science-Foundations", "EDA, Stats, Preprocessing", "⭐☆☆☆☆", "30"),
    ("02-05", "Classical Machine Learning", "Supervised, Unsupervised, Ensembles", "⭐⭐☆☆☆", "60"),
    ("06", "06-Neural-Networks-Foundations", "Perceptrons, Backprop, PyTorch", "⭐⭐⭐☆☆", "40"),
    ("07 & 11", "Computer Vision & CNNs", "CNNs, Object Detection, Segmentation", "⭐⭐⭐☆☆", "50"),
    ("08 & 12", "NLP Basics", "RNNs, Embeddings, Text Classification", "⭐⭐⭐☆☆", "40"),
    ("09", "09-Transformers", "Attention, BERT, GPT, ViT", "⭐⭐⭐⭐☆", "50"),
    ("10", "10-Generative", "GANs, VAEs, Diffusion", "⭐⭐⭐⭐☆", "40"),
    ("11", "Modern LLMs & RAG", "Fine-Tuning, Vector DBs", "⭐⭐⭐⭐⭐", "60"),
    ("13 & 14", "Advanced Topics & Distributed", "RL, GNNs, Big Data", "⭐⭐⭐⭐⭐", "50"),
    ("15", "15-ML-In-Production", "MLOps, Pipelines, Deployment", "⭐⭐⭐⭐⭐", "60"),
    ("16", "16-Projects", "Projects & Showcases", "🎓", "On-going"),
]

# Get list of notebooks and projects dynamically
notebooks = []
projects = []

for root, dirs, files in os.walk(base_dir):
    if 'venv' in root or '.git' in root:
        continue
    rel_root = os.path.relpath(root, base_dir).replace('\\', '/')
    for f in files:
        if f.endswith('.ipynb'):
            notebooks.append({
                "name": f.replace('.ipynb', '').replace('-', ' '),
                "link": f"./{rel_root}/{f}",
                "topic": "Interactive Notebook",
                "status": "✅ Available"
            })
        elif f.endswith('.md') and 'Projects' in rel_root and f != 'README.md':
            # Extract some mock info based on filename
            projects.append({
                "name": f.replace('.md', '').replace('-', ' '),
                "link": f"./{rel_root}/{f}",
                "difficulty": "Varies",
                "skills": "Varies based on project"
            })

# Ensure at least placeholders if empty
if not notebooks:
    notebooks = [
        {"name": "01 Intro to PyTorch", "link": "./Jupyter-Notebooks/01.ipynb", "topic": "Deep Learning", "status": "Planned"},
        {"name": "02 Building RAG", "link": "./Jupyter-Notebooks/02.ipynb", "topic": "LLMs", "status": "Planned"}
    ]

if not projects:
    projects = [
        {"name": "Real Estate Predictor", "link": "./16-Projects/real_estate.md", "difficulty": "Beginner", "skills": "Pandas, Regression"},
        {"name": "Pneumonia Detection", "link": "./16-Projects/pneumonia.md", "difficulty": "Intermediate", "skills": "PyTorch, CNNs"}
    ]

# Construct the README content
readme_content = f"""# 🚀 [The Ultimate Machine Learning, DL & Data Science Roadmap](./README.md)
> **[A world-class, comprehensive, math-heavy, and code-rich learning path taking you from absolute beginner to production-ready AI Engineer.](./README.md)**

[Welcome to the definitive guide for mastering Machine Learning, Deep Learning, Computer Vision, NLP, LLMs, and MLOps.](./README.md) [This repository is meticulously designed to bridge the gap between academic theory and industry-grade production AI.](./README.md)

### 🎯 [Who is this for?](./README.md)
- **[Self-Taught Learners](./00-Prerequisites)** seeking a structured, step-by-step curriculum.
- **[Students & Undergraduates](./01-Data-Science-Foundations)** looking for deep mathematical intuition alongside practical code.
- **[Data Scientists & ML Engineers](./06-Neural-Networks-Foundations)** transitioning into Deep Learning, LLMs, and MLOps.
- **[AI Researchers](./13-Advanced)** who want a quick reference for foundational models and math.

### 🌟 [Learning Outcomes](./README.md)
By the end of this roadmap, you will be able to:
- [Build, train, and deploy robust ML models from scratch.](./02-Supervised-Learning)
- [Architect Deep Learning and Computer Vision systems using PyTorch.](./07-CNNs)
- [Develop advanced NLP applications using Transformers and LLMs.](./09-Transformers)
- [Construct production-ready RAG (Retrieval-Augmented Generation) pipelines.](./10-Generative)
- [Deploy scalable AI systems utilizing modern MLOps practices.](./15-ML-In-Production)

---

## 🗺️ [Visual Overview & Architecture](./README.md)

```mermaid
flowchart TD
    A["[00 - Prerequisites: Python & Math](./00-Prerequisites)"] --> B["[01 - Data Science & EDA](./01-Data-Science-Foundations)"]
    B --> C["[02-05 Classical Machine Learning](./02-Supervised-Learning)"]
    C --> D["[06 - Neural Networks Foundations](./06-Neural-Networks-Foundations)"]
    
    D --> E["[07/11 - Computer Vision & CNNs](./07-CNNs)"]
    D --> F["[12 - NLP](./12-NLP)"]
    
    F --> G["[08/09 - Sequence Models & Transformers](./09-Transformers)"]
    E --> G
    
    G --> H["[10 - Generative AI & Diffusion](./10-Generative)"]
    G --> I["[11/12 - LLMs & RAG Pipelines](./09-Transformers)"]
    
    H --> J["[13/14 - Advanced Topics & RL](./13-Advanced)"]
    I --> J
    
    J --> K["[15 - MLOps & Production AI Systems](./15-ML-In-Production)"]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style K fill:#bbf,stroke:#333,stroke-width:2px
```

---

## 📅 [The Learning Roadmap](./README.md)

[A structured progression estimating **~500 Hours** of total learning.](./README.md)

| [Stage](./README.md) | [Topic](./README.md) | [Difficulty](./README.md) | [Estimated Hours](./README.md) |
| ----- | ----- | ---------- | --------------- |
"""

for item in roadmap_data:
    stage, title, desc, diff, hours = item
    folder = title if '-' in title else title.replace(' ', '-')
    # Use real folder if exists, or fallback
    link_path = f"./{folder}" if os.path.exists(os.path.join(base_dir, folder)) else f"./{folder} (Coming Soon)"
    readme_content += f"| `[{stage}]({link_path})` | [{title}]({link_path}) | [{diff}]({link_path}) | [{hours}]({link_path}) |\n"

readme_content += """
---

## ✨ [Feature Highlights](./README.md)

- 🧱 **[Structured Roadmap](./README.md)**: No more jumping between random tutorials. A clear A-to-Z learning path.
- 🔗 **[Hyperlinked Learning System](./README.md)**: Every concept, file, and topic is cross-linked for seamless navigation.
- 💼 **[Industry-Focused Learning](./15-ML-In-Production)**: Focuses heavily on what actually matters in modern tech.
- 🛠️ **[Hands-On Projects](./16-Projects)**: Move from theory to practice with carefully scoped real-world projects.
- 🎤 **[Interview Preparation](./Cheat-Sheets)**: Curated guides for answering the toughest ML questions.
- 📝 **[Cheat Sheets & Notes](./Cheat-Sheets)**: Quick reference guides for algorithms.

---

## 🚀 [Skills You Will Gain](./README.md)

- ✅ **[Build ML models](./02-Supervised-Learning)**
- ✅ **[Train deep learning systems](./06-Neural-Networks-Foundations)**
- ✅ **[Build computer vision applications](./07-CNNs)**
- ✅ **[Build NLP applications](./12-NLP)**
- ✅ **[Use Transformers and LLMs](./09-Transformers)**
- ✅ **[Build RAG systems](./10-Generative)**
- ✅ **[Deploy AI systems](./15-ML-In-Production)**
- ✅ **[Create production ML pipelines](./15-ML-In-Production)**

---

## 🛠️ [Project Showcase](./16-Projects)

| [Project](./16-Projects) | [Difficulty](./16-Projects) | [Skills Learned](./16-Projects) |
| ------- | ---------- | -------------- |
"""

for proj in projects:
    readme_content += f"| [{proj['name']}]({proj['link']}) | [{proj['difficulty']}]({proj['link']}) | [{proj['skills']}]({proj['link']}) |\n"

readme_content += """
*(Recommended Future Projects: Multi-Modal Chatbots, Distributed Reinforcement Learning Agents).*

---

## 📓 [Interactive Notebook Showcase](./Jupyter-Notebooks)

| [Notebook](./Jupyter-Notebooks) | [Topic](./Jupyter-Notebooks) | [Status](./Jupyter-Notebooks) |
| -------- | ----- | ------ |
"""

for nb in notebooks:
    readme_content += f"| [{nb['name']}]({nb['link']}) | [{nb['topic']}]({nb['link']}) | [{nb['status']}]({nb['link']}) |\n"

readme_content += """
---

## ⚡ [Quick Start](./README.md)

### [1. Clone the repository](./README.md)
```bash
git clone https://github.com/sandaruns2004/Full-ML-DL-CV-and-Data-Science-Roadmap.git
cd Full-ML-DL-CV-and-Data-Science-Roadmap
```

### [2. Create a virtual environment](./README.md)
```bash
python -m venv venv
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate
```

### [3. Install requirements](./README.md)
```bash
pip install -r requirements.txt
```

### [4. Navigation Guide](./README.md)
- [Start → Python](./00-Prerequisites)
- [Next → Data Science Foundations](./01-Data-Science-Foundations)
- [Next → Machine Learning](./02-Supervised-Learning)

---

## 🤝 [Contributor Guide](./README.md)

- **[Contribution Guidelines](./README.md)**: Please review before submitting a PR.
- **[Folder Structure Rules](./README.md)**: Follow our chronological numbered module format.
- **[Naming Conventions](./README.md)**: Use Hyphenated-Pascal-Case for Markdown files.
- **[Pull Request Process](./README.md)**: Fork, branch, and submit a detailed description.
- **[Issue Reporting System](./README.md)**: Found a bug? Open an issue with reproduction steps.

---

## ❓ [FAQ](./README.md)

<details><summary><b>[1. Do I need math first?](./00-Prerequisites)</b></summary>Yes, Phase 00 covers the essential math required.</details>
<details><summary><b>[2. Can beginners use this?](./00-Prerequisites)</b></summary>Absolutely. It is designed from the ground up.</details>
<details><summary><b>[3. How long does it take?](./README.md)</b></summary>Expect ~6 to 9 months studying 15 hours a week.</details>
<details><summary><b>[4. Which project should I start with?](./16-Projects)</b></summary>Start with the beginner EDA projects.</details>
<details><summary><b>[5. Is this enough for a job?](./15-ML-In-Production)</b></summary>Yes, especially the MLOps sections.</details>
<details><summary><b>[6. Do I need prior coding experience?](./00-Prerequisites)</b></summary>No, basic Python is taught in Phase 00.</details>
<details><summary><b>[7. Can I become an AI engineer using this roadmap?](./README.md)</b></summary>Yes, this covers everything an AI engineer needs.</details>
<details><summary><b>[8. Should I learn PyTorch or TensorFlow?](./06-Neural-Networks-Foundations)</b></summary>We heavily focus on PyTorch due to industry demand.</details>
<details><summary><b>[9. Are the datasets included?](./01-Data-Science-Foundations)</b></summary>Open-source dataset links are provided in each module.</details>
<details><summary><b>[10. Can I skip the classical ML phase?](./02-Supervised-Learning)</b></summary>Not recommended. Deep Learning builds upon it.</details>
<details><summary><b>[11. Is there a certificate?](./README.md)</b></summary>No, this is an open-source guide, not a formal course.</details>
<details><summary><b>[12. How do I practice MLOps?](./15-ML-In-Production)</b></summary>We use Docker and local tools that simulate cloud environments.</details>
<details><summary><b>[13. Can I use Google Colab?](./Jupyter-Notebooks)</b></summary>Yes, all notebooks are Colab-compatible.</details>
<details><summary><b>[14. How are LLMs covered?](./09-Transformers)</b></summary>We dive deep into Transformers, Fine-tuning, and RLHF.</details>
<details><summary><b>[15. Are there any video tutorials?](./README.md)</b></summary>This is a text and code-first repository.</details>
<details><summary><b>[16. Can I contribute a translation?](./README.md)</b></summary>Yes, translations are welcome in PRs.</details>
<details><summary><b>[17. Does it cover Reinforcement Learning?](./13-Advanced)</b></summary>Yes, Phase 13 introduces RL algorithms.</details>
<details><summary><b>[18. How do I track progress?](./README.md)</b></summary>Fork the repo and check off modules as you complete them.</details>
<details><summary><b>[19. Is Computer Vision covered?](./07-CNNs)</b></summary>Yes, including YOLO, ResNet, and Vision Transformers.</details>
<details><summary><b>[20. What's next after finishing?](./16-Projects)</b></summary>Build an end-to-end Capstone project and apply for jobs.</details>

---

## 📈 [Repository Enhancements & Tracking](./README.md)

[![GitHub Repo stars](https://img.shields.io/github/stars/sandaruns2004/Full-ML-DL-CV-and-Data-Science-Roadmap?style=for-the-badge&color=yellow)](./README.md)
[![GitHub forks](https://img.shields.io/github/forks/sandaruns2004/Full-ML-DL-CV-and-Data-Science-Roadmap?style=for-the-badge&color=orange)](./README.md)

### [Progress Tracking Checklist](./README.md)
- [ ] [Phase 00 - 05 completed](./00-Prerequisites)
- [ ] [Phase 06 - 12 completed](./06-Neural-Networks-Foundations)
- [ ] [Phase 13 - 15 completed](./15-ML-In-Production)
- [ ] [All Projects completed](./16-Projects)

### [Star History](./README.md)
*(Placeholder for Star History Graph)*
```html
<a href="https://star-history.com/#sandaruns2004/Full-ML-DL-CV-and-Data-Science-Roadmap&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=sandaruns2004/Full-ML-DL-CV-and-Data-Science-Roadmap&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=sandaruns2004/Full-ML-DL-CV-and-Data-Science-Roadmap&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=sandaruns2004/Full-ML-DL-CV-and-Data-Science-Roadmap&type=Date" />
 </picture>
</a>
```

### [Contribution Graph](./README.md)
*(Placeholder for GitHub Contribution Graph)*

---
> **Keywords for SEO**: [Machine Learning Roadmap](./README.md), [Deep Learning Roadmap](./README.md), [Data Science Roadmap](./README.md), [AI Engineer Roadmap](./README.md), [LLM Roadmap](./README.md), [MLOps Roadmap](./README.md), [Python for AI](./README.md), [Artificial Intelligence Learning Path](./README.md).
"""

with open(os.path.join(base_dir, 'README.md'), 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("README.md has been generated with hyperlinked structure.")
