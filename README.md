# EuroSAT Image Classification  

## 📌 Project Overview  
This project explores **land use classification** using the [EuroSAT dataset](https://zenodo.org/records/7711810#.ZAm3k-zMKEA), which contains labeled satellite images across 10 land use and land cover categories (e.g., residential, industrial, agricultural).  

I implemented both **custom CNN architectures** and **transfer learning** with ResNet-34 in PyTorch, achieving strong performance on the dataset.  

---

## 🚀 Key Features  
- **Transfer Learning (ResNet-34):** Achieved **98.6% accuracy** on the EuroSAT dataset  
- **Custom CNN:** Designed and trained a self-built CNN that reached **90%+ accuracy**  
- **Reproducible Pipeline:** End-to-end workflow for preprocessing, training, evaluation, and comparison of models  
- **Model Comparison:** Analyzed trade-offs between accuracy and computational efficiency for self-built vs. pre-trained models  

---

## 🛠️ Tech Stack  
- **Languages & Tools:** Python, Google Colab, Git  
- **Libraries & Frameworks:** PyTorch, torchvision, NumPy, pandas, matplotlib


---

## 📊 Results  
| Model            | Accuracy | Train Time | Notes                                   |
|------------------|----------|------------|-----------------------------------------|
| ResNet-34 (TL)   | **98.6%** | ~45 min    | Transfer learning, fine-tuned on EuroSAT |
| Custom CNN       | 90%       | ~10 min    | 3 convolutional layers, trained from scratch |


## 📑 Full Visualizations  

All training curves, confusion matrices, and results are compiled here:  
[📂 EuroSAT Visualizations (PDF)](visualizations.pdf)


---


## 📖 References  
- [EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://arxiv.org/abs/1709.00029)  
- [PyTorch Documentation](https://pytorch.org/)  



