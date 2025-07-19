# 🧠 BayesFlow Workshop Materials

Welcome to the official repository for **BayesFlow workshop materials**!  
This repo contains hands-on resources and code examples designed to help you learn and apply modern Bayesian inference techniques using the **[BayesFlow](https://github.com/bayesflow-org/bayesflow)** library.

---

## 📚 Overview

Modern Bayesian inference combines computational tools for estimating, validating, and drawing conclusions from probabilistic models.

In recent years, a new class of **simulation-based inference (SBI)** methods has emerged as a powerful approach for scaling Bayesian inference to **complex models** and **large datasets**. These materials guide you through the foundations and applications of these methods using the BayesFlow library.

---

## 🏗️ Repository Structure

The repository is organized into **one folder per workshop**, each containing self-contained materials:


- Each `workshop` folder contains:
  - 📓 **Jupyter Notebooks**: Interactive code examples and exercises
  - 📁 **Data**: Supporting datasets (if any)  
  - 📄 **README**: Brief instructions and learning goals for the specific workshop
  - 🛠️ **Helpers**: Utility functions and modules to simplify common tasks or reduce boilerplate

---

## ⚙️ Getting Started

### 1. Get the Workshop Materials

You can either **clone the repository** (if you're familiar with Git) or **download it as a ZIP**.

#### 🔧 Option 1: Clone with Git (recommended)

```bash
git clone https://github.com/your-username/bayesflow-workshops.git
cd bayesflow-workshops
```

#### 📦 Option 2: Download ZIP
1. Click the green "Code" button
2. Select "Download ZIP"
3. Extract the ZIP file and navigate into the folder


### 2. Set up the environment
Make sure you have conda installed. Then create the environment from the corresponding workshop folder, e.g.:

```bash
conda env create --file [workshop]/environment.yaml
```

Activate the environment with:

```bash
conda activate bf
```

### 3. Start amortizing!

## 🤝 Contributing
If you find a bug, have suggestions, or want to contribute improvements or additional workshop content, feel free to open an issue or submit a pull request.
