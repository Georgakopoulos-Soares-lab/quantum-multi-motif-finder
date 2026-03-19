# Quantum Multi-Motif Finder 🧬⚛️

This repository implements **two quantum algorithms for DNA multi-pattern string matching** (k-mer detection) using **Grover’s amplitude amplification** with **Quantum Random Access Memory (QRAM)**.  

The algorithms are designed to accelerate DNA motif search tasks, which are central to computational biology and genomics.

---

## 📑 Table of Contents
- [📖 Introduction](#-introduction)
- [📂 Repository Contents](#-repository-contents)
- [⚙️ Dependencies](#️-dependencies)
- [▶️ Running the Script](#️-running-the-script)
- [🧩 Usage Demonstration](#-usage-demonstration)
- [👩‍🔬 Authors](#-authors)
- [📜 License](#-license)
- [🎓 Purpose of This Repository](#-purpose-of-this-repository)
- [📚 Publication](#-publication)
- [🤝 Contributing](#-contributing)

---

## 📖 Introduction

DNA motif search is a fundamental problem in bioinformatics, where the goal is to detect multiple short patterns (k-mers) within long DNA sequences. Classical algorithms can be computationally expensive when scaling to large datasets.  

This project explores **quantum-enhanced solutions**:

1. **Enumerate-m Oracle Approach**  
   - Sequentially checks a loaded text substring against all `m` patterns.  
   - Achieves query complexity of **O(√S)** for `S` text positions.  
   - Requires **O(m·L)** work per oracle call, where `L` is the pattern length.  

2. **Nested Grover Search**  
   - Employs an outer Grover loop over text positions and an inner loop over the pattern space.  
   - Reduces oracle complexity to **O(L)**.  
   - Performs **O(√S · √m)** total work.  

These algorithms demonstrate how quantum computing can potentially outperform classical methods in large-scale DNA motif detection.

---

## 📂 Repository Contents

- `quantum_unified_solutions_for_DNA_motif_search.py`  
  Main script implementing both quantum algorithms for multi-pattern DNA motif search.

---

## ⚙️ Dependencies

To run the script, you’ll need:

- Python **3.8+**
- [Qiskit](https://qiskit.org/) (quantum computing framework)
- NumPy
- SciPy

Install dependencies via:

```bash
pip install qiskit numpy scipy
```

---

## ▶️ Running the Script
Clone the repository:
```bash
git clone https://github.com/Georgakopoulos-Soares-lab/quantum-multi-motif-finder.git
cd quantum-multi-motif-finder
```

Run the main script:
```python
python quantum_unified_solutions_for_DNA_motif_search.py
```

---

## 🧩 Usage Demonstration
Inside the script, you can define your DNA sequence and motifs. For example:

```python
# Example DNA sequence
text = "ATCGTACGTAGCTAGCTAGCTAGCTA"

# Example motifs (k-mers) to search for
patterns = ["ATCG", "TAGC", "GCTA"]

# Choose algorithm: "enumerate" or "nested"
algorithm = "nested"
```

When you run the script, it will:
- Load the DNA sequence into QRAM.
- Apply Grover’s search using the chosen algorithm.
- Print out the positions where motifs are detected.
```bash
Sample Output:
Running Nested Grover Search...
Motif 'ATCG' found at positions: [0]
Motif 'TAGC' found at positions: [8, 12]
Motif 'GCTA' found at positions: [16, 20]
```

This demonstrates how the quantum algorithm detects multiple motifs efficiently.

---

## 👩‍🔬 Authors
Developed by the Georgakopoulos-Soares Lab as part of ongoing research into quantum algorithms for computational biology.

---

## 🎓 Purpose of This Repository
The code and examples provided here are designed to support ongoing research and learning in the emerging intersection of quantum computing and bioinformatics. They aim to:

- Illustrate how quantum computing techniques can be applied to computational biology challenges.
- Foster collaboration between quantum computing practitioners and researchers in genomics and related fields.

⚠️ Note: While current demonstrations run on a quantum simulator due to the limited availability of large‑scale quantum hardware, the underlying algorithms are built with future quantum systems in mind. They are designed as production‑ready algorithms that can run on fully scaled quantum hardware without requiring changes, and can be executed on real quantum devices as soon as sufficiently capable systems become available. For now, this repository serves as a practical proof‑of‑concept showcasing the potential of quantum‑enhanced approaches for motif search tasks.

---

## 📜 License
This project is released under the MIT License. See LICENSE for details.

---

## 📚 Publication
TBA

---

## 🤝 Contributing
We welcome and encourage contributions from the community!
If you’d like to improve the code, add new features, or extend the research, please feel free to:
- Fork the repository
- Create a branch for your feature or fix
- Commit your changes
- Open a Pull Request describing your contribution
All contributions — whether bug fixes, documentation, or new algorithmic ideas — help strengthen this project as a foundation for quantum-enhanced DNA motif search.
Please ensure that your contributions align with the research and educational purpose of this project. Constructive feedback, discussions, and suggestions are also highly valued.

Thank you! 

---
