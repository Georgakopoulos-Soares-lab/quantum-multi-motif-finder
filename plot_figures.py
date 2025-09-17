import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import circuit_drawer
import seaborn as sns
import math

# Set a consistent style for all plots
plt.style.use('default')
sns.set_palette("colorblind")

# --- FIGURE 1: High-Level Algorithm Flowchart
mermaid_code = """
flowchart TD
    A[Start] --> B[Initialize Quantum Registers]
    B --> C[Prepare Superposition<br>over Text Indices idx]
    C --> D[Apply Grover Operator<br>G Times]
    D --> E[QRAM Query:<br>Load substring into data]
    E --> F[Oracle:<br>Check data against patterns]
    F --> G[QRAM Uncompute]
    G --> H[Diffuser:<br>Amplify Solution States]
    H --> I{Loop G Times?}
    I -- Yes --> D
    I -- No --> J[Measure idx Register]
    J --> K[Output Potential Match Indices]
"""

print("Figure 1: Mermaid Code for Flowchart")
print("Copy the below code into a Mermaid editor (e.g., https://mermaid.live/):")
print(mermaid_code)
print("\n")

# --- FIGURE 2: Quantum Circuit Diagram for a Small Instance ---
print("Generating Figure 2: Quantum Circuit Diagram...")

# Build a very small circuit for visualization
text_small = "ACGT"
patterns_small = ["AC", "GT"]
L_small = 2

# Use the Inner Grover demo circuit for a small example
qc_small, _ = build_qram_inner_grover_circuit(
    text_small, patterns_small, L_small,
    outer_iters=1,
    inner_iters=1,
    qram_mode='demo',
    use_ancillas=True
)

# Draw the circuit and save it
fig_circuit = circuit_drawer(qc_small, output='mpl', style={'name': 'bw'}, plot_barriers=False, scale=0.7)
fig_circuit.savefig("figure_2_circuit_diagram.png", dpi=300, bbox_inches='tight')
print("Figure 2 saved as 'figure_2_circuit_diagram.png'")
print("\n")

# --- FIGURE 3: Theoretical Scaling Comparison ---
print("Generating Figure 3: Theoretical Scaling Comparison...")

# Generate data for plotting
n_values = np.logspace(1, 8, 100, base=10)  # n from 10 to 100 million
m = 10000  # Fixed number of patterns
L = 20     # Fixed pattern length

# Classical Aho-Corasick: O(n)
classical_complexity = n_values

# Quantum Enumerate-m: O(sqrt(n) * m)
quantum_enum_complexity = np.sqrt(n_values) * m

# Quantum Inner Grover: O(sqrt(n) * sqrt(m))
quantum_inner_complexity = np.sqrt(n_values) * np.sqrt(m)

# Create the plot
plt.figure(figsize=(10, 6))
plt.loglog(n_values, classical_complexity, label='Aho-Corasick: O(n)', linewidth=3)
plt.loglog(n_values, quantum_enum_complexity, label='Quantum Enumerate-m: O(√n · m)', linewidth=3, linestyle='--')
plt.loglog(n_values, quantum_inner_complexity, label='Quantum Inner Grover: O(√n · √m)', linewidth=3)

plt.xlabel('Text Size (n)', fontsize=12)
plt.ylabel('Time Complexity (Operations)', fontsize=12)
plt.title('Theoretical Scaling Comparison: Quantum vs. Classical Pattern Matching', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.4)

# Add annotations
plt.annotate('Quantum Advantage Region', xy=(1e6, 1e7), xytext=(1e5, 1e8),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=11, ha='center')

plt.tight_layout()
plt.savefig("figure_3_scaling_comparison.png", dpi=300)
print("Figure 3 saved as 'figure_3_scaling_comparison.png'")
print("\n")

# --- FIGURE 4: Bucket-Brigade QRAM Architecture Schematic ---
print("Generating Figure 4: Bucket-Brigade QRAM Schematic...")

# Create a schematic representation using matplotlib
fig, ax = plt.subplots(figsize=(12, 8))

# Draw the binary tree structure
def draw_tree(ax, x, y, dx, dy, depth, max_depth, address_bits):
    if depth > max_depth:
        return
       
    # Draw current node
    if depth < max_depth:
        node_color = 'lightblue'
        node_label = f'Routing Qubit\ndepth {depth}'
    else:
        node_color = 'lightgreen'
        mem_value = ['A', 'C', 'G', 'T'][int(x/(dx*2)) % 4]
        node_label = f'Memory Cell\n{mem_value}'
   
    ax.add_patch(plt.Circle((x, y), 0.4, fill=True, color=node_color, ec='black', lw=2))
    ax.text(x, y, node_label, ha='center', va='center', fontsize=9 if depth < max_depth else 8)
   
    if depth < max_depth:
        # Draw left child
        ax.plot([x, x-dx], [y, y-dy], 'k-', lw=2)
        draw_tree(ax, x-dx, y-dy, dx/2, dy, depth+1, max_depth, address_bits)
       
        # Draw right child
        ax.plot([x, x+dx], [y, y-dy], 'k-', lw=2)
        draw_tree(ax, x+dx, y-dy, dx/2, dy, depth+1, max_depth, address_bits)
       
        # Add address bit labels
        if depth == 0:
            ax.text(x-dx-0.2, y-dy/2, f'|0⟩', fontsize=12, fontweight='bold', color='red')
            ax.text(x+dx-0.2, y-dy/2, f'|1⟩', fontsize=12, fontweight='bold', color='red')

# Draw the tree
draw_tree(ax, 0, 0, 4, 1.5, 0, 3, "01")

# Add title and explanation
ax.set_title('Bucket-Brigade QRAM Architecture Schematic', fontsize=16, pad=20)
ax.text(0, -7, 'Query Path for Address |01⟩ Highlighted in Red', ha='center', fontsize=12, style='italic')

# Set axis properties
ax.set_xlim(-8, 8)
ax.set_ylim(-7, 2)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig("figure_4_qram_schematic.png", dpi=300)
print("Figure 4 saved as 'figure_4_qram_schematic.png'")
print("\n")

#FIGURE 5: Simulation Results for Toy Example
print("Generating Figure 5: Simulation Results for Toy Example...")

# Run simulation on the small circuit
counts = simulate_demo_circuit(qc_small, shots=1024)

# Prepare data for plotting
indices = [f'{i}' for i in range(4)]  # For n=4, L=2 → indices 0, 1, 2
values = [counts.get(f'{i:02b}', 0) for i in range(4)]  # Get counts for each index

# Create bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(indices, values, color=['skyblue' if i in [0, 2] else 'lightgray' for i in range(4)])
plt.xlabel('Starting Index (i)', fontsize=12)
plt.ylabel('Measurement Count', fontsize=12)
plt.title('Quantum Pattern Matching Simulation Results\n(Text="ACGT", Patterns=["AC", "GT"])', fontsize=14)

# Highlight the correct matches (indices 0 and 2 for patterns "AC" and "GT")
bars[0].set_color('salmon')
bars[2].set_color('salmon')

# Add value labels on top of bars
for i, v in enumerate(values):
    plt.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')

# Add explanation text
plt.text(1.5, max(values) * 0.8, 'Red bars indicate correct pattern matches\n(Index 0: "AC", Index 2: "GT")',
         ha='center', va='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("figure_5_simulation_results.png", dpi=300)
print("Figure 5 saved as 'figure_5_simulation_results.png'")
print("\n")

print("All figures generated successfully!")
print("Please check your directory for the following files:")
print("1. figure_2_circuit_diagram.png")
print("2. figure_3_scaling_comparison.png")
print("3. figure_4_qram_schematic.png")
print("4. figure_5_simulation_results.png")
print("5. Figure 1 requires manual creation using the provided Mermaid code")
