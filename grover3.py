from math import ceil, log2, pi, sqrt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
import numpy as np
from typing import List

# DNA encoding
DNA2BITS = {'A': (0, 0), 'C': (0, 1), 'G': (1, 0), 'T': (1, 1)}

class QuantumTrie:
    #Quantum-optimized trie structure for pattern matching
    def __init__(self, patterns: List[str]):
        self.patterns = patterns
        self.max_length = max(len(p) for p in patterns) if patterns else 0
        self.pattern_masks = {p: [b for char in p for b in DNA2BITS[char]] for p in patterns}

def controlled_text_loader(qc, control_qubits, text_reg, text_window):
    #Load text window when control qubits are |1⟩
    for char_idx, char in enumerate(text_window):
        if char in DNA2BITS and 2*char_idx+1 < len(text_reg):
            b0, b1 = DNA2BITS[char]
            if b0:
                if control_qubits:
                    qc.mcx(control_qubits, text_reg[2*char_idx])
                else:
                    qc.x(text_reg[2*char_idx])
            if b1:
                if control_qubits:
                    qc.mcx(control_qubits, text_reg[2*char_idx+1])
                else:
                    qc.x(text_reg[2*char_idx+1])

def quantum_text_loader(qc, idx_reg, text_reg, text, window_size):
    #Load text windows into quantum superposition
    n = len(text)
    max_positions = min(2**len(idx_reg), n - window_size + 1)
   
    for pos in range(max_positions):
        pos_binary = format(pos, f'0{len(idx_reg)}b')
        controls = []
        for i, bit in enumerate(pos_binary):
            if bit == '0':
                qc.x(idx_reg[i])
                controls.append(idx_reg[i])
        controlled_text_loader(qc, controls, text_reg, text[pos:pos+window_size])
        for qubit in controls:
            qc.x(qubit)

def pattern_oracle(qc, text_reg, patterns, trie, match_qubit=None):
    #Oracle that flips phase for states matching any pattern
    if match_qubit is None:
        match_qubit = QuantumRegister(1, 'match')
        qc.add_register(match_qubit)
    else:
        # Reset the match qubit to |0>
        qc.reset(match_qubit[0])

   
    for pattern in patterns:
        if pattern not in trie.pattern_masks: continue
        pattern_bits = trie.pattern_masks[pattern]
        x_applied = []
       
        for i, bit in enumerate(pattern_bits):
            if i >= len(text_reg): break
            if bit == 0:
                qc.x(text_reg[i])
                x_applied.append(i)
       
        # Multi-controlled NOT
        controls = [text_reg[i] for i in range(min(len(pattern_bits), len(text_reg)))]
        if controls:
            qc.mcx(controls, match_qubit[0])
       
        qc.z(match_qubit[0])
       
        # Uncompute
        if controls: qc.mcx(controls, match_qubit[0])
        for i in x_applied: qc.x(text_reg[i])
   
    return match_qubit

def grover_diffuser(qc, qubits):
    #Grover diffusion operator
    # Apply transformation |s> -> |00..0> (H gates)
    for qubit in qubits:
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X gates)
    for qubit in qubits:
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    # Apply transformation |11..1> -> |00..0>
    for qubit in qubits:
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in qubits:
        qc.h(qubit)

def quantum_pattern_search(text: str, patterns: List[str], shots: int = 1000):
    #Quantum multiple pattern matching using Grover's algorithm
    if not patterns or not text: return {}
    L = len(patterns[0])
    if not all(len(p) == L for p in patterns) or L > len(text): return {}
   
    trie = QuantumTrie(patterns)
    n, S = len(text), len(text) - L + 1
    n_idx = ceil(log2(S)) if S > 1 else 1
   
    # Classical preprocessing
    classical_matches = sum(1 for i in range(S) if text[i:i+L] in patterns)
    if classical_matches == 0: return {}
   
    # Quantum circuit setup
    idx_reg = QuantumRegister(n_idx, 'idx')
    text_reg = QuantumRegister(2*L, 'text')
    match_qubit = QuantumRegister(1, 'match')
    classical_reg = ClassicalRegister(n_idx, 'result')
    qc = QuantumCircuit(idx_reg, text_reg, match_qubit, classical_reg)
   
    # Initialize superposition
    qc.h(idx_reg)
   
    # Grover iterations
    N = 2**n_idx
    theta = np.arcsin(sqrt(classical_matches/N))
    iterations = min(8, max(1, round(pi/(4*theta)-0.5)))
   
    for _ in range(iterations):
        quantum_text_loader(qc, idx_reg, text_reg, text, L)
        pattern_oracle(qc, text_reg, patterns, trie, match_qubit)

        quantum_text_loader(qc, idx_reg, text_reg, text, L)  # Unload
        if _ < iterations-1: grover_diffuser(qc, idx_reg)
   
    # Measurement
    qc.measure(idx_reg, classical_reg)
   
    # Execute
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=shots, optimization_level=1)
    counts = job.result().get_counts()
   
    # Process results
    results = {}
    for bitstring, count in counts.items():
        pos = int(bitstring[::-1], 2)
        if pos < S:
            substring = text[pos:pos+L]
            if substring in patterns:
                if substring not in results: results[substring] = set()
                results[substring].add(pos)
   
    return results

# Example usage
if __name__ == "__main__":
    text = "GCATATAAGGCCAATGCGATATAAATGCCAATTATAGC"
    patterns = ["TATA", "AATG", "CGCA"]
   
    print("Initiating quantum DNA pattern matching..")
    quantum_results = quantum_pattern_search(text, patterns)
   
    print("\nQuantum matches found:")
    for pattern, positions in quantum_results.items():
        print(f"{pattern}: {sorted(positions)}")
