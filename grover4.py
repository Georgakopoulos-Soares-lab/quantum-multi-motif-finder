from math import ceil, log2, pi, sqrt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import MCMT
import numpy as np

# DNA encoding
DNA2BITS = {'A': (0, 0), 'C': (0, 1), 'G': (1, 0), 'T': (1, 1)}
BASE_LIST = "ACGT"

class QROMSubstringLoader:
    #QROM-based substring loader
    def __init__(self, text, L, n_idx):
        self.text = text
        self.L = L

        self.n_idx = n_idx
        self.S = len(text) - L + 1
       
    def build_qrom_circuit(self, addr_reg, data_reg):
        qc = QuantumCircuit(addr_reg, data_reg, name="QROM_Loader")
        n_data = 2 * self.L  # 2 bits per character
       
        #QROM structure: Tree-like multiplexers
        for k in range(self.L):
            # Precompute character bits for all addresses
            char_bits = [DNA2BITS[self.text[i + k]] for i in range(self.S)]
           
            # Load bit 0 of character k
            self._add_qrom_layer(qc, addr_reg, data_reg[2*k],
                                [bits[0] for bits in char_bits])
           
            # Load bit 1 of character k
            self._add_qrom_layer(qc, addr_reg, data_reg[2*k+1],
                                [bits[1] for bits in char_bits])
        return qc
   
    def _add_qrom_layer(self, qc, addr_reg, target_qubit, bit_values):
       
        #add multiplexer layer for one bit
        #base case for single address qubit
        if len(addr_reg) == 1:
            if bit_values[0]:
                qc.cx(addr_reg[0], target_qubit)
            if bit_values[1]:
                qc.x(addr_reg[0])
                qc.cx(addr_reg[0], target_qubit)
                qc.x(addr_reg[0])
            return
       
        #recursive tree decomposition
        mid = len(addr_reg) // 2
        reg_low = addr_reg[:mid]
        reg_high = addr_reg[mid:]
       
        #compute left and right patterns
        left_pattern = self._compute_subpattern(reg_low, bit_values, 0)
        right_pattern = self._compute_subpattern(reg_high, bit_values, mid)
       
        #recursive construction
        self._add_qrom_layer(qc, reg_low, target_qubit, left_pattern)
        self._add_qrom_layer(qc, reg_high, target_qubit, right_pattern)

    def _compute_subpattern(self, sub_reg, bit_values, offset):
        #compute bit pattern for address subspace
        n_bits = len(sub_reg)
        pattern = []
        for i in range(2**n_bits):
            addr = i << offset
            if addr < len(bit_values):
                pattern.append(bit_values[addr])
            else:
                pattern.append(0)
        return pattern

def optimized_oracle(text, patterns):
    L = len(patterns[0])
    n = len(text)
    S = n - L + 1
    n_idx = ceil(log2(S)) if S > 0 else 1
   
    #quantum registers
    idx = QuantumRegister(n_idx, 'idx')
    char = QuantumRegister(2*L, 'char')
    match = QuantumRegister(1, 'match')
    ancillas = QuantumRegister(2, 'anc')  # Για MCMT
   
    qc = QuantumCircuit(idx, char, match, ancillas, name='OptimizedOracle')
   
    #QROM substring loader
    qrom_loader = QROMSubstringLoader(text, L, n_idx)
    qc.append(qrom_loader.build_qrom_circuit(idx, char), idx[:] + char[:])
   
    #phse flip
    pattern_hashes = pattern_hash(patterns)
    for hash_val in pattern_hashes:
        # apply X to have our pattern on |1...1⟩
        for bit in range(2*L):
            if (hash_val >> bit) & 1 == 0:
                qc.x(char[bit])
       
        # MCX - phase kickback at |match⟩ = |−⟩
        qc.mcx(
            control_qubits=char[:],
            target_qubit=match[0],
            ancilla_qubits=ancillas[:],
            mode='recursion'
        )
       
        # Uncompute X
        for bit in range(2*L):
            if (hash_val >> bit) & 1 == 0:
                qc.x(char[bit])
   
    # Uncompute QROM
    qc.append(qrom_loader.build_qrom_circuit(idx, char).inverse(), idx[:] + char[:])
   
    return qc

def pattern_hash(patterns):
    #Convert patterns to quantum-hashable integers
    hash_dict = {}
    for pat in patterns:
        hash_val = 0
        for j, ch in enumerate(pat):
            b0, b1 = DNA2BITS[ch]
            hash_val |= (b0 << (2*j))
            hash_val |= (b1 << (2*j + 1))
        hash_dict[hash_val] = pat
    return hash_dict

def linear_diffuser(n_qubits):
    #Ancilla-assisted  diffuser
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits-1)
    if n_qubits > 1:
        qc.mcx(list(range(n_qubits-1)), n_qubits-1)
    else:
        qc.x(0)
    qc.h(n_qubits-1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc

def grover_optimized(text, patterns, shots=1000):
    L = len(patterns[0])
    n = len(text)
    S = n - L + 1
    n_idx = ceil(log2(S)) if S > 0 else 1
   
    #quantum registers
    idx = QuantumRegister(n_idx, 'idx')
    char = QuantumRegister(2*L, 'char')
    match = QuantumRegister(1, 'match')
    ancillas = QuantumRegister(2, 'anc')
    cr = ClassicalRegister(n_idx, 'meas')
   
    qc = QuantumCircuit(idx, char, match, ancillas, cr)
   
    #init
    qc.h(idx)
    qc.x(match)
    qc.h(match)  # |match⟩ = |−⟩
   
    #Grover reps
    Uf = optimized_oracle(text, patterns)
    D = linear_diffuser(n_idx)
   
    N = 2**n_idx
    M = 0
    for i in range(S):
        substr = text[i:i+L]
        if any(substr == pat for pat in patterns):
            M += 1
   
    k = max(1, round((pi/4) * sqrt(N/M))) if M > 0 else 1
    print(f"Running {k} Grover iterations (N={N}, M={M})")
   
    for _ in range(k):
        qc.append(Uf, [*idx, *char, *match, *ancillas])
        qc.append(D, idx)
   
    qc.measure(idx, cr)
   
    #execution step
    backend = Aer.get_backend('qasm_simulator')
    counts = execute(qc, backend, shots=shots, optimization_level=0).result().get_counts()
   
    pattern_matches = {pat: set() for pat in patterns}
    total_counts = sum(counts.values())
    threshold = 0.01
   
    for bitstr, count in counts.items():
        prob = count / total_counts
        if prob >= threshold:
            i = int(bitstr[::-1], 2)
            if i < S:
                substr = text[i:i+L]
                for pat in patterns:
                    if substr == pat:
                        pattern_matches[pat].add(i)
   
    return pattern_matches, counts

#test and verify
if __name__ == "__main__":
    text = "ACGTACGTACGTACGT"
    patterns = ["ACGT", "GTAC"]
   
    #classical verification
    def classical_match(text, patterns):
        L = len(patterns[0])
        matches = {pat: set() for pat in patterns}
        S = len(text) - L + 1
        for i in range(S):
            substr = text[i:i+L]
            for pat in patterns:
                if substr == pat:
                    matches[pat].add(i)
        return matches
   
    classical_matches = classical_match(text, patterns)
    print("Classical Results:")
    for pat, positions in classical_matches.items():
        print(f"Pattern '{pat}' found at positions: {sorted(positions)}")
   
    #quantum grover for multiple patterns execution
    quantum_matches, counts = grover_optimized(text, patterns, shots=1000)
    print("\nQuantum Results:")
    for pat, positions in quantum_matches.items():
        print(f"Pattern '{pat}' found at positions: {sorted(positions)}")
   
    #validation/verification
    success = True
    for pat in patterns:
        if classical_matches[pat] != quantum_matches[pat]:
            success = False
            missing = classical_matches[pat] - quantum_matches[pat]
            extra = quantum_matches[pat] - classical_matches[pat]
            if missing:
                print(f"Error: Quantum missed {pat} at positions {sorted(missing)}")
            if extra:
                print(f"Error: Quantum found extra {pat} at positions {sorted(extra)}")
   
    if success:
        print("\nVerification Correct: Quantum matches classical results")
    else:
        print("\nVerification FAILED")
   
    print("\nSample measurement counts:")
    for bitstr, count in list(counts.items())[:5]:
        print(f"Index {bitstr}: {count} shots")
      
