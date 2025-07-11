from math import ceil, log2, pi, sqrt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
import numpy as np

# DNA encoding
DNA2BITS = {'A': (0, 0), 'C': (0, 1), 'G': (1, 0), 'T': (1, 1)}
BASE_LIST = "ACGT"

def build_qram(text):
    """Precompute text for quantum access"""
    return [DNA2BITS[ch] for ch in text]

def quantum_substring_loader(qc, idx_reg, char_reg, text, L, n_idx):
    """Efficient substring loading via multiplexed addressing"""
    n = len(text)
    S = n - L + 1
   
    # load substring character by character
    for k in range(L):
        for pos in range(S):
            addr = pos + k
            if addr >= n:
                continue
            base = text[addr]
            b0, b1 = DNA2BITS[base]
           
            # get bit representation of position
            bit_rep = [(pos >> i) & 1 for i in range(n_idx)]
            flip_list = []
            controls = []
           
            # prepare control qubits
            for i in range(n_idx):
                if bit_rep[i] == 0:
                    flip_list.append(idx_reg[i])
                controls.append(idx_reg[i])
           
            # apply X gates for negative controls
            if flip_list:
                qc.x(flip_list)
           
            # set character bits
            if b0:
                qc.mcx(controls, char_reg[2*k])
            if b1:
                qc.mcx(controls, char_reg[2*k+1])
           
            # uncompute X gates
            if flip_list:
                qc.x(flip_list)

def pattern_hash(patterns):
    """Convert patterns to quantum-hashable integers"""
    hash_dict = {}
    for pat in patterns:
        hash_val = 0
        for j, ch in enumerate(pat):
            b0, b1 = DNA2BITS[ch]
            hash_val |= (b0 << (2*j))
            hash_val |= (b1 << (2*j + 1))
        hash_dict[hash_val] = pat
    return hash_dict

def optimized_oracle(text, patterns):
    """Efficient oracle with pattern hashing"""
    L = len(patterns[0])
    n = len(text)
    S = n - L + 1
    n_idx = ceil(log2(S)) if S > 0 else 1
   
    # Quantum registers
    idx = QuantumRegister(n_idx, 'idx')
    char = QuantumRegister(2*L, 'char')
    match = QuantumRegister(1, 'match')
    hash_reg = QuantumRegister(2*L, 'hash')
    ancillas = QuantumRegister(2, 'anc')  # For MCMT
   
    qc = QuantumCircuit(idx, char, match, hash_reg, ancillas, name='OptimizedOracle')
   
    # load substring
    quantum_substring_loader(qc, idx, char, text, L, n_idx)
   
    #compute pattern hash
    for i in range(2*L):
        qc.cx(char[i], hash_reg[i])
   
    # Phase flip for matching patterns
    pattern_hashes = pattern_hash(patterns)
    for hash_val in pattern_hashes:
        # prepare control bits
        ctrl_bits = []
        for bit in range(2*L):
            if (hash_val >> bit) & 1 == 0:
                qc.x(hash_reg[bit])
            ctrl_bits.append(hash_reg[bit])
       
        # apply multi-controlled Z using H-MCX-H
        qc.h(match)
        qc.mct(ctrl_bits, match[0], ancillas[:], mode='recursion')


        qc.h(match)
       
        # uncompute basis change
        for bit in range(2*L):
            if (hash_val >> bit) & 1 == 0:
                qc.x(hash_reg[bit])
   
    # uncompute hash and substring
    for i in range(2*L):
        qc.cx(char[i], hash_reg[i])
    quantum_substring_loader(qc, idx, char, text, L, n_idx)  # Self-inverse
   
    return qc

def linear_diffuser(n_qubits):
    """Ancilla-assisted O(n) diffuser"""
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
   
    # Registers
    idx = QuantumRegister(n_idx, 'idx')
    char = QuantumRegister(2*L, 'char')
    match = QuantumRegister(1, 'match')
    hash_reg = QuantumRegister(2*L, 'hash')
    ancillas = QuantumRegister(2, 'anc')
    cr = ClassicalRegister(n_idx, 'meas')
   
    qc = QuantumCircuit(idx, char, match, hash_reg, ancillas, cr)
   
    #initialization
    qc.h(idx)
    qc.x(match)
    qc.h(match)
   
    # Grover iterations
    Uf = optimized_oracle(text, patterns)
    D = linear_diffuser(n_idx)
   
    # calculate iterations (quadratic speedup)
    N = 2**n_idx
    # count actual matches
    M = 0
    for i in range(S):
        substr = text[i:i+L]
        if any(substr == pat for pat in patterns):
            M += 1
   
    k = max(1, round((pi/4) * sqrt(N/M))) if M > 0 else 1
    print(f"Running {k} Grover iterations (N={N}, M={M})")
   
    for _ in range(k):
        qc.append(Uf, qc.qubits)
        qc.append(D, idx)
   
    qc.measure(idx, cr)
   
    # execute
    backend = Aer.get_backend('qasm_simulator')
    counts = execute(qc, backend, shots=shots, optimization_level=0).result().get_counts()
   
    # post-processing
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

# test and verify
if __name__ == "__main__":
    text = "ACGTACGTACGTACGT"
    patterns = ["ACGT", "GTAC"]
   
    # classical for verification
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
   
    # Quantum
    quantum_matches, counts = grover_optimized(text, patterns, shots=1000)
    print("\nQuantum Results:")
    for pat, positions in quantum_matches.items():
        print(f"Pattern '{pat}' found at positions: {sorted(positions)}")
   
    # verification
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
        print("\nVerification SUCCESS: Quantum matches classical results")
    else:
        print("\nVerification FAILED")
   
    print("\nSample measurement counts:")
    for bitstr, count in list(counts.items())[:5]:
        print(f"Index {bitstr}: {count} shots")
