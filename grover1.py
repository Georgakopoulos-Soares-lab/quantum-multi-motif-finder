from math import ceil, log2, pi, sqrt
from collections import deque
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import MCXGate

# DNA to binary encoding
dna2bits = {'A': (0, 0), 'C': (0, 1), 'G': (1, 0), 'T': (1, 1)}

def int2bits(x, w):
    return [(x >> k) & 1 for k in range(w)]

# classical Aho-Corasick
def build_trie(patterns):
    root = {'fail': None, 'out': []}
    for pid, pat in enumerate(patterns):
        node = root
        for ch in pat:
            node = node.setdefault(ch, {'fail': None, 'out': []})
        node['out'].append((pid, pat))
   
    q = deque()
    for ch in "ACGT":
        if ch in root:
            root[ch]['fail'] = root
            q.append(root[ch])
        else:
            root[ch] = root

    while q:
        cur = q.popleft()
        for ch in "ACGT":
            if ch not in cur:
                continue
            nxt = cur[ch]
            f = cur['fail']
            while ch not in f and f is not root:
                f = f['fail']
            nxt['fail'] = f[ch] if ch in f else root
            nxt['out'].extend(nxt['fail']['out'])
            q.append(nxt)
    root['fail'] = root
    return root

def aho_corasick(text, patterns):
    root = build_trie(patterns)
    node = root
    res = []
    for i, ch in enumerate(text):
        while ch not in node and node is not root:
            node = node['fail']
        node = node[ch] if ch in node else root
        for pid, pat in node['out']:
            res.append((i - len(pat) + 1, pat, pid))
    return sorted(res)

# Quantum Oracle
def oracle_multi(text, patterns):
    L = len(patterns[0])
    n = len(text)
    S = n - L + 1
    n_idx = ceil(log2(S)) if S > 0 else 1

    # define registers
    idx = QuantumRegister(n_idx, 'idx')
    phase = QuantumRegister(1, 'phase')
    char = QuantumRegister(2*L, 'char')
    match = QuantumRegister(1, 'match')
    temp = QuantumRegister(1, 'temp')
    qc = QuantumCircuit(idx, phase, char, match, temp, name='Oracle')

    for k in range(L):
        for pos in range(S):
            j = pos + k
            b1, b0 = dna2bits[text[j]]
            bits = int2bits(pos, n_idx)
            ctrls = [idx[q] for q, b in enumerate(bits) if b]
            neg_ctrls = [idx[q] for q, b in enumerate(bits) if not b]
           
            for q in neg_ctrls: qc.x(q)
            if b0: qc.mcx(ctrls + neg_ctrls, char[2*k])
            if b1: qc.mcx(ctrls + neg_ctrls, char[2*k + 1])
            for q in neg_ctrls: qc.x(q)

    #initialize match qubit for phase kickback
    qc.x(match)
    qc.h(match)

    for pat in patterns:
        qc.x(temp)
        for k, base in enumerate(pat):
            b1, b0 = dna2bits[base]
            if not b0: qc.x(char[2*k])
            if not b1: qc.x(char[2*k + 1])
            qc.ccx(char[2*k], char[2*k + 1], temp[0])
            if not b0: qc.x(char[2*k])
            if not b1: qc.x(char[2*k + 1])
        qc.cz(temp, match)
        for k, base in reversed(list(enumerate(pat))):
            b1, b0 = dna2bits[base]
            if not b0: qc.x(char[2*k])
            if not b1: qc.x(char[2*k + 1])
            qc.ccx(char[2*k], char[2*k + 1], temp[0])
            if not b0: qc.x(char[2*k])
            if not b1: qc.x(char[2*k + 1])
        qc.x(temp)

    for k in reversed(range(L)):
        for pos in reversed(range(S)):
            j = pos + k
            b1, b0 = dna2bits[text[j]]
            bits = int2bits(pos, n_idx)
            ctrls = [idx[q] for q, b in enumerate(bits) if b]
            neg_ctrls = [idx[q] for q, b in enumerate(bits) if not b]
           
            for q in neg_ctrls: qc.x(q)
            if b0: qc.mcx(ctrls + neg_ctrls, char[2*k])
            if b1: qc.mcx(ctrls + neg_ctrls, char[2*k + 1])
            for q in neg_ctrls: qc.x(q)
# Reset match qubit
    qc.h(match)
    qc.x(match)

    return qc.to_gate(label='Oracle')

def diffuser(n):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)
    qc.x(range(n))
    qc.h(range(n))
    return qc.to_gate(label='Diffuser')

def grover_multi_patterns(text, patterns, shots=10000, threshold=0.01):
    L = len(patterns[0])
    n = len(text)
    S = n - L + 1
    n_idx = ceil(log2(S)) if S > 0 else 1

    #quantum registers
    idx = QuantumRegister(n_idx, 'idx')
    phase = QuantumRegister(1, 'phase')
    char = QuantumRegister(2*L, 'char')
    match = QuantumRegister(1, 'match')
    temp = QuantumRegister(1, 'temp')
    creg = ClassicalRegister(n_idx, 'c')
   
    qc = QuantumCircuit(idx, phase, char, match, temp, creg)

    #initialize superposition
    qc.h(idx)
    qc.x(phase)
    qc.h(phase)

    # Create oracle and diffuser
    Uf = oracle_multi(text, patterns)
    D = diffuser(n_idx)

    # calculate optimal iterations
    N = 2**n_idx
    M = len([i for i in range(S) if any(text[i:i+L] == pat for pat in patterns)])
    k = max(1, round((pi/4) * sqrt(N/M))) if M > 0 else 1

    #apply Grover iterations
    for _ in range(k):
        qc.append(Uf, qc.qubits[:n_idx + 1 + 2*L + 2])
        qc.append(D, idx[:])

    # measure
    qc.measure(idx, creg)

    # execute on IBM qiskit
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=shots)
    counts = job.result().get_counts()

    # matches per pattern
    pattern_matches = {pat: set() for pat in patterns}
    total_counts = sum(counts.values())
   
    for bitstr, count in counts.items():
        prob = count / total_counts
        if prob >= threshold:
            i = int(bitstr[::-1], 2)
            if i < S:
                for pat in patterns:
                    if text[i:i+L] == pat:
                        pattern_matches[pat].add(i)

    return pattern_matches, counts

if __name__ == "__main__":
    text = "ACGTACGTACGTACGT" 
    patterns = ["ACGT", "GTAC"] 

    # Classical
    classical_hits = aho_corasick(text, patterns)
    print("Classical Aho-Corasick results:")
    for pos, pat, pid in classical_hits:
        print(f"Pattern '{pat}' found at position {pos}")

    # Quantum
    quantum_matches, histogram = grover_multi_patterns(
        text, patterns, shots=10000, threshold=0.01)
    print("\nQuantum Grover results (per pattern):")
    for pat, positions in quantum_matches.items():
        print(f"Pattern '{pat}' found at positions: {sorted(positions)}")

    # Verification
    classical_positions = {pat: set() for pat in patterns}
    for pos, pat, _ in classical_hits:
        classical_positions[pat].add(pos)

    success = True
    for pat in patterns:
        if classical_positions[pat] != quantum_matches[pat]:
            success = False
            missing = classical_positions[pat] - quantum_matches[pat]
            extra = quantum_matches[pat] - classical_positions[pat]
            if missing:
                print(f"\nMismatch for pattern '{pat}': Quantum missed positions {sorted(missing)}")
            if extra:
                print(f"\nMismatch for pattern '{pat}': Quantum reported extra positions {sorted(extra)}")

    if success:
        print("\nVerification: SUCCESS – Quantum results match classical exactly!")
