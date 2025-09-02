from math import ceil, log2, pi
import numpy as np
from typing import List, Optional, Dict, Any
from collections import defaultdict
# Qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Instruction
from qiskit.circuit.library import StatePreparation    
from qiskit_aer import AerSimulator

#helpers/utilities
DNA2BITS = {'A': (0, 0), 'C': (0, 1), 'G': (1, 0), 'T': (1, 1)}

def pattern_bits_2L(pattern: str) -> List[int]:
    """Return a little-endian 2L-bit array encoding a DNA pattern."""
    bits = []
    for ch in pattern:
        b0, b1 = DNA2BITS[ch]
        bits.extend([b0, b1])  # data[0] -> least significant
    return bits

def opaque_instruction(name: str, num_qubits: int, num_clbits: int = 0, params=None) -> Instruction:
    """Create an blackbox instruction """
    return Instruction(name=name, num_qubits=num_qubits, num_clbits=num_clbits, params=params or [])

#cost tracking
class CostTracker:
    """cost accounting for asymptotic work and op counts."""
    def __init__(self):
        self.counters = defaultdict(int)

    def inc(self, key: str, val: int = 1):
        self.counters[key] += val

    def add(self, key: str, val: int):
        self.counters[key] += val

    def snapshot(self) -> Dict[str, int]:
        return dict(self.counters)

# wrapped ops with cost counting
def op_x(qc: QuantumCircuit, q, cost: CostTracker):
    qc.x(q)
    cost.inc('x')

def op_h(qc: QuantumCircuit, q, cost: CostTracker):
    qc.h(q)
    cost.inc('h')

def op_mcx(qc: QuantumCircuit, ctrls, tgt, cost: CostTracker, anc: Optional[QuantumRegister] = None):
    """MCX with optional recursion mode/ancillas; count as one MCX."""
    if anc is not None and len(ctrls) > 4:
        qc.mcx(list(ctrls), tgt, ancilla_qubits=list(anc), mode='recursion')
    else:
        qc.mcx(list(ctrls), tgt)
    cost.inc('mcx')

#state preparation on address (unitary)
def prepare_address_superposition(qc: QuantumCircuit, addr: QuantumRegister, S: int, cost: CostTracker):
    """Prepare uniform superposition over first S states of addr (unitary)."""
    n_idx = len(addr)
    N = 1 << n_idx
    if S == N:
        for q in addr:
            op_h(qc, q, cost)
    else:
        amps = np.zeros(N, dtype=complex)
        amps[:S] = 1/np.sqrt(S)
        prep = StatePreparation(amps, normalize=True)
        qc.append(prep, addr)
        cost.inc('state_prep_unitary')

#Diffuser on a register (explicit MCX/H)
def diffuser_on_register(qc: QuantumCircuit, reg: QuantumRegister, cost: CostTracker,
                         anc: Optional[QuantumRegister] = None):
    """Standard Grover diffuser on a register (MCX/H explicit)."""
    for q in reg:
        op_h(qc, q, cost)
    for q in reg:
        op_x(qc, q, cost)

    if len(reg) == 1:
        qc.z(reg[0])
        cost.inc('z')
    else:
        op_h(qc, reg[-1], cost)
        op_mcx(qc, reg[:-1], reg[-1], cost, anc)
        op_h(qc, reg[-1], cost)

    for q in reg:
        op_x(qc, q, cost)
    for q in reg:
        op_h(qc, q, cost)

#equality test: data == pdat (explicit, non-opaque)
def equality_flip_on_match(qc: QuantumCircuit,
                           data: QuantumRegister,
                           pdat: QuantumRegister,
                           target,  # single qubit
                           cost: CostTracker,
                           anc_eq: Optional[List] = None,
                           anc: Optional[QuantumRegister] = None):
    """Flip target if data == pdat, using anc_eq for XNOR bits (explicit, non-blackbox)."""
    L2 = len(data)
    if anc_eq is None or len(anc_eq) < L2:
        raise ValueError("Need anc_eq of length >= len(data) (2L) for comparator.")

    #build XNOR bits into anc_eq
    for k in range(L2):
        qc.cx(data[k], anc_eq[k]); cost.inc('cx')
        qc.cx(pdat[k], anc_eq[k]); cost.inc('cx')
        op_x(qc, anc_eq[k], cost)  #invert XOR -> XNOR (1 when equal)

    #one big MCX into target
    op_mcx(qc, anc_eq, target, cost, anc)

    #uncompute XNOR
    for k in reversed(range(L2)):
        op_x(qc, anc_eq[k], cost)
        qc.cx(pdat[k], anc_eq[k]); cost.inc('cx')
        qc.cx(data[k], anc_eq[k]); cost.inc('cx')

#demo QRAM-like loaders (explicit, simulatable)
def qram_text_query_demo(qc: QuantumCircuit, text: str, L: int,
                         idx: QuantumRegister, data: QuantumRegister,
                         anc: Optional[QuantumRegister], cost: CostTracker):
    """Demo QRAM-like text query: load 2L data bits for each of S = len(text)-L+1 addresses."""
    S = len(text) - L + 1
    for k in range(L):
        for i in range(S):
            b0, b1 = DNA2BITS[text[i + k]]
            bits = format(i, f'0{len(idx)}b')[::-1]  # little-endian controls
            # Select address |i>
            for qb, b in zip(idx, bits):
                if b == '0':
                    op_x(qc, qb, cost)
            # Toggle data bits if 1
            if b0 == 1:
                op_mcx(qc, list(idx), data[2*k], cost, anc)
            if b1 == 1:
                op_mcx(qc, list(idx), data[2*k+1], cost, anc)
            # Unselect address
            for qb, b in zip(idx, bits):
                if b == '0':
                    op_x(qc, qb, cost)
    cost.add('demo_qram_text_ops', S * L)

def qram_pattern_query_demo(qc: QuantumCircuit, patterns: List[str], L: int,
                            pid: QuantumRegister, pdat: QuantumRegister,
                            anc: Optional[QuantumRegister], cost: CostTracker):
    """Demo QRAM-like pattern query: load 2L data bits for each of m = len(patterns) patterns."""
    m = len(patterns)
    for j, pat in enumerate(patterns):
        bits = format(j, f'0{len(pid)}b')[::-1]
        #select pid == j
        for qb, b in zip(pid, bits):
            if b == '0':
                op_x(qc, qb, cost)
        # write pattern bits
        pbits = pattern_bits_2L(pat)
        for k, b in enumerate(pbits):
            if b == 1:
                op_mcx(qc, list(pid), pdat[k], cost, anc)
        #unselect
        for qb, b in zip(pid, bits):
            if b == '0':
                op_x(qc, qb, cost)
    cost.add('demo_qram_pattern_ops', m * L)

#theory: opaque QRAM gates - the readiness effort
def append_qram_text_opaque(qc: QuantumCircuit, idx: QuantumRegister, data: QuantumRegister,
                            L: int, cost: CostTracker):
    instr = opaque_instruction('qram_text_query', num_qubits=len(idx) + len(data), params=[L])
    qc.append(instr, list(idx) + list(data))
    cost.inc('qram_text_query')

def append_qram_text_unquery_opaque(qc: QuantumCircuit, idx: QuantumRegister, data: QuantumRegister,
                                    L: int, cost: CostTracker):
    instr = opaque_instruction('qram_text_unquery', num_qubits=len(idx) + len(data), params=[L])
    qc.append(instr, list(idx) + list(data))
    cost.inc('qram_text_unquery')

def append_qram_pattern_opaque(qc: QuantumCircuit, pid: QuantumRegister, pdat: QuantumRegister,
                               L: int, cost: CostTracker):
    instr = opaque_instruction('qram_pattern_query', num_qubits=len(pid) + len(pdat), params=[L])
    qc.append(instr, list(pid) + list(pdat))
    cost.inc('qram_pattern_query')

#enumerate-m Oracle (explicit, non-opaque)
def oracle_enumerate_patterns(qc: QuantumCircuit,
                              data: QuantumRegister,
                              match: QuantumRegister,
                              patterns: List[str],
                              L: int,
                              cost: CostTracker,
                              anc: Optional[QuantumRegister] = None):
    """Oracle that flips match if data matches any of the given patterns."""
    for pat in patterns:
        pbits = pattern_bits_2L(pat)
        # X on zeros to map equality to all-ones
        for k, b in enumerate(pbits):
            if b == 0:
                op_x(qc, data[k], cost)
        #Flip match if all ones
        op_mcx(qc, list(data), match[0], cost, anc)
        #Uncompute
        for k, b in enumerate(pbits):
            if b == 0:
                op_x(qc, data[k], cost)
        cost.add('enumerate_pattern_equality_checks', L)

#build circuits
def build_qram_enumerate_m_circuit(text: str,
                                   patterns: List[str],
                                   L: int,
                                   outer_iters: int,
                                   qram_mode: str = 'opaque',   # 'opaque' (THEORY) or 'demo' (SIM)
                                   use_ancillas: bool = True) -> (QuantumCircuit, Dict[str, int]):
    """Build a quantum circuit that uses QRAM to find occurrences of any of the given patterns in the text."""
    assert all(len(p) == L for p in patterns), "All patterns must have equal length L."
    n = len(text)
    S = n - L + 1
    assert S > 0, "L must be ≤ len(text)."

    cost = CostTracker()

    n_idx = max(1, ceil(log2(S)))
    idx = QuantumRegister(n_idx, 'idx')
    data = QuantumRegister(2 * L, 'data')
    match = QuantumRegister(1, 'match')
    anc_size = max(1, max(n_idx, 2 * L) - 2) if use_ancillas else 0
    anc = QuantumRegister(anc_size, 'anc') if anc_size > 0 else None
    creg = ClassicalRegister(n_idx, 'c')

    regs = [idx, data, match] + ([anc] if anc is not None else []) + [creg]
    qc = QuantumCircuit(*regs, name=f"EnumerateM_{qram_mode}")

    # Init address superposition and |-> match
    prepare_address_superposition(qc, idx, S, cost)
    op_x(qc, match[0], cost)
    op_h(qc, match[0], cost)

    for _ in range(outer_iters):
        # Compute: load substring into data
        if qram_mode == 'opaque':
            append_qram_text_opaque(qc, idx, data, L, cost)
        elif qram_mode == 'demo':
            qram_text_query_demo(qc, text, L, idx, data, anc, cost)
        else:
            raise ValueError("qram_mode must be 'opaque' or 'demo'")

        # Oracle (enumerate patterns)
        oracle_enumerate_patterns(qc, data, match, patterns, L, cost, anc)

        # Uncompute: remove the text data
        if qram_mode == 'opaque':
            append_qram_text_unquery_opaque(qc, idx, data, L, cost)
        else:
            qram_text_query_demo(qc, text, L, idx, data, anc, cost)  # self-inverse

        # Diffuser on idx (explicit)
        diffuser_on_register(qc, idx, cost, anc)

    qc.measure(idx, creg)
    return qc, cost.snapshot()


def build_qram_inner_grover_circuit(text: str,
                                    patterns: List[str],
                                    L: int,
                                    outer_iters: int,
                                    inner_iters: Optional[int] = None,
                                    qram_mode: str = 'opaque',  # 'opaque' (THEORY) or 'demo' (SIM)
                                    use_ancillas: bool = True) -> (QuantumCircuit, Dict[str, int]):
    """Build a quantum circuit that uses QRAM and Grover's algorithm to find occurrences of any of the given patterns in the text."""
    m = len(patterns)
    assert m >= 1, "Provide at least one pattern."
    assert all(len(p) == L for p in patterns), "All patterns must have equal length L."

    n = len(text)
    S = n - L + 1
    assert S > 0, "L must be ≤ len(text)."

    cost = CostTracker()

    n_idx = max(1, ceil(log2(S)))
    n_pid = max(1, ceil(log2(max(1, m))))

    #Registers
    idx = QuantumRegister(n_idx, 'idx')
    data = QuantumRegister(2 * L, 'data')
    pid = QuantumRegister(n_pid, 'pid')
    pdat = QuantumRegister(2 * L, 'pdat')
    match = QuantumRegister(1, 'match')

    #Ancillas:need enough for diffusers and equality anc_eq (2L)
    anc_size = 2 * L + max(0, max(n_idx, n_pid) - 2) if use_ancillas else 0
    anc = QuantumRegister(anc_size, 'anc') if anc_size > 0 else None
    creg = ClassicalRegister(n_idx, 'c')

    regs = [idx, data, pid, pdat, match] + ([anc] if anc is not None else []) + [creg]
    qc = QuantumCircuit(*regs, name=f"InnerGrover_{qram_mode}")

    #address superposition and uniform over pid
    prepare_address_superposition(qc, idx, S, cost)
    for q in pid:
        op_h(qc, q, cost)

    #|match> = |->
    op_x(qc, match[0], cost)
    op_h(qc, match[0], cost)

    #determine inner iters if not given: ~ ceil((π/4) √m)
    if inner_iters is None:
        inner_iters = max(1, int(np.ceil((pi / 4.0) * np.sqrt(max(1, m)))))
    cost.add('inner_iters', inner_iters)

    #Slice anc into 2L comparator ancillas + rest for MCX recursion
    def anc_view(start, end):
        return [anc[i] for i in range(start, end)] if anc is not None and end <= len(anc) else None

    anc_eq = anc_view(0, 2 * L)
    anc_rest = anc_view(2 * L, (2 * L) + max(0, len(anc) - 2 * L))  # may be None

    for _ in range(outer_iters):
        # Compute: load substring
        if qram_mode == 'opaque':
            append_qram_text_opaque(qc, idx, data, L, cost)
        else:
            qram_text_query_demo(qc, text, L, idx, data, anc, cost)

        #Inner Grover over pid
        for _ in range(inner_iters):
            # Load pattern by pid
            if qram_mode == 'opaque':
                append_qram_pattern_opaque(qc, pid, pdat, L, cost)
            else:
                qram_pattern_query_demo(qc, patterns, L, pid, pdat, anc, cost)

            #Inner oracle: flip 'match' iff data == pdat (explicit comparator)
            equality_flip_on_match(qc, data, pdat, match[0], cost, anc_eq=anc_eq, anc=anc_rest)
            cost.inc('inner_oracle_equality_calls')

            #unquery pattern to make the inner oracle phase-only
            if qram_mode == 'opaque':
                append_qram_pattern_opaque(qc, pid, pdat, L, cost)
            else:
                qram_pattern_query_demo(qc, patterns, L, pid, pdat, anc, cost)

            #Inner diffuser on pid (explicit)
            diffuser_on_register(qc, pid, cost, anc=anc_rest)

        #uncompute: unload substring
        if qram_mode == 'opaque':
            append_qram_text_unquery_opaque(qc, idx, data, L, cost)
        else:
            qram_text_query_demo(qc, text, L, idx, data, anc, cost)

        # Outer diffuser on idx
        diffuser_on_register(qc, idx, cost, anc=anc_rest)

    qc.measure(idx, creg)
    return qc, cost.snapshot()

#demo Execution
def simulate_demo_circuit(qc: QuantumCircuit, shots: int = 2048) -> Dict[str, int]:
    """Simulate the given quantum circuit using Qiskit AerSimulator and return the measurement counts."""
    sim = AerSimulator(method="automatic")
    tqc = transpile(qc, sim, optimization_level=0)  # <= changed from 2 to 0
    result = sim.run(tqc, shots=shots).result()
    return result.get_counts()

if __name__ == "__main__":
    # Small toy instance (DEMO)
    text = "ACGTACGT"
    L = 3
    patterns = ["ACG", "CGT", "GTA", "TAC"]  # m = 4
    outer_iters = 1  # for toy sizes
    shots = 1024

    print("\nideal QRAM: Enumerate-m")
    qc_theory_enum, cost_theory_enum = build_qram_enumerate_m_circuit(
        text, patterns, L, outer_iters=2, qram_mode='opaque')
    print(qc_theory_enum)
    print("Cost counters:", cost_theory_enum)
    print("Note: opaque QRAM => not directly simulatable; per-iteration oracle cost ~ Θ(m·L).")

    print("\nideal QRAM:Inner Grover")
    qc_theory_inner, cost_theory_inner = build_qram_inner_grover_circuit(
        text, patterns, L, outer_iters=2, inner_iters=None, qram_mode='opaque')
    print(qc_theory_inner)
    print("Cost counters:", cost_theory_inner)
    print("blackbox QRAM -> not directly simulatable. oracle cost (per-iter) ~ Θ(L·√m).")

    print("\nDEMO QRAM-like loaders:Enumerate-m (simulatable, O(n·L) loader)")
    qc_demo_enum, cost_demo_enum = build_qram_enumerate_m_circuit(
        text, patterns, L, outer_iters=1, qram_mode='demo')
    print(qc_demo_enum)
    print("Cost counters:", cost_demo_enum)
    counts_enum = simulate_demo_circuit(qc_demo_enum, shots=shots)
    print("Measurement counts (idx):", counts_enum)

    print("\nDEMO QRAM-like loaders: Inner Grover (simulatable, O(n·L)+O(m·L) loaders)")
    qc_demo_inner, cost_demo_inner = build_qram_inner_grover_circuit(
        text, patterns, L, outer_iters=1, inner_iters=None, qram_mode='demo')
    print(qc_demo_inner)
    print("Cost counters:", cost_demo_inner)
    counts_inner = simulate_demo_circuit(qc_demo_inner, shots=shots)
    print("Measurement counts (idx):", counts_inner)
