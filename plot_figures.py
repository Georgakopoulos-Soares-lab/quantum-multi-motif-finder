from math import ceil, log2, pi
import numpy as np
from typing import List, Optional, Dict, Any
from collections import defaultdict
# Qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import Instruction
from qiskit.circuit.library import StatePreparation    
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from qiskit.visualization import circuit_drawer

#helpers/utilities
DNA2BITS = {'A': (0, 0), 'C': (0, 1), 'G': (1, 0), 'T': (1, 1)}

def pattern_bits_2L(pattern: str) -> List[int]:
   
    bits = []
    for ch in pattern:
        b0, b1 = DNA2BITS[ch]
        bits.extend([b0, b1])
    return bits

def opaque_instruction(name: str, num_qubits: int, num_clbits: int = 0, params=None) -> Instruction:
   
    return Instruction(name=name, num_qubits=num_qubits, num_clbits=num_clbits, params=params or [])

#cost tracking
class CostTracker:
   
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
   
    if anc is not None and len(ctrls) > 4:
        qc.mcx(list(ctrls), tgt, ancilla_qubits=list(anc), mode='recursion')
    else:
        qc.mcx(list(ctrls), tgt)
    cost.inc('mcx')

#state preparation on address (unitary)
def prepare_address_superposition(qc: QuantumCircuit, addr: QuantumRegister, S: int, cost: CostTracker):
   
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

#equality test / data == pdat (explicit, non-opaque)
def equality_flip_on_match(qc: QuantumCircuit,
                           data: QuantumRegister,
                           pdat: QuantumRegister,
                           target,
                           cost: CostTracker,
                           anc_eq: Optional[List] = None,
                           anc: Optional[QuantumRegister] = None):
   
    L2 = len(data)
    if anc_eq is None or len(anc_eq) < L2:
        raise ValueError("Need anc_eq of length >= len(data) (2L) for comparator.")

    #build XNOR bits into anc_eq
    for k in range(L2):
        qc.cx(data[k], anc_eq[k]); cost.inc('cx')
        qc.cx(pdat[k], anc_eq[k]); cost.inc('cx')
        op_x(qc, anc_eq[k], cost)

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
   
    S = len(text) - L + 1
    for k in range(L):
        for i in range(S):
            b0, b1 = DNA2BITS[text[i + k]]
            bits = format(i, f'0{len(idx)}b')[::-1]
            for qb, b in zip(idx, bits):
                if b == '0':
                    op_x(qc, qb, cost)
            if b0 == 1:
                op_mcx(qc, list(idx), data[2*k], cost, anc)
            if b1 == 1:
                op_mcx(qc, list(idx), data[2*k+1], cost, anc)
            for qb, b in zip(idx, bits):
                if b == '0':
                    op_x(qc, qb, cost)
    cost.add('demo_qram_text_ops', S * L)

def qram_pattern_query_demo(qc: QuantumCircuit, patterns: List[str], L: int,
                            pid: QuantumRegister, pdat: QuantumRegister,
                            anc: Optional[QuantumRegister], cost: CostTracker):
   
    m = len(patterns)
    for j, pat in enumerate(patterns):
        bits = format(j, f'0{len(pid)}b')[::-1]
        for qb, b in zip(pid, bits):
            if b == '0':
                op_x(qc, qb, cost)
        pbits = pattern_bits_2L(pat)
        for k, b in enumerate(pbits):
            if b == 1:
                op_mcx(qc, list(pid), pdat[k], cost, anc)
        for qb, b in zip(pid, bits):
            if b == '0':
                op_x(qc, qb, cost)
    cost.add('demo_qram_pattern_ops', m * L)

#theory: opaque QRAM gates
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
   
    for pat in patterns:
        pbits = pattern_bits_2L(pat)
        for k, b in enumerate(pbits):
            if b == 0:
                op_x(qc, data[k], cost)
        op_mcx(qc, list(data), match[0], cost, anc)
        for k, b in enumerate(pbits):
            if b == 0:
                op_x(qc, data[k], cost)
        cost.add('enumerate_pattern_equality_checks', L)

#build circuits
def build_qram_enumerate_m_circuit(text: str,
                                   patterns: List[str],
                                   L: int,
                                   outer_iters: int,
                                   qram_mode: str = 'opaque',
                                   use_ancillas: bool = True) -> (QuantumCircuit, Dict[str, int]):
   
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

    prepare_address_superposition(qc, idx, S, cost)
    op_x(qc, match[0], cost)
    op_h(qc, match[0], cost)

    for _ in range(outer_iters):
        if qram_mode == 'opaque':
            append_qram_text_opaque(qc, idx, data, L, cost)
        elif qram_mode == 'demo':
            qram_text_query_demo(qc, text, L, idx, data, anc, cost)
        else:
            raise ValueError("qram_mode must be 'opaque' or 'demo'")

        oracle_enumerate_patterns(qc, data, match, patterns, L, cost, anc)

        if qram_mode == 'opaque':
            append_qram_text_unquery_opaque(qc, idx, data, L, cost)
        else:
            qram_text_query_demo(qc, text, L, idx, data, anc, cost)

        diffuser_on_register(qc, idx, cost, anc)

    qc.measure(idx, creg)
    return qc, cost.snapshot()


def build_qram_inner_grover_circuit(text: str,
                                    patterns: List[str],
                                    L: int,
                                    outer_iters: int,
                                    inner_iters: Optional[int] = None,
                                    qram_mode: str = 'opaque',
                                    use_ancillas: bool = True) -> (QuantumCircuit, Dict[str, int]):
   
    m = len(patterns)
    assert m >= 1, "Provide at least one pattern."
    assert all(len(p) == L for p in patterns), "All patterns must have equal length L."

    n = len(text)
    S = n - L + 1
    assert S > 0, "L must be ≤ len(text)."

    cost = CostTracker()

    n_idx = max(1, ceil(log2(S)))
    n_pid = max(1, ceil(log2(max(1, m))))

    idx = QuantumRegister(n_idx, 'idx')
    data = QuantumRegister(2 * L, 'data')
    pid = QuantumRegister(n_pid, 'pid')
    pdat = QuantumRegister(2 * L, 'pdat')
    match = QuantumRegister(1, 'match')

    anc_size = 2 * L + max(0, max(n_idx, n_pid) - 2) if use_ancillas else 0
    anc = QuantumRegister(anc_size, 'anc') if anc_size > 0 else None
    creg = ClassicalRegister(n_idx, 'c')

    regs = [idx, data, pid, pdat, match] + ([anc] if anc is not None else []) + [creg]
    qc = QuantumCircuit(*regs, name=f"InnerGrover_{qram_mode}")

    prepare_address_superposition(qc, idx, S, cost)
    for q in pid:
        op_h(qc, q, cost)

    op_x(qc, match[0], cost)
    op_h(qc, match[0], cost)

    if inner_iters is None:
        inner_iters = max(1, int(np.ceil((pi / 4.0) * np.sqrt(max(1, m)))))
    cost.add('inner_iters', inner_iters)

    def anc_view(start, end):
        return [anc[i] for i in range(start, end)] if anc is not None and end <= len(anc) else None

    anc_eq = anc_view(0, 2 * L)
    anc_rest = anc_view(2 * L, (2 * L) + max(0, len(anc) - 2 * L))

    for _ in range(outer_iters):
        if qram_mode == 'opaque':
            append_qram_text_opaque(qc, idx, data, L, cost)
        else:
            qram_text_query_demo(qc, text, L, idx, data, anc, cost)

        for _ in range(inner_iters):
            if qram_mode == 'opaque':
                append_qram_pattern_opaque(qc, pid, pdat, L, cost)
            else:
                qram_pattern_query_demo(qc, patterns, L, pid, pdat, anc, cost)

            equality_flip_on_match(qc, data, pdat, match[0], cost, anc_eq=anc_eq, anc=anc_rest)
            cost.inc('inner_oracle_equality_calls')

            if qram_mode == 'opaque':
                append_qram_pattern_opaque(qc, pid, pdat, L, cost)
            else:
                qram_pattern_query_demo(qc, patterns, L, pid, pdat, anc, cost)

            diffuser_on_register(qc, pid, cost, anc=anc_rest)

        if qram_mode == 'opaque':
            append_qram_text_unquery_opaque(qc, idx, data, L, cost)
        else:
            qram_text_query_demo(qc, text, L, idx, data, anc, cost)

        diffuser_on_register(qc, idx, cost, anc=anc_rest)

    qc.measure(idx, creg)
    return qc, cost.snapshot()


def save_circuit_images(text: str, patterns: List[str], L: int, outer_iters: int = 1,
                        output_dir: str = "./circuit_images"):
   
        #DNA text string
        #patterns: List of DNA patterns to search
        #L: Pattern length
        #outer_iters: Number of outer Grover iterations
        #output_dir: Directory to save images
   
    import os
    os.makedirs(output_dir, exist_ok=True)
   
    #print("=" * 80)
    #print("GENERATING QUANTUM CIRCUIT IMAGES")
    #print("=" * 80)
    print(f"\nInput Parameters:")
    print(f"  Text: {text}")
    print(f"  Patterns: {patterns}")
    print(f"  Pattern Length L: {L}")
    print(f"  Outer Iterations: {outer_iters}")
    print(f"  Number of patterns (m): {len(patterns)}")
    print(f"  Text positions (S): {len(text) - L + 1}")
    print(f"\nOutput Directory: {output_dir}")
    print()
   
    #algorithm 1: Enumerate-m with opaque QRAM
    print("\n" + "=" * 80)
    print("ALGORITHM 1: ENUMERATE-m ORACLE")
    print("=" * 80)
    qc1_opaque, costs1_opaque = build_qram_enumerate_m_circuit(
        text, patterns, L, outer_iters, qram_mode='opaque', use_ancillas=True
    )
   
    print(f"\nCircuit Statistics:")
    print(f"  Total Qubits: {qc1_opaque.num_qubits}")
    print(f"  Circuit Depth: {qc1_opaque.depth()}")
    print(f"  Gate Counts: {dict(costs1_opaque)}")
   
    # opaque version
    filename1_opaque = f"{output_dir}/enumerate_m_opaque.png"
    fig1 = qc1_opaque.draw(output='mpl', style='iqp', fold=-1)
    plt.savefig(filename1_opaque, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {filename1_opaque}")
   
    #Algorithm 1: Demo version (only for small circuits)
    if len(text) <= 4 and L <= 2 and len(patterns) <= 3:
        qc1_demo, costs1_demo = build_qram_enumerate_m_circuit(
            text, patterns, L, outer_iters, qram_mode='demo', use_ancillas=True
        )
        filename1_demo = f"{output_dir}/enumerate_m_demo.png"
        fig1d = qc1_demo.draw(output='mpl', style='iqp', fold=140)
        plt.savefig(filename1_demo, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename1_demo}")
   
    # Algorithm 2: Inner Grover with opaque QRAM
    #print("\n" + "=" * 80)
    #print("ALGORITHM 2: INNER GROVER SEARCH")
    #print("=" * 80)
    qc2_opaque, costs2_opaque = build_qram_inner_grover_circuit(
        text, patterns, L, outer_iters, inner_iters=None, qram_mode='opaque', use_ancillas=True
    )
   
    print(f"\nCircuit Statistics:")
    print(f"  Total Qubits: {qc2_opaque.num_qubits}")
    print(f"  Circuit Depth: {qc2_opaque.depth()}")
    print(f"  Gate Counts: {dict(costs2_opaque)}")
    print(f"  Inner Iterations: {costs2_opaque.get('inner_iters', 'N/A')}")
   
    # Save opaque version
    filename2_opaque = f"{output_dir}/inner_grover_opaque.png"
    fig2 = qc2_opaque.draw(output='mpl', style='iqp', fold=-1)
    plt.savefig(filename2_opaque, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {filename2_opaque}")
   
    # Algorithm 2: Demo version (only for very small circuits)
    if len(text) <= 4 and L <= 2 and len(patterns) <= 2:
        qc2_demo, costs2_demo = build_qram_inner_grover_circuit(
            text, patterns, L, outer_iters, inner_iters=1, qram_mode='demo', use_ancillas=True
        )
        filename2_demo = f"{output_dir}/inner_grover_demo.png"
        fig2d = qc2_demo.draw(output='mpl', style='iqp', fold=140)
        plt.savefig(filename2_demo, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename2_demo}")
   
    # Create comparison figure
    #print("\n" + "=" * 80)
    print("COMPARISON CHART")
    #print("=" * 80)
   
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Quantum DNA Pattern Matching - Circuit Comparison', fontsize=16, fontweight='bold')
   
    # Statistics comparison
    stats_labels = ['Qubits', 'Depth', 'H Gates', 'X Gates', 'MCX Gates']
    enum_stats = [
        qc1_opaque.num_qubits,
        qc1_opaque.depth(),
        costs1_opaque.get('h', 0),
        costs1_opaque.get('x', 0),
        costs1_opaque.get('mcx', 0)
    ]
    grover_stats = [
        qc2_opaque.num_qubits,
        qc2_opaque.depth(),
        costs2_opaque.get('h', 0),
        costs2_opaque.get('x', 0),
        costs2_opaque.get('mcx', 0)
    ]
   
    x = np.arange(len(stats_labels))
    width = 0.35
   
    axes[0, 0].bar(x - width/2, enum_stats, width, label='Enumerate-m', alpha=0.8, color='#4F46E5')
    axes[0, 0].bar(x + width/2, grover_stats, width, label='Inner Grover', alpha=0.8, color='#10B981')
    axes[0, 0].set_xlabel('Metric')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Circuit Statistics Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(stats_labels, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
   
    # Gate distribution for Enumerate-m
    enum_gates = {k: v for k, v in costs1_opaque.items() if k in ['h', 'x', 'cx', 'mcx', 'z']}
    axes[0, 1].pie(enum_gates.values(), labels=enum_gates.keys(), autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Algorithm 1: Gate Distribution')
   
    # Gate distribution for Inner Grover
    grover_gates = {k: v for k, v in costs2_opaque.items() if k in ['h', 'x', 'cx', 'mcx', 'z']}
    axes[1, 0].pie(grover_gates.values(), labels=grover_gates.keys(), autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Algorithm 2: Gate Distribution')
   
    # Key differences table
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
   
    comparison_data = [
        ['Metric', 'Enumerate-m', 'Inner Grover'],
        ['Total Qubits', str(qc1_opaque.num_qubits), str(qc2_opaque.num_qubits)],
        ['Circuit Depth', str(qc1_opaque.depth()), str(qc2_opaque.depth())],
        ['QRAM Queries', str(costs1_opaque.get('qram_text_query', 0)),
         str(costs2_opaque.get('qram_text_query', 0) + costs2_opaque.get('qram_pattern_query', 0))],
        ['Oracle Type', 'Sequential Check', 'Grover Search'],
        ['Patterns (m)', str(len(patterns)), str(len(patterns))],
        ['Inner Iterations', 'N/A', str(costs2_opaque.get('inner_iters', 'N/A'))]
    ]
   
    table = axes[1, 1].table(cellText=comparison_data, cellLoc='center', loc='center',
                             colWidths=[0.35, 0.325, 0.325])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
   
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4F46E5')
        table[(0, i)].set_text_props(weight='bold', color='white')
   
    axes[1, 1].set_title('Key Differences', fontweight='bold', pad=20)
   
    plt.tight_layout()
    comparison_filename = f"{output_dir}/comparison_chart.png"
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {comparison_filename}")
   
    #print("\n" + "=" * 80)
    #print("IMAGE GENERATION COMPLETE")
    #print("=" * 80)
    #print(f"\nAll images saved to: {output_dir}/")
    #print("\nGenerated files:")
    #print(f"  1. enumerate_m_opaque.png - Algorithm 1 circuit diagram")
    #print(f"  2. inner_grover_opaque.png - Algorithm 2 circuit diagram")
    #if len(text) <= 4 and L <= 2:
    #    print(f"  3. enumerate_m_demo.png - Algorithm 1 with explicit gates")
    #    if len(patterns) <= 2:
    #        print(f"  4. inner_grover_demo.png - Algorithm 2 with explicit gates")
    #print(f"  N. comparison_chart.png - Statistical comparison")
   
    return qc1_opaque, qc2_opaque


if __name__ == "__main__":
    # Example 1: Small instance
    print("\n1: Small Instance")
    text1 = "ACGT"
    L1 = 2
    patterns1 = ["AC", "CG", "GT"]
   
    qc1a, qc1b = save_circuit_images(text1, patterns1, L1, outer_iters=1,
                                      output_dir="./circuit_images_small")
   
    # Example 2: Medium instance
    print("\n\n2: Medium Instance")
    text2 = "ACGTACGT"
    L2 = 3
    patterns2 = ["ACG", "CGT", "GTA", "TAC"]
   
    qc2a, qc2b = save_circuit_images(text2, patterns2, L2, outer_iters=1,
                                      output_dir="./circuit_images_medium")
   
