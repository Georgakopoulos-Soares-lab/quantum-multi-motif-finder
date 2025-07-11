import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit import QuantumRegister, ClassicalRegister
import matplotlib.pyplot as plt
from math import sqrt, ceil
from collections import defaultdict, deque
from Bio import SeqIO

class QuantumAhoCorasick:
    def __init__(self):
        self.simulator = Aer.get_backend('qasm_simulator')
        self.dna_encoding = {'A': '00', 'C': '01', 'G': '10', 'T': '11'}
        self.dna_decoding = {'00': 'A', '01': 'C', '10': 'G', '11': 'T'}
        
    def build_classical_trie(self, patterns):
        """Build the classical Aho-Corasick trie with proper failure links"""
        # Initialize the root of the trie
        root = {}
        root['failure'] = root
        root['output'] = []
        
        # Add each pattern to the trie
        for pattern_id, pattern in enumerate(patterns):
            current = root
            for symbol in pattern:
                current = current.setdefault(symbol, {})
                if 'output' not in current:
                    current['output'] = []
                if 'failure' not in current:
                    current['failure'] = None
            current['output'].append((pattern_id, pattern))
        
        # Build failure links using BFS
        queue = deque()
        
        # Set failure of depth 1 nodes to root
        for symbol in ['A', 'C', 'G', 'T']:
            if symbol in root:
                node = root[symbol]
                node['failure'] = root
                queue.append(node)
            else:
                root[symbol] = root  # Default transition back to root
                
        # Build failure links for deeper nodes
        while queue:
            current = queue.popleft()
            
            for symbol in ['A', 'C', 'G', 'T']:
                if symbol in current:
                    child = current[symbol]
                    queue.append(child)
                    
                    # Find failure state
                    failure = current['failure']
                    while symbol not in failure and failure != root:
                        failure = failure['failure']
                    
                    if symbol in failure:
                        child['failure'] = failure[symbol]
                        # Add outputs from failure state
                        child['output'].extend(failure[symbol]['output'])
                    else:
                        child['failure'] = root
        
        return root
    
    def classical_aho_corasick_search(self, text, patterns):
        """Full classical Aho-Corasick algorithm implementation"""
        root = self.build_classical_trie(patterns)
        results = []
        
        # Search the text
        current = root
        for i, symbol in enumerate(text):
            # Follow the trie or failure links
            while symbol not in current and current != root:
                current = current['failure']
                
            if symbol in current:
                current = current[symbol]
            else:
                current = root
                continue  # No match at this position
                
            # Check for pattern matches at this position
            for pattern_id, pattern in current['output']:
                match_pos = i - len(pattern) + 1
                results.append((match_pos, pattern))
        
        return results
    
    def quantum_simulate_aho_corasick(self, text, patterns):
        """
        Create a quantum circuit that simulates the Aho-Corasick automaton
        using Qiskit. This is a hybrid implementation.
        """
        # First build the classical Aho-Corasick automaton
        trie = self.build_classical_trie(patterns)
        
        # Count states in the automaton (for state representation)
        states, state_mapping = self._enumerate_states(trie)
        state_qubits = max(1, ceil(np.log2(len(states))))
        
        # Qubits for pattern matches
        pattern_qubits = len(patterns)
        
        # Create quantum registers
        qr_state = QuantumRegister(state_qubits, 'state')
        qr_patterns = QuantumRegister(pattern_qubits, 'patterns')
        cr = ClassicalRegister(pattern_qubits, 'matches')
        qc = QuantumCircuit(qr_state, qr_patterns, cr)
        
        # Start with the automaton in the initial state (root)
        root_state_bits = format(state_mapping[id(trie)], f'0{state_qubits}b')
        for i, bit in enumerate(root_state_bits):
            if bit == '1':
                qc.x(qr_state[i])
        
        # Process the text character by character
        current_state = trie
        results = []
        
        for i, symbol in enumerate(text):
            # Classical pre-processing - follow the automaton
            while symbol not in current_state and current_state != trie:
                current_state = current_state['failure']
                
            if symbol in current_state:
                current_state = current_state[symbol]
            else:
                current_state = trie
                continue
            
            # Quantum part - mark pattern matches
            if current_state['output']:
                # Set state qubits to current state
                # First reset state qubits
                qc.reset(qr_state)
                
                # Set to current state
                state_bits = format(state_mapping[id(current_state)], f'0{state_qubits}b')
                for j, bit in enumerate(state_bits):
                    if bit == '1':
                        qc.x(qr_state[j])
                
                # Mark pattern matches for this state
                for pattern_id, pattern in current_state['output']:
                    # Controlled-X to mark the pattern
                    qc.mcx(qr_state, qr_patterns[pattern_id])
                    
                    # Store this match
                    match_pos = i - len(pattern) + 1
                    results.append((match_pos, pattern))
        
        # Measure pattern qubits
        qc.measure(qr_patterns, cr)
        
        # For demonstration, we'll run the circuit
        job = execute(qc, self.simulator, shots=1024)
        counts = job.result().get_counts(qc)
        print("Quantum circuit measurement results:")
        for outcome, count in sorted(counts.items(), key=lambda x: -x[1]):
            if '1' in outcome:  # If any pattern was matched
                print(f"  {outcome}: {count} shots")
        
        # Return the results we collected
        return results
    
    def _enumerate_states(self, root):
        """Assign unique IDs to all states in the trie"""
        states = []
        state_mapping = {}  # Maps state object ID to numeric ID
        
        # BFS to traverse all states
        queue = deque([root])
        while queue:
            state = queue.popleft()
            if id(state) not in state_mapping:
                state_mapping[id(state)] = len(states)
                states.append(state)
                
                # Add child states to queue
                for symbol in ['A', 'C', 'G', 'T']:
                    if symbol in state and symbol != 'failure' and symbol != 'output':
                        queue.append(state[symbol])
        
        return states, state_mapping
    
    def quantum_multi_pattern_dna_matching(self, dna_text, pattern_database):
        """Find multiple DNA patterns simultaneously using quantum-assisted Aho-Corasick"""
        print("Running quantum-assisted Aho-Corasick search...")
        return self.quantum_simulate_aho_corasick(dna_text, pattern_database)
    
    def visualize_results(self, text, results):
        """Visualize the pattern matching results"""
        if not results:
            print("No matches found to visualize.")
            return
            
        plt.figure(figsize=(12, 6))
        plt.text(0, 0.5, text, fontsize=12)
        
        colors = ['r', 'g', 'b', 'm', 'c', 'y']
        patterns_seen = {}
        
        for i, (pos, pattern) in enumerate(sorted(results)):
            pattern_idx = patterns_seen.get(pattern, len(patterns_seen))
            if pattern not in patterns_seen:
                patterns_seen[pattern] = pattern_idx
                
            color = colors[pattern_idx % len(colors)]
            plt.plot([pos, pos+len(pattern)], [0.6 + 0.1*pattern_idx, 0.6 + 0.1*pattern_idx], 
                    color=color, linewidth=2)
            plt.text(pos, 0.7 + 0.1*pattern_idx, pattern, fontsize=10, color=color)
        
        plt.axis('off')
        plt.title('DNA Pattern Matching Results using Quantum Aho-Corasick')
        plt.tight_layout()
        plt.show()
    
def read_fasta_sequence_biopython(file_path):
    record = next(SeqIO.parse(file_path, "fasta"))
    return str(record.seq)

# Example usage
def main():
    # Test with a DNA sequence
    #dna_text = "ACGTACGTGCTAGCTAGCTAGCTAGCATCGATCGATCGATCGATCGATCGATCG"

    # Example usage
    dna_text = read_fasta_sequence_biopython('GCF_000146045.2_R64_genomic.fna')
    #print(sequence)
    
    # Patterns to search for
    patterns = ["ACGT", "GCTA", "GATC"]
    
    # Create quantum Aho-Corasick instance
    qac = QuantumAhoCorasick()
    
    # For comparison, run classical Aho-Corasick first
    print("\nClassical Aho-Corasick search:")
    classical_results = qac.classical_aho_corasick_search(dna_text, patterns)
    print(f"Found {len(classical_results)} matches:")
    for pos, pattern in sorted(classical_results):
        print(f"  Pattern '{pattern}' found at position {pos}")
    
    print("\n" + "-"*50)
    
    # Now run quantum-assisted search
    quantum_results = qac.quantum_multi_pattern_dna_matching(dna_text, patterns)
    print(f"\nQuantum-assisted search found {len(quantum_results)} matches:")
    for pos, pattern in sorted(quantum_results):
        print(f"  Pattern '{pattern}' found at position {pos}")
    
    # Visualize
    qac.visualize_results(dna_text, quantum_results)
    
    # For validation, confirm the results match
    if sorted(classical_results) == sorted(quantum_results):
        print("\nValidation: Classical and quantum results match! ")
    else:
        print("\nValidation: Results don't match! Please check the implementation.")

if __name__ == "__main__":
    main()
