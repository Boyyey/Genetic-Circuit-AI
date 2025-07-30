"""
Bioinformatics utilities for genetic circuit design.

This module provides bioinformatics tools for sequence analysis,
primer design, restriction site analysis, and secondary structure prediction.

Author: Genetic Circuit Design Team
Version: 1.0.0
License: MIT
"""

import logging
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from Bio import SeqIO, Align
from Bio.Seq import Seq
from Bio.SeqUtils import GC
from Bio.Restriction import Analysis, CommOnly
from Bio.SeqUtils.ProtParam import ProteinAnalysis

logger = logging.getLogger(__name__)


def sequence_analysis(sequence: str) -> Dict[str, Any]:
    """
    Perform comprehensive sequence analysis.
    
    Args:
        sequence: DNA/RNA sequence to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    results = {
        "length": len(sequence),
        "gc_content": calculate_gc_content(sequence),
        "restriction_sites": find_restriction_sites(sequence),
        "secondary_structure": analyze_secondary_structure(sequence),
        "composition": analyze_composition(sequence),
        "repeats": find_repeats(sequence),
        "motifs": find_motifs(sequence)
    }
    
    return results


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate GC content percentage.
    
    Args:
        sequence: DNA/RNA sequence
        
    Returns:
        GC content as percentage
    """
    if not sequence:
        return 0.0
    
    gc_count = sequence.upper().count('G') + sequence.upper().count('C')
    return (gc_count / len(sequence)) * 100


def find_restriction_sites(sequence: str) -> Dict[str, List[int]]:
    """
    Find restriction enzyme recognition sites.
    
    Args:
        sequence: DNA sequence to analyze
        
    Returns:
        Dictionary mapping enzyme names to positions
    """
    try:
        # Create BioPython sequence object
        seq = Seq(sequence.upper())
        
        # Analyze with common restriction enzymes
        rb = Analysis(CommOnly, seq)
        
        # Get restriction sites
        sites = {}
        for enzyme in rb.format_output():
            if enzyme:
                enzyme_name = enzyme.split()[0]
                positions = [int(pos) for pos in enzyme.split()[1:]]
                if positions:
                    sites[enzyme_name] = positions
        
        return sites
        
    except Exception as e:
        logger.warning(f"Error finding restriction sites: {e}")
        return {}


def design_primers(sequence: str, target_length: int = 20, 
                  min_tm: float = 55.0, max_tm: float = 65.0) -> List[Dict[str, Any]]:
    """
    Design PCR primers for a sequence.
    
    Args:
        sequence: Target DNA sequence
        target_length: Target primer length
        min_tm: Minimum melting temperature
        max_tm: Maximum melting temperature
        
    Returns:
        List of primer designs
    """
    primers = []
    
    # Design forward primers
    for i in range(len(sequence) - target_length + 1):
        primer = sequence[i:i + target_length]
        tm = calculate_melting_temperature(primer)
        
        if min_tm <= tm <= max_tm:
            primers.append({
                "sequence": primer,
                "position": i,
                "type": "forward",
                "melting_temperature": tm,
                "gc_content": calculate_gc_content(primer)
            })
    
    # Design reverse primers (complement)
    rev_comp = str(Seq(sequence).reverse_complement())
    for i in range(len(rev_comp) - target_length + 1):
        primer = rev_comp[i:i + target_length]
        tm = calculate_melting_temperature(primer)
        
        if min_tm <= tm <= max_tm:
            primers.append({
                "sequence": primer,
                "position": len(sequence) - i - target_length,
                "type": "reverse",
                "melting_temperature": tm,
                "gc_content": calculate_gc_content(primer)
            })
    
    return primers


def calculate_melting_temperature(sequence: str) -> float:
    """
    Calculate melting temperature using Wallace rule.
    
    Args:
        sequence: DNA sequence
        
    Returns:
        Melting temperature in Celsius
    """
    if not sequence:
        return 0.0
    
    # Wallace rule: Tm = 2°C(A+T) + 4°C(G+C)
    a_count = sequence.upper().count('A')
    t_count = sequence.upper().count('T')
    g_count = sequence.upper().count('G')
    c_count = sequence.upper().count('C')
    
    return 2 * (a_count + t_count) + 4 * (g_count + c_count)


def analyze_secondary_structure(sequence: str) -> Dict[str, Any]:
    """
    Analyze RNA secondary structure (simplified).
    
    Args:
        sequence: RNA sequence
        
    Returns:
        Secondary structure analysis results
    """
    # This is a simplified analysis - in practice, use tools like RNAfold
    results = {
        "hairpins": find_hairpins(sequence),
        "base_pairs": count_base_pairs(sequence),
        "structure_score": calculate_structure_score(sequence)
    }
    
    return results


def find_hairpins(sequence: str) -> List[Dict[str, Any]]:
    """
    Find potential hairpin structures.
    
    Args:
        sequence: RNA sequence
        
    Returns:
        List of hairpin structures
    """
    hairpins = []
    
    # Look for inverted repeats that could form hairpins
    for i in range(len(sequence) - 10):
        for j in range(i + 10, len(sequence) - 5):
            # Check for potential hairpin
            stem_length = min(j - i, len(sequence) - j)
            if stem_length >= 5:
                stem1 = sequence[i:i + stem_length]
                stem2 = sequence[j:j + stem_length]
                rev_comp_stem2 = str(Seq(stem2).reverse_complement())
                
                if stem1 == rev_comp_stem2:
                    hairpins.append({
                        "start": i,
                        "end": j + stem_length,
                        "stem_length": stem_length,
                        "loop_length": j - i - stem_length
                    })
    
    return hairpins


def count_base_pairs(sequence: str) -> int:
    """
    Count potential base pairs in RNA sequence.
    
    Args:
        sequence: RNA sequence
        
    Returns:
        Number of potential base pairs
    """
    # Count complementary base pairs
    pairs = 0
    for i in range(len(sequence) // 2):
        base1 = sequence[i]
        base2 = sequence[-(i + 1)]
        
        if (base1 == 'A' and base2 == 'U') or (base1 == 'U' and base2 == 'A'):
            pairs += 1
        elif (base1 == 'G' and base2 == 'C') or (base1 == 'C' and base2 == 'G'):
            pairs += 1
        elif (base1 == 'G' and base2 == 'U') or (base1 == 'U' and base2 == 'G'):
            pairs += 1
    
    return pairs


def calculate_structure_score(sequence: str) -> float:
    """
    Calculate a simple structure stability score.
    
    Args:
        sequence: RNA sequence
        
    Returns:
        Structure stability score
    """
    if not sequence:
        return 0.0
    
    # Simple scoring based on GC content and length
    gc_content = calculate_gc_content(sequence)
    length_factor = min(len(sequence) / 100.0, 1.0)
    
    return (gc_content / 100.0) * length_factor


def analyze_composition(sequence: str) -> Dict[str, Any]:
    """
    Analyze sequence composition.
    
    Args:
        sequence: DNA/RNA sequence
        
    Returns:
        Composition analysis results
    """
    if not sequence:
        return {}
    
    seq_upper = sequence.upper()
    
    composition = {
        "A": seq_upper.count('A'),
        "T": seq_upper.count('T'),
        "G": seq_upper.count('G'),
        "C": seq_upper.count('C'),
        "U": seq_upper.count('U'),
        "N": seq_upper.count('N'),
        "total": len(sequence)
    }
    
    # Calculate percentages
    for base in ['A', 'T', 'G', 'C', 'U', 'N']:
        composition[f"{base}_percent"] = (composition[base] / composition["total"]) * 100
    
    return composition


def find_repeats(sequence: str, min_length: int = 3) -> List[Dict[str, Any]]:
    """
    Find repeated sequences.
    
    Args:
        sequence: DNA/RNA sequence
        min_length: Minimum repeat length
        
    Returns:
        List of repeat structures
    """
    repeats = []
    
    for length in range(min_length, len(sequence) // 2 + 1):
        for i in range(len(sequence) - length + 1):
            pattern = sequence[i:i + length]
            
            # Count occurrences
            count = sequence.count(pattern)
            if count > 1:
                # Find all positions
                positions = []
                start = 0
                while True:
                    pos = sequence.find(pattern, start)
                    if pos == -1:
                        break
                    positions.append(pos)
                    start = pos + 1
                
                repeats.append({
                    "pattern": pattern,
                    "length": length,
                    "count": count,
                    "positions": positions
                })
    
    return repeats


def find_motifs(sequence: str) -> List[Dict[str, Any]]:
    """
    Find common biological motifs.
    
    Args:
        sequence: DNA/RNA sequence
        
    Returns:
        List of found motifs
    """
    motifs = []
    
    # Common promoter motifs
    promoter_motifs = {
        "-10_box": "TATAAT",
        "-35_box": "TTGACA",
        "TATA_box": "TATA",
        "CAAT_box": "CAAT",
        "GC_box": "GGGCGG"
    }
    
    # Common regulatory motifs
    regulatory_motifs = {
        "CAP_site": "AAATGTGATCTAGCTCAC",
        "operator": "AATTGTGAGCGGATAACAATT",
        "ribosome_binding": "AGGAGG"
    }
    
    # Search for motifs
    for motif_name, motif_seq in {**promoter_motifs, **regulatory_motifs}.items():
        positions = []
        start = 0
        while True:
            pos = sequence.upper().find(motif_seq, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        if positions:
            motifs.append({
                "name": motif_name,
                "sequence": motif_seq,
                "positions": positions,
                "count": len(positions)
            })
    
    return motifs


def translate_sequence(sequence: str, reading_frame: int = 1) -> str:
    """
    Translate DNA sequence to protein.
    
    Args:
        sequence: DNA sequence
        reading_frame: Reading frame (1, 2, or 3)
        
    Returns:
        Translated protein sequence
    """
    try:
        seq = Seq(sequence)
        
        if reading_frame == 1:
            protein = seq.translate()
        elif reading_frame == 2:
            protein = seq[1:].translate()
        elif reading_frame == 3:
            protein = seq[2:].translate()
        else:
            raise ValueError("Reading frame must be 1, 2, or 3")
        
        return str(protein)
        
    except Exception as e:
        logger.warning(f"Error translating sequence: {e}")
        return ""


def analyze_protein(protein_sequence: str) -> Dict[str, Any]:
    """
    Analyze protein sequence properties.
    
    Args:
        protein_sequence: Protein sequence
        
    Returns:
        Protein analysis results
    """
    try:
        protein = ProteinAnalysis(protein_sequence)
        
        analysis = {
            "molecular_weight": protein.molecular_weight(),
            "isoelectric_point": protein.isoelectric_point(),
            "amino_acid_composition": protein.get_amino_acids_percent(),
            "secondary_structure": protein.secondary_structure_fraction(),
            "flexibility": protein.flexibility(),
            "gravy": protein.gravy(),
            "aromaticity": protein.aromaticity()
        }
        
        return analysis
        
    except Exception as e:
        logger.warning(f"Error analyzing protein: {e}")
        return {}


def calculate_codon_usage(sequence: str) -> Dict[str, float]:
    """
    Calculate codon usage statistics.
    
    Args:
        sequence: DNA sequence
        
    Returns:
        Codon usage frequencies
    """
    if len(sequence) % 3 != 0:
        sequence = sequence[:-(len(sequence) % 3)]
    
    codons = {}
    total_codons = len(sequence) // 3
    
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i+3].upper()
        if len(codon) == 3:
            codons[codon] = codons.get(codon, 0) + 1
    
    # Calculate frequencies
    codon_frequencies = {}
    for codon, count in codons.items():
        codon_frequencies[codon] = (count / total_codons) * 100
    
    return codon_frequencies


def optimize_codon_usage(sequence: str, target_organism: str = "E. coli") -> str:
    """
    Optimize codon usage for a target organism.
    
    Args:
        sequence: DNA sequence
        target_organism: Target organism for optimization
        
    Returns:
        Optimized sequence
    """
    # This is a simplified implementation
    # In practice, use codon usage tables for specific organisms
    
    # E. coli preferred codons (simplified)
    ecoli_preferred = {
        "AAA": "Lys", "AAC": "Asn", "AAG": "Lys", "AAT": "Asn",
        "ACA": "Thr", "ACC": "Thr", "ACG": "Thr", "ACT": "Thr",
        "AGA": "Arg", "AGC": "Ser", "AGG": "Arg", "AGT": "Ser",
        "ATA": "Ile", "ATC": "Ile", "ATG": "Met", "ATT": "Ile",
        "CAA": "Gln", "CAC": "His", "CAG": "Gln", "CAT": "His",
        "CCA": "Pro", "CCC": "Pro", "CCG": "Pro", "CCT": "Pro",
        "CGA": "Arg", "CGC": "Arg", "CGG": "Arg", "CGT": "Arg",
        "CTA": "Leu", "CTC": "Leu", "CTG": "Leu", "CTT": "Leu",
        "GAA": "Glu", "GAC": "Asp", "GAG": "Glu", "GAT": "Asp",
        "GCA": "Ala", "GCC": "Ala", "GCG": "Ala", "GCT": "Ala",
        "GGA": "Gly", "GGC": "Gly", "GGG": "Gly", "GGT": "Gly",
        "GTA": "Val", "GTC": "Val", "GTG": "Val", "GTT": "Val",
        "TAA": "STOP", "TAC": "Tyr", "TAG": "STOP", "TAT": "Tyr",
        "TCA": "Ser", "TCC": "Ser", "TCG": "Ser", "TCT": "Ser",
        "TGA": "STOP", "TGC": "Cys", "TGG": "Trp", "TGT": "Cys",
        "TTA": "Leu", "TTC": "Phe", "TTG": "Leu", "TTT": "Phe"
    }
    
    # For now, return the original sequence
    # In a full implementation, you would replace codons with preferred ones
    return sequence 