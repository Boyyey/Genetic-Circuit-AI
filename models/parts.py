"""
Genetic part models for biological components.

This module defines data structures for representing genetic parts including
promoters, genes, terminators, ribosome binding sites, proteins, and small molecules.
It provides a comprehensive framework for modeling biological components with
their properties, interactions, and regulatory mechanisms.

Author: Genetic Circuit Design Team
Version: 1.0.0
License: MIT
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
import numpy as np
from pydantic import BaseModel, Field, validator
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


class PartType(Enum):
    """Types of genetic parts."""
    PROMOTER = "promoter"
    GENE = "gene"
    TERMINATOR = "terminator"
    RBS = "ribosome_binding_site"
    PROTEIN = "protein"
    SMALL_MOLECULE = "small_molecule"
    PLASMID = "plasmid"
    CHROMOSOME = "chromosome"
    OPERON = "operon"
    ENZYME = "enzyme"
    RECEPTOR = "receptor"
    TRANSPORTER = "transporter"


class PromoterType(Enum):
    """Types of promoters."""
    CONSTITUTIVE = "constitutive"
    INDUCIBLE = "inducible"
    REPRESSIBLE = "repressible"
    DUAL = "dual"
    SYNTHETIC = "synthetic"
    NATURAL = "natural"


class RegulationMechanism(Enum):
    """Types of regulation mechanisms."""
    TRANSCRIPTIONAL = "transcriptional"
    TRANSLATIONAL = "translational"
    POST_TRANSLATIONAL = "post_translational"
    ALLOSTERIC = "allosteric"
    COMPETITIVE = "competitive"
    COOPERATIVE = "cooperative"


class Organism(Enum):
    """Supported organisms."""
    E_COLI = "E. coli"
    S_CEREVISIAE = "S. cerevisiae"
    B_SUBTILIS = "B. subtilis"
    P_PUTIDA = "P. putida"
    MAMMALIAN = "mammalian"
    SYNTHETIC = "synthetic"


@dataclass
class Sequence:
    """Represents a DNA, RNA, or protein sequence."""
    
    sequence: str = ""
    sequence_type: str = "DNA"  # DNA, RNA, PROTEIN
    length: int = 0
    gc_content: float = 0.0
    melting_temperature: float = 0.0
    secondary_structure: Optional[str] = None
    annotations: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived properties."""
        if self.sequence:
            self.length = len(self.sequence)
            self.gc_content = self._calculate_gc_content()
            self.melting_temperature = self._calculate_melting_temperature()
    
    def _calculate_gc_content(self) -> float:
        """Calculate GC content percentage."""
        if not self.sequence:
            return 0.0
        gc_count = self.sequence.upper().count('G') + self.sequence.upper().count('C')
        return (gc_count / len(self.sequence)) * 100
    
    def _calculate_melting_temperature(self) -> float:
        """Calculate melting temperature using Wallace rule."""
        if not self.sequence or self.sequence_type != "DNA":
            return 0.0
        # Wallace rule: Tm = 2°C(A+T) + 4°C(G+C)
        a_count = self.sequence.upper().count('A')
        t_count = self.sequence.upper().count('T')
        g_count = self.sequence.upper().count('G')
        c_count = self.sequence.upper().count('C')
        return 2 * (a_count + t_count) + 4 * (g_count + c_count)
    
    def to_biopython(self) -> SeqRecord:
        """Convert to BioPython SeqRecord."""
        return SeqRecord(
            Seq(self.sequence),
            id=self.sequence_type,
            description=f"Length: {self.length}, GC: {self.gc_content:.1f}%"
        )
    
    def reverse_complement(self) -> "Sequence":
        """Get reverse complement of DNA sequence."""
        if self.sequence_type != "DNA":
            raise ValueError("Reverse complement only available for DNA sequences")
        
        complement = str(Seq(self.sequence).complement())
        return Sequence(
            sequence=complement[::-1],
            sequence_type="DNA",
            annotations=self.annotations
        )


@dataclass
class KineticParameters:
    """Kinetic parameters for biological reactions."""
    
    k_on: float = 0.0  # Association rate constant
    k_off: float = 0.0  # Dissociation rate constant
    k_cat: float = 0.0  # Catalytic rate constant
    k_m: float = 0.0  # Michaelis constant
    v_max: float = 0.0  # Maximum velocity
    k_d: float = 0.0  # Dissociation constant
    hill_coefficient: float = 1.0  # Hill coefficient
    cooperativity: float = 1.0  # Cooperativity factor
    
    @property
    def equilibrium_constant(self) -> float:
        """Calculate equilibrium constant."""
        if self.k_off > 0:
            return self.k_on / self.k_off
        return 0.0
    
    @property
    def catalytic_efficiency(self) -> float:
        """Calculate catalytic efficiency."""
        if self.k_m > 0:
            return self.k_cat / self.k_m
        return 0.0


@dataclass
class GeneticPart:
    """Base class for genetic parts."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    part_type: PartType = PartType.GENE
    organism: Organism = Organism.E_COLI
    sequence: Optional[Sequence] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    kinetic_parameters: Optional[KineticParameters] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate and set default properties."""
        if not self.name:
            self.name = f"{self.part_type.value}_{self.id[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert part to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "part_type": self.part_type.value,
            "organism": self.organism.value,
            "sequence": self.sequence.__dict__ if self.sequence else None,
            "properties": self.properties,
            "kinetic_parameters": self.kinetic_parameters.__dict__ if self.kinetic_parameters else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneticPart":
        """Create part from dictionary representation."""
        if data.get("sequence"):
            data["sequence"] = Sequence(**data["sequence"])
        if data.get("kinetic_parameters"):
            data["kinetic_parameters"] = KineticParameters(**data["kinetic_parameters"])
        
        data["part_type"] = PartType(data["part_type"])
        data["organism"] = Organism(data["organism"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        
        return cls(**data)
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property value."""
        return self.properties.get(key, default)
    
    def set_property(self, key: str, value: Any) -> None:
        """Set a property value."""
        self.properties[key] = value
        self.updated_at = datetime.now()
    
    def validate(self) -> List[str]:
        """Validate the genetic part and return error messages."""
        errors = []
        
        if not self.name:
            errors.append("Part must have a name")
        
        if self.sequence and self.sequence.length == 0:
            errors.append("Sequence cannot be empty if provided")
        
        return errors


@dataclass
class Promoter(GeneticPart):
    """Represents a promoter sequence."""
    
    promoter_type: PromoterType = PromoterType.CONSTITUTIVE
    strength: float = 1.0  # Relative promoter strength
    regulation_mechanisms: List[RegulationMechanism] = field(default_factory=list)
    inducers: List[str] = field(default_factory=list)
    repressors: List[str] = field(default_factory=list)
    transcription_start_site: Optional[int] = None
    binding_sites: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize promoter-specific properties."""
        super().__post_init__()
        self.part_type = PartType.PROMOTER
        
        # Set default properties
        if "promoter_strength" not in self.properties:
            self.properties["promoter_strength"] = self.strength
    
    def calculate_activity(self, inducer_concentrations: Dict[str, float] = None,
                          repressor_concentrations: Dict[str, float] = None) -> float:
        """Calculate promoter activity based on regulatory molecules."""
        activity = self.strength
        
        # Apply inducer effects
        if inducer_concentrations:
            for inducer, concentration in inducer_concentrations.items():
                if inducer in self.inducers:
                    # Simple activation model
                    k_d = self.get_property(f"{inducer}_kd", 1.0)
                    activity *= (concentration / (k_d + concentration))
        
        # Apply repressor effects
        if repressor_concentrations:
            for repressor, concentration in repressor_concentrations.items():
                if repressor in self.repressors:
                    # Simple repression model
                    k_d = self.get_property(f"{repressor}_kd", 1.0)
                    activity *= (k_d / (k_d + concentration))
        
        return max(0.0, min(1.0, activity))  # Clamp between 0 and 1


@dataclass
class Gene(GeneticPart):
    """Represents a gene sequence."""
    
    product: str = ""  # Name of the protein product
    function: str = ""  # Biological function
    molecular_weight: float = 0.0  # kDa
    is_essential: bool = False
    expression_level: float = 1.0  # Relative expression level
    degradation_rate: float = 0.0  # Protein degradation rate
    localization: str = "cytoplasm"  # Cellular localization
    domains: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize gene-specific properties."""
        super().__post_init__()
        self.part_type = PartType.GENE
    
    def calculate_expression(self, promoter_activity: float, 
                           translation_efficiency: float = 1.0) -> float:
        """Calculate gene expression level."""
        return promoter_activity * self.expression_level * translation_efficiency


@dataclass
class Terminator(GeneticPart):
    """Represents a transcription terminator."""
    
    efficiency: float = 0.95  # Termination efficiency (0-1)
    termination_signal: str = "rho-independent"  # Type of termination
    hairpin_structure: Optional[str] = None  # Predicted hairpin structure
    
    def __post_init__(self):
        """Initialize terminator-specific properties."""
        super().__post_init__()
        self.part_type = PartType.TERMINATOR


@dataclass
class RibosomeBindingSite(GeneticPart):
    """Represents a ribosome binding site."""
    
    strength: float = 1.0  # RBS strength
    shine_dalgarno_sequence: str = ""
    spacing: int = 0  # Distance to start codon
    efficiency: float = 1.0  # Translation efficiency
    
    def __post_init__(self):
        """Initialize RBS-specific properties."""
        super().__post_init__()
        self.part_type = PartType.RBS
    
    def calculate_translation_efficiency(self, mrna_concentration: float) -> float:
        """Calculate translation efficiency."""
        return self.efficiency * self.strength * mrna_concentration


@dataclass
class Protein(GeneticPart):
    """Represents a protein molecule."""
    
    molecular_weight: float = 0.0  # kDa
    isoelectric_point: float = 7.0
    stability: float = 1.0  # Protein stability
    activity: float = 1.0  # Protein activity
    binding_partners: List[str] = field(default_factory=list)
    post_translational_modifications: List[str] = field(default_factory=list)
    structure_file: Optional[str] = None  # Path to 3D structure file
    
    def __post_init__(self):
        """Initialize protein-specific properties."""
        super().__post_init__()
        self.part_type = PartType.PROTEIN
    
    def calculate_activity(self, conditions: Dict[str, float] = None) -> float:
        """Calculate protein activity under given conditions."""
        base_activity = self.activity * self.stability
        
        if conditions:
            # Apply environmental effects
            if "temperature" in conditions:
                temp = conditions["temperature"]
                # Simple temperature effect model
                if temp > 37:  # Above optimal temperature
                    base_activity *= max(0.1, 1.0 - (temp - 37) * 0.05)
                elif temp < 20:  # Below optimal temperature
                    base_activity *= max(0.5, 1.0 - (20 - temp) * 0.02)
            
            if "ph" in conditions:
                ph = conditions["ph"]
                # pH effect model
                optimal_ph = self.isoelectric_point
                ph_diff = abs(ph - optimal_ph)
                base_activity *= max(0.1, 1.0 - ph_diff * 0.1)
        
        return max(0.0, min(1.0, base_activity))


@dataclass
class SmallMolecule(GeneticPart):
    """Represents a small molecule."""
    
    molecular_formula: str = ""
    molecular_weight: float = 0.0
    charge: int = 0
    solubility: float = 0.0  # mg/mL
    permeability: float = 0.0  # Membrane permeability
    toxicity: float = 0.0  # Toxicity level (0-1)
    metabolites: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize small molecule-specific properties."""
        super().__post_init__()
        self.part_type = PartType.SMALL_MOLECULE
    
    def calculate_uptake(self, external_concentration: float,
                        membrane_permeability: float = None) -> float:
        """Calculate cellular uptake rate."""
        perm = membrane_permeability or self.permeability
        return perm * external_concentration


class PartDatabase(BaseModel):
    """Database of genetic parts."""
    
    parts: Dict[str, GeneticPart] = Field(default_factory=dict)
    organism: Organism = Organism.E_COLI
    version: str = "1.0.0"
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def add_part(self, part: GeneticPart) -> None:
        """Add a part to the database."""
        self.parts[part.id] = part
        self.last_updated = datetime.now()
    
    def get_part(self, part_id: str) -> Optional[GeneticPart]:
        """Get a part by ID."""
        return self.parts.get(part_id)
    
    def search_parts(self, query: str, part_type: Optional[PartType] = None) -> List[GeneticPart]:
        """Search parts by name or properties."""
        results = []
        query_lower = query.lower()
        
        for part in self.parts.values():
            if part_type and part.part_type != part_type:
                continue
            
            if (query_lower in part.name.lower() or
                query_lower in part.properties.get("description", "").lower()):
                results.append(part)
        
        return results
    
    def get_parts_by_type(self, part_type: PartType) -> List[GeneticPart]:
        """Get all parts of a specific type."""
        return [part for part in self.parts.values() if part.part_type == part_type]
    
    def validate_database(self) -> List[str]:
        """Validate all parts in the database."""
        errors = []
        for part_id, part in self.parts.items():
            part_errors = part.validate()
            for error in part_errors:
                errors.append(f"Part {part_id}: {error}")
        return errors
    
    def export_to_json(self, filepath: str) -> None:
        """Export database to JSON file."""
        data = {
            "organism": self.organism.value,
            "version": self.version,
            "last_updated": self.last_updated.isoformat(),
            "parts": {part_id: part.to_dict() for part_id, part in self.parts.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str) -> "PartDatabase":
        """Load database from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        database = cls(
            organism=Organism(data["organism"]),
            version=data["version"],
            last_updated=datetime.fromisoformat(data["last_updated"])
        )
        
        for part_id, part_data in data["parts"].items():
            part = GeneticPart.from_dict(part_data)
            database.parts[part_id] = part
        
        return database 