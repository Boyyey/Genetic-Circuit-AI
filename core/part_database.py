"""
Part Database Module for Genetic Circuit Design Platform.

This module provides comprehensive management of genetic parts and biological
components, including databases, search capabilities, and part validation.

Author: Genetic Circuit Design Team
Version: 1.0.0
License: MIT
"""

import logging
import json
import sqlite3
import requests
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import os
from models.parts import (
    GeneticPart, Promoter, Gene, Terminator, RibosomeBindingSite, 
    Protein, SmallMolecule, PartDatabase, PartType, Organism
)
from models.circuit import Circuit


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PartSearchCriteria:
    """Search criteria for genetic parts."""
    
    part_type: Optional[PartType] = None
    organism: Optional[Organism] = None
    name_pattern: Optional[str] = None
    function: Optional[str] = None
    strength_range: Optional[Tuple[float, float]] = None
    sequence_pattern: Optional[str] = None
    tags: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "part_type": self.part_type.value if self.part_type else None,
            "organism": self.organism.value if self.organism else None,
            "name_pattern": self.name_pattern,
            "function": self.function,
            "strength_range": self.strength_range,
            "sequence_pattern": self.sequence_pattern,
            "tags": self.tags
        }


class PartManager:
    """Advanced part manager for genetic parts database."""
    
    def __init__(self, database_path: str = "data/parts_database.db"):
        """Initialize the part manager."""
        self.database_path = database_path
        self.ensure_database_exists()
        self.initialize_database()
        
        # Load external databases
        self.external_sources = self._load_external_sources()
        
        logger.info("Part manager initialized successfully")
    
    def ensure_database_exists(self):
        """Ensure the database directory and file exist."""
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
    
    def initialize_database(self):
        """Initialize the SQLite database with tables."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.cursor()
            
            # Create parts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    part_type TEXT NOT NULL,
                    organism TEXT NOT NULL,
                    sequence TEXT,
                    properties TEXT,
                    metadata TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            # Create tags table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    part_id TEXT,
                    tag TEXT,
                    FOREIGN KEY (part_id) REFERENCES parts (id)
                )
            """)
            
            # Create relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    part_id_1 TEXT,
                    part_id_2 TEXT,
                    relationship_type TEXT,
                    strength REAL,
                    FOREIGN KEY (part_id_1) REFERENCES parts (id),
                    FOREIGN KEY (part_id_2) REFERENCES parts (id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_part_type ON parts(part_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_organism ON parts(organism)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON parts(name)")
            
            conn.commit()
    
    def _load_external_sources(self) -> Dict[str, Dict[str, Any]]:
        """Load external part databases."""
        return {
            "synbiohub": {
                "url": "https://synbiohub.org",
                "api_endpoint": "/api/v1/parts",
                "enabled": True
            },
            "igem": {
                "url": "https://parts.igem.org",
                "api_endpoint": "/api/parts",
                "enabled": True
            },
            "jbei": {
                "url": "https://public-registry.jbei.org",
                "api_endpoint": "/api/parts",
                "enabled": False
            }
        }
    
    def add_part(self, part: GeneticPart) -> bool:
        """Add a part to the database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Insert part
                cursor.execute("""
                    INSERT OR REPLACE INTO parts 
                    (id, name, part_type, organism, sequence, properties, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    part.id,
                    part.name,
                    part.part_type.value,
                    part.organism.value,
                    part.sequence.sequence if part.sequence else "",
                    json.dumps(part.properties),
                    json.dumps(part.metadata),
                    part.created_at.isoformat(),
                    part.updated_at.isoformat()
                ))
                
                # Add tags
                if hasattr(part, 'tags') and part.tags:
                    for tag in part.tags:
                        cursor.execute("""
                            INSERT OR IGNORE INTO tags (part_id, tag)
                            VALUES (?, ?)
                        """, (part.id, tag))
                
                conn.commit()
                logger.info(f"Part {part.name} added to database")
                return True
                
        except Exception as e:
            logger.error(f"Error adding part to database: {e}")
            return False
    
    def get_part(self, part_id: str) -> Optional[GeneticPart]:
        """Get a part by ID."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, name, part_type, organism, sequence, properties, metadata, created_at, updated_at
                    FROM parts WHERE id = ?
                """, (part_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_part(row)
                return None
                
        except Exception as e:
            logger.error(f"Error getting part from database: {e}")
            return None
    
    def search_parts(self, criteria: PartSearchCriteria) -> List[GeneticPart]:
        """Search parts based on criteria."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                query = "SELECT id, name, part_type, organism, sequence, properties, metadata, created_at, updated_at FROM parts WHERE 1=1"
                params = []
                
                if criteria.part_type:
                    query += " AND part_type = ?"
                    params.append(criteria.part_type.value)
                
                if criteria.organism:
                    query += " AND organism = ?"
                    params.append(criteria.organism.value)
                
                if criteria.name_pattern:
                    query += " AND name LIKE ?"
                    params.append(f"%{criteria.name_pattern}%")
                
                if criteria.function:
                    query += " AND properties LIKE ?"
                    params.append(f"%{criteria.function}%")
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_part(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Error searching parts: {e}")
            return []
    
    def _row_to_part(self, row: Tuple) -> GeneticPart:
        """Convert database row to GeneticPart object."""
        part_id, name, part_type, organism, sequence, properties, metadata, created_at, updated_at = row
        
        # Create appropriate part type
        if part_type == "promoter":
            part = Promoter(
                id=part_id,
                name=name,
                organism=Organism(organism)
            )
        elif part_type == "gene":
            part = Gene(
                id=part_id,
                name=name,
                organism=Organism(organism)
            )
        elif part_type == "terminator":
            part = Terminator(
                id=part_id,
                name=name,
                organism=Organism(organism)
            )
        elif part_type == "ribosome_binding_site":
            part = RibosomeBindingSite(
                id=part_id,
                name=name,
                organism=Organism(organism)
            )
        elif part_type == "protein":
            part = Protein(
                id=part_id,
                name=name,
                organism=Organism(organism)
            )
        elif part_type == "small_molecule":
            part = SmallMolecule(
                id=part_id,
                name=name,
                organism=Organism(organism)
            )
        else:
            part = GeneticPart(
                id=part_id,
                name=name,
                part_type=PartType(part_type),
                organism=Organism(organism)
            )
        
        # Set properties
        if properties:
            part.properties = json.loads(properties)
        
        if metadata:
            part.metadata = json.loads(metadata)
        
        if sequence:
            from models.parts import Sequence
            part.sequence = Sequence(sequence=sequence)
        
        part.created_at = datetime.fromisoformat(created_at)
        part.updated_at = datetime.fromisoformat(updated_at)
        
        return part
    
    def get_parts_by_type(self, part_type: PartType) -> List[GeneticPart]:
        """Get all parts of a specific type."""
        criteria = PartSearchCriteria(part_type=part_type)
        return self.search_parts(criteria)
    
    def get_parts_by_organism(self, organism: Organism) -> List[GeneticPart]:
        """Get all parts for a specific organism."""
        criteria = PartSearchCriteria(organism=organism)
        return self.search_parts(criteria)
    
    def update_part(self, part: GeneticPart) -> bool:
        """Update an existing part."""
        part.updated_at = datetime.now()
        return self.add_part(part)
    
    def delete_part(self, part_id: str) -> bool:
        """Delete a part from the database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Delete related records
                cursor.execute("DELETE FROM tags WHERE part_id = ?", (part_id,))
                cursor.execute("DELETE FROM relationships WHERE part_id_1 = ? OR part_id_2 = ?", (part_id, part_id))
                
                # Delete part
                cursor.execute("DELETE FROM parts WHERE id = ?", (part_id,))
                
                conn.commit()
                logger.info(f"Part {part_id} deleted from database")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting part from database: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Total parts
                cursor.execute("SELECT COUNT(*) FROM parts")
                total_parts = cursor.fetchone()[0]
                
                # Parts by type
                cursor.execute("SELECT part_type, COUNT(*) FROM parts GROUP BY part_type")
                parts_by_type = dict(cursor.fetchall())
                
                # Parts by organism
                cursor.execute("SELECT organism, COUNT(*) FROM parts GROUP BY organism")
                parts_by_organism = dict(cursor.fetchall())
                
                # Recent additions
                cursor.execute("SELECT COUNT(*) FROM parts WHERE created_at > datetime('now', '-7 days')")
                recent_additions = cursor.fetchone()[0]
                
                return {
                    "total_parts": total_parts,
                    "parts_by_type": parts_by_type,
                    "parts_by_organism": parts_by_organism,
                    "recent_additions": recent_additions
                }
                
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {}
    
    def export_database(self, filepath: str, format: str = "json") -> bool:
        """Export database to file."""
        try:
            if format.lower() == "json":
                return self._export_to_json(filepath)
            elif format.lower() == "csv":
                return self._export_to_csv(filepath)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            logger.error(f"Error exporting database: {e}")
            return False
    
    def _export_to_json(self, filepath: str) -> bool:
        """Export database to JSON format."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM parts")
                rows = cursor.fetchall()
                
                data = {
                    "export_date": datetime.now().isoformat(),
                    "parts": []
                }
                
                for row in rows:
                    part = self._row_to_part(row)
                    data["parts"].append(part.to_dict())
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"Database exported to {filepath}")
                return True
                
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return False
    
    def _export_to_csv(self, filepath: str) -> bool:
        """Export database to CSV format."""
        try:
            import pandas as pd
            
            with sqlite3.connect(self.database_path) as conn:
                df = pd.read_sql_query("SELECT * FROM parts", conn)
                df.to_csv(filepath, index=False)
                
                logger.info(f"Database exported to {filepath}")
                return True
                
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False
    
    def import_database(self, filepath: str, format: str = "json") -> bool:
        """Import database from file."""
        try:
            if format.lower() == "json":
                return self._import_from_json(filepath)
            elif format.lower() == "csv":
                return self._import_from_csv(filepath)
            else:
                logger.error(f"Unsupported import format: {format}")
                return False
                
        except Exception as e:
            logger.error(f"Error importing database: {e}")
            return False
    
    def _import_from_json(self, filepath: str) -> bool:
        """Import database from JSON format."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            success_count = 0
            for part_data in data.get("parts", []):
                part = GeneticPart.from_dict(part_data)
                if self.add_part(part):
                    success_count += 1
            
            logger.info(f"Imported {success_count} parts from {filepath}")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error importing from JSON: {e}")
            return False
    
    def _import_from_csv(self, filepath: str) -> bool:
        """Import database from CSV format."""
        try:
            import pandas as pd
            
            df = pd.read_csv(filepath)
            success_count = 0
            
            for _, row in df.iterrows():
                part_data = row.to_dict()
                part = GeneticPart.from_dict(part_data)
                if self.add_part(part):
                    success_count += 1
            
            logger.info(f"Imported {success_count} parts from {filepath}")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error importing from CSV: {e}")
            return False
    
    def search_external_databases(self, query: str, source: str = "synbiohub") -> List[Dict[str, Any]]:
        """Search external part databases."""
        if source not in self.external_sources or not self.external_sources[source]["enabled"]:
            logger.warning(f"External source {source} not available")
            return []
        
        try:
            source_config = self.external_sources[source]
            url = f"{source_config['url']}{source_config['api_endpoint']}"
            
            response = requests.get(url, params={"q": query}, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data.get("results", [])
            
        except Exception as e:
            logger.error(f"Error searching external database {source}: {e}")
            return []
    
    def validate_part(self, part: GeneticPart) -> List[str]:
        """Validate a genetic part and return error messages."""
        errors = []
        
        # Check required fields
        if not part.name:
            errors.append("Part must have a name")
        
        if not part.part_type:
            errors.append("Part must have a type")
        
        if not part.organism:
            errors.append("Part must have an organism")
        
        # Check sequence if present
        if part.sequence and part.sequence.sequence:
            if not self._validate_sequence(part.sequence.sequence, part.part_type):
                errors.append("Invalid sequence for part type")
        
        # Check for duplicate names
        existing_parts = self.search_parts(PartSearchCriteria(name_pattern=part.name))
        if existing_parts and existing_parts[0].id != part.id:
            errors.append(f"Part with name '{part.name}' already exists")
        
        return errors
    
    def _validate_sequence(self, sequence: str, part_type: PartType) -> bool:
        """Validate sequence for part type."""
        if part_type == PartType.PROMOTER:
            # Check for promoter consensus sequences
            consensus_sequences = ["TATAAT", "TTGACA", "TATA"]
            return any(consensus in sequence.upper() for consensus in consensus_sequences)
        
        elif part_type == PartType.GENE:
            # Check for start and stop codons
            start_codons = ["ATG", "GTG", "TTG"]
            stop_codons = ["TAA", "TAG", "TGA"]
            
            has_start = any(start in sequence.upper() for start in start_codons)
            has_stop = any(stop in sequence.upper() for stop in stop_codons)
            
            return has_start and has_stop
        
        elif part_type == PartType.TERMINATOR:
            # Check for terminator patterns
            terminator_patterns = ["AATAAA", "TTTTTT", "GCGCGC"]
            return any(pattern in sequence.upper() for pattern in terminator_patterns)
        
        else:
            # Basic validation for other part types
            return len(sequence) > 0 and all(base in "ATCG" for base in sequence.upper())


class PartDatabase:
    """Main part database interface."""
    
    def __init__(self, database_path: str = "data/parts_database.db"):
        """Initialize the part database."""
        self.manager = PartManager(database_path)
        self.load_default_parts()
    
    def load_default_parts(self):
        """Load default parts into the database."""
        default_parts = self._get_default_parts()
        
        for part in default_parts:
            if not self.manager.get_part(part.id):
                self.manager.add_part(part)
    
    def _get_default_parts(self) -> List[GeneticPart]:
        """Get default genetic parts."""
        return [
            # Promoters
            Promoter(
                name="pTac",
                organism=Organism.E_COLI,
                properties={
                    "strength": 1.0,
                    "type": "inducible",
                    "inducer": "IPTG",
                    "description": "Tac promoter, inducible by IPTG"
                }
            ),
            Promoter(
                name="pLac",
                organism=Organism.E_COLI,
                properties={
                    "strength": 0.8,
                    "type": "inducible",
                    "inducer": "IPTG",
                    "description": "Lac promoter, inducible by IPTG"
                }
            ),
            Promoter(
                name="pTet",
                organism=Organism.E_COLI,
                properties={
                    "strength": 1.2,
                    "type": "inducible",
                    "inducer": "aTc",
                    "description": "Tet promoter, inducible by aTc"
                }
            ),
            
            # Genes
            Gene(
                name="LacI",
                organism=Organism.E_COLI,
                properties={
                    "product": "LacI repressor",
                    "function": "repressor",
                    "target": "lac operon",
                    "description": "LacI repressor protein"
                }
            ),
            Gene(
                name="TetR",
                organism=Organism.E_COLI,
                properties={
                    "product": "TetR repressor",
                    "function": "repressor",
                    "target": "tet operon",
                    "description": "TetR repressor protein"
                }
            ),
            Gene(
                name="GFP",
                organism=Organism.E_COLI,
                properties={
                    "product": "Green Fluorescent Protein",
                    "function": "reporter",
                    "excitation": "488 nm",
                    "emission": "509 nm",
                    "description": "Green fluorescent protein reporter"
                }
            ),
            
            # Terminators
            Terminator(
                name="T1",
                organism=Organism.E_COLI,
                properties={
                    "efficiency": 0.95,
                    "type": "rho-independent",
                    "description": "Strong rho-independent terminator"
                }
            ),
            Terminator(
                name="T7",
                organism=Organism.E_COLI,
                properties={
                    "efficiency": 0.98,
                    "type": "rho-independent",
                    "description": "T7 terminator"
                }
            )
        ]
    
    def add_part(self, part: GeneticPart) -> bool:
        """Add a part to the database."""
        return self.manager.add_part(part)
    
    def get_part(self, part_id: str) -> Optional[GeneticPart]:
        """Get a part by ID."""
        return self.manager.get_part(part_id)
    
    def search_parts(self, criteria: PartSearchCriteria) -> List[GeneticPart]:
        """Search parts based on criteria."""
        return self.manager.search_parts(criteria)
    
    def get_parts_by_type(self, part_type: PartType) -> List[GeneticPart]:
        """Get all parts of a specific type."""
        return self.manager.get_parts_by_type(part_type)
    
    def get_parts_by_organism(self, organism: Organism) -> List[GeneticPart]:
        """Get all parts for a specific organism."""
        return self.manager.get_parts_by_organism(organism)
    
    def update_part(self, part: GeneticPart) -> bool:
        """Update an existing part."""
        return self.manager.update_part(part)
    
    def delete_part(self, part_id: str) -> bool:
        """Delete a part from the database."""
        return self.manager.delete_part(part_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.manager.get_statistics()
    
    def export_database(self, filepath: str, format: str = "json") -> bool:
        """Export database to file."""
        return self.manager.export_database(filepath, format)
    
    def import_database(self, filepath: str, format: str = "json") -> bool:
        """Import database from file."""
        return self.manager.import_database(filepath, format)
    
    def validate_part(self, part: GeneticPart) -> List[str]:
        """Validate a genetic part."""
        return self.manager.validate_part(part) 