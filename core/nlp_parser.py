"""
Natural Language Processing for Genetic Circuit Design.

This module provides advanced NLP capabilities for parsing natural language
descriptions of genetic circuits and converting them into structured logic
representations. It uses state-of-the-art language models and custom parsing
algorithms to understand complex biological requirements.

Author: Genetic Circuit Design Team
Version: 1.0.0
License: MIT
"""

import re
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import numpy as np
import openai
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    import logging
    logging.warning("transformers or torch/torchvision not available; NLP features will be limited.")
from models.circuit import Circuit, CircuitNode, CircuitEdge, LogicGate, LogicGateType
from models.parts import GeneticPart, Promoter, Gene, Organism


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParsedLogic:
    """Represents parsed logic from natural language."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    logic_expression: str = ""
    variables: List[str] = field(default_factory=list)
    operators: List[str] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    temporal_constraints: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "logic_expression": self.logic_expression,
            "variables": self.variables,
            "operators": self.operators,
            "conditions": self.conditions,
            "temporal_constraints": self.temporal_constraints,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class BiologicalEntity:
    """Represents a biological entity extracted from text."""
    
    name: str = ""
    entity_type: str = ""  # gene, protein, molecule, condition, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "properties": self.properties,
            "relationships": self.relationships,
            "confidence": self.confidence
        }


class CircuitLogicExtractor:
    """Extracts logic expressions from natural language circuit descriptions."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the logic extractor."""
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
        
        # Initialize language models
        self._initialize_models()
        
        # Define biological vocabulary
        self.biological_terms = self._load_biological_vocabulary()
        
        # Define logic patterns
        self.logic_patterns = self._load_logic_patterns()
    
    def _initialize_models(self):
        """Initialize language models for text processing."""
        if TRANSFORMERS_AVAILABLE:
            try:
                # Load sentiment analysis model for confidence scoring
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
                # Load NER model for entity extraction
                self.ner_model = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english"
                )
                logger.info("Language models initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize language models: {e}")
                self.sentiment_analyzer = None
                self.ner_model = None
        else:
            self.sentiment_analyzer = None
            self.ner_model = None
    
    def _load_biological_vocabulary(self) -> Dict[str, List[str]]:
        """Load biological vocabulary for entity recognition."""
        return {
            "genes": [
                "gene", "mRNA", "transcript", "coding sequence", "CDS",
                "lacZ", "gfp", "rfp", "yfp", "cfp", "lux", "cat", "amp",
                "tet", "kan", "spec", "chloramphenicol", "ampicillin"
            ],
            "proteins": [
                "protein", "enzyme", "transcription factor", "repressor",
                "activator", "polymerase", "ribosome", "LacI", "TetR",
                "AraC", "CRP", "sigma factor", "RNA polymerase"
            ],
            "molecules": [
                "sugar", "glucose", "lactose", "arabinose", "IPTG",
                "aTc", "tetracycline", "antibiotic", "inducer", "repressor",
                "metabolite", "substrate", "product"
            ],
            "conditions": [
                "high", "low", "absent", "present", "stress", "heat",
                "cold", "pH", "temperature", "oxygen", "anaerobic",
                "aerobic", "stationary phase", "exponential phase"
            ],
            "temporal": [
                "oscillate", "oscillating", "periodic", "cycle", "rhythm",
                "every", "hours", "minutes", "seconds", "continuous",
                "pulse", "spike", "sustained", "transient"
            ],
            "operators": [
                "and", "or", "not", "only when", "if", "then", "else",
                "when", "while", "during", "in the presence of",
                "in the absence of", "requires", "inhibits", "activates"
            ]
        }
    
    def _load_logic_patterns(self) -> List[Dict[str, Any]]:
        """Load patterns for logic extraction."""
        return [
            {
                "pattern": r"express(?:es|ed|ing)?\s+(\w+)\s+only\s+when\s+(.+)",
                "logic": "AND",
                "description": "Conditional expression"
            },
            {
                "pattern": r"(\w+)\s+activates?\s+(\w+)",
                "logic": "ACTIVATION",
                "description": "Activation relationship"
            },
            {
                "pattern": r"(\w+)\s+represses?\s+(\w+)",
                "logic": "REPRESSION", 
                "description": "Repression relationship"
            },
            {
                "pattern": r"oscillat(?:es|ed|ing)\s+(?:every\s+)?(\d+)\s*(hours?|minutes?|seconds?)",
                "logic": "OSCILLATOR",
                "description": "Oscillatory behavior"
            },
            {
                "pattern": r"(\w+)\s+high\s+and\s+(\w+)\s+low",
                "logic": "AND_NOT",
                "description": "High-low condition"
            },
            {
                "pattern": r"(\w+)\s+or\s+(\w+)",
                "logic": "OR",
                "description": "OR condition"
            }
        ]
    
    def extract_logic(self, text: str) -> ParsedLogic:
        """Extract logic from natural language text."""
        logger.info(f"Extracting logic from: {text}")
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Extract entities
        entities = self._extract_entities(cleaned_text)
        
        # Extract logic patterns
        logic_patterns = self._extract_logic_patterns(cleaned_text)
        
        # Generate logic expression
        logic_expression = self._generate_logic_expression(entities, logic_patterns)
        
        # Calculate confidence
        confidence = self._calculate_confidence(cleaned_text, entities, logic_patterns)
        
        # Create parsed logic object
        parsed_logic = ParsedLogic(
            logic_expression=logic_expression,
            variables=[entity.name for entity in entities],
            operators=[pattern["logic"] for pattern in logic_patterns],
            conditions=self._extract_conditions(cleaned_text),
            temporal_constraints=self._extract_temporal_constraints(cleaned_text),
            confidence=confidence,
            metadata={
                "original_text": text,
                "cleaned_text": cleaned_text,
                "entities": [entity.to_dict() for entity in entities],
                "patterns": logic_patterns
            }
        )
        
        logger.info(f"Extracted logic: {logic_expression} (confidence: {confidence:.2f})")
        return parsed_logic
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation that might interfere with parsing
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize biological terms
        text = self._normalize_biological_terms(text)
        
        return text.strip()
    
    def _normalize_biological_terms(self, text: str) -> str:
        """Normalize biological terms for consistent parsing."""
        # Common abbreviations and variations
        replacements = {
            "gene a": "gene_a",
            "gene b": "gene_b", 
            "sugar x": "sugar_x",
            "stress y": "stress_y",
            "protein": "protein",
            "mrna": "mRNA",
            "dna": "DNA",
            "rna": "RNA"
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _extract_entities(self, text: str) -> List[BiologicalEntity]:
        """Extract biological entities from text."""
        entities = []
        
        # Use NER model if available
        if self.ner_model:
            try:
                ner_results = self.ner_model(text)
                for result in ner_results:
                    entity = BiologicalEntity(
                        name=result["word"],
                        entity_type=result["entity_group"],
                        confidence=result["score"]
                    )
                    entities.append(entity)
            except Exception as e:
                logger.warning(f"NER extraction failed: {e}")
        
        # Fallback to rule-based extraction
        if not entities:
            entities = self._rule_based_entity_extraction(text)
        
        return entities
    
    def _rule_based_entity_extraction(self, text: str) -> List[BiologicalEntity]:
        """Extract entities using rule-based approach."""
        entities = []
        
        # Extract genes
        gene_pattern = r'\b(gene_[a-z]|gfp|rfp|yfp|cfp|lux|lac[z]?|tet[a-z]?|amp[a-z]?|kan[a-z]?|spec[a-z]?)\b'
        for match in re.finditer(gene_pattern, text):
            entity = BiologicalEntity(
                name=match.group(1),
                entity_type="gene",
                confidence=0.8
            )
            entities.append(entity)
        
        # Extract molecules
        molecule_pattern = r'\b(sugar_[a-z]|glucose|lactose|arabinose|iptg|atc|tetracycline|antibiotic)\b'
        for match in re.finditer(molecule_pattern, text):
            entity = BiologicalEntity(
                name=match.group(1),
                entity_type="molecule",
                confidence=0.8
            )
            entities.append(entity)
        
        # Extract conditions
        condition_pattern = r'\b(high|low|absent|present|stress_[a-z]|heat|cold|anaerobic|aerobic)\b'
        for match in re.finditer(condition_pattern, text):
            entity = BiologicalEntity(
                name=match.group(1),
                entity_type="condition",
                confidence=0.7
            )
            entities.append(entity)
        
        return entities
    
    def _extract_logic_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract logic patterns from text."""
        patterns = []
        
        for pattern_info in self.logic_patterns:
            matches = re.finditer(pattern_info["pattern"], text, re.IGNORECASE)
            for match in matches:
                pattern = {
                    "logic": pattern_info["logic"],
                    "description": pattern_info["description"],
                    "matches": match.groups(),
                    "span": match.span(),
                    "confidence": 0.8
                }
                patterns.append(pattern)
        
        return patterns
    
    def _generate_logic_expression(self, entities: List[BiologicalEntity], 
                                 patterns: List[Dict[str, Any]]) -> str:
        """Generate logic expression from entities and patterns."""
        if not patterns:
            return ""
        
        # Start with the first pattern
        expression = self._pattern_to_logic(patterns[0])
        
        # Combine with additional patterns
        for pattern in patterns[1:]:
            pattern_logic = self._pattern_to_logic(pattern)
            if pattern_logic:
                expression = f"({expression}) AND ({pattern_logic})"
        
        return expression
    
    def _pattern_to_logic(self, pattern: Dict[str, Any]) -> str:
        """Convert a pattern to logic expression."""
        logic_type = pattern["logic"]
        matches = pattern["matches"]
        
        if logic_type == "AND":
            if len(matches) >= 2:
                return f"{matches[0]}_HIGH AND {matches[1]}_HIGH"
        elif logic_type == "OR":
            if len(matches) >= 2:
                return f"{matches[0]}_HIGH OR {matches[1]}_HIGH"
        elif logic_type == "AND_NOT":
            if len(matches) >= 2:
                return f"{matches[0]}_HIGH AND NOT {matches[1]}_HIGH"
        elif logic_type == "OSCILLATOR":
            if len(matches) >= 2:
                period = matches[0]
                unit = matches[1]
                return f"OSCILLATOR_{period}_{unit}"
        elif logic_type == "ACTIVATION":
            if len(matches) >= 2:
                return f"{matches[0]}_ACTIVATES_{matches[1]}"
        elif logic_type == "REPRESSION":
            if len(matches) >= 2:
                return f"{matches[0]}_REPRESSES_{matches[1]}"
        
        return ""
    
    def _extract_conditions(self, text: str) -> List[Dict[str, Any]]:
        """Extract conditional statements from text."""
        conditions = []
        
        # Extract "only when" conditions
        only_when_pattern = r"only\s+when\s+(.+)"
        for match in re.finditer(only_when_pattern, text):
            condition = {
                "type": "only_when",
                "condition": match.group(1),
                "confidence": 0.8
            }
            conditions.append(condition)
        
        # Extract "if" conditions
        if_pattern = r"if\s+(.+?)\s+then\s+(.+)"
        for match in re.finditer(if_pattern, text):
            condition = {
                "type": "if_then",
                "condition": match.group(1),
                "action": match.group(2),
                "confidence": 0.8
            }
            conditions.append(condition)
        
        return conditions
    
    def _extract_temporal_constraints(self, text: str) -> List[Dict[str, Any]]:
        """Extract temporal constraints from text."""
        constraints = []
        
        # Extract oscillatory patterns
        oscillator_pattern = r"oscillat(?:es|ed|ing)\s+(?:every\s+)?(\d+)\s*(hours?|minutes?|seconds?)"
        for match in re.finditer(oscillator_pattern, text):
            constraint = {
                "type": "oscillator",
                "period": int(match.group(1)),
                "unit": match.group(2),
                "confidence": 0.9
            }
            constraints.append(constraint)
        
        # Extract timing patterns
        timing_pattern = r"(\d+)\s*(hours?|minutes?|seconds?)\s+(?:later|after|before)"
        for match in re.finditer(timing_pattern, text):
            constraint = {
                "type": "timing",
                "duration": int(match.group(1)),
                "unit": match.group(2),
                "confidence": 0.7
            }
            constraints.append(constraint)
        
        return constraints
    
    def _calculate_confidence(self, text: str, entities: List[BiologicalEntity], 
                            patterns: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the extraction."""
        confidence = 0.0
        
        # Base confidence from entity extraction
        if entities:
            entity_confidence = np.mean([entity.confidence for entity in entities])
            confidence += entity_confidence * 0.4
        
        # Pattern matching confidence
        if patterns:
            pattern_confidence = np.mean([pattern["confidence"] for pattern in patterns])
            confidence += pattern_confidence * 0.4
        
        # Text complexity factor
        text_complexity = min(len(text.split()) / 20.0, 1.0)  # Normalize by expected length
        confidence += text_complexity * 0.2
        
        # Use sentiment analysis if available
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer(text)[0]
                if sentiment["label"] == "POSITIVE":
                    confidence += 0.1
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
        
        return min(confidence, 1.0)


class NLPParser:
    """Main NLP parser for genetic circuit descriptions."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the NLP parser."""
        self.logic_extractor = CircuitLogicExtractor(openai_api_key)
        self.openai_api_key = openai_api_key
        
        # Load circuit templates
        self.circuit_templates = self._load_circuit_templates()
        
        # Initialize OpenAI client if API key is provided
        if openai_api_key:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")
                self.openai_client = None
        else:
            self.openai_client = None
    
    def _load_circuit_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined circuit templates."""
        return {
            "simple_repression": {
                "description": "Simple repression circuit",
                "components": ["promoter", "repressor", "gene"],
                "logic": "NOT repressor",
                "complexity": "low"
            },
            "simple_activation": {
                "description": "Simple activation circuit", 
                "components": ["promoter", "activator", "gene"],
                "logic": "activator",
                "complexity": "low"
            },
            "and_gate": {
                "description": "AND logic gate",
                "components": ["promoter1", "promoter2", "repressor1", "repressor2", "gene"],
                "logic": "promoter1 AND promoter2 AND NOT repressor1 AND NOT repressor2",
                "complexity": "medium"
            },
            "or_gate": {
                "description": "OR logic gate",
                "components": ["promoter1", "promoter2", "gene"],
                "logic": "promoter1 OR promoter2",
                "complexity": "medium"
            },
            "oscillator": {
                "description": "Oscillatory circuit",
                "components": ["promoter", "repressor", "gene", "delay"],
                "logic": "OSCILLATOR",
                "complexity": "high"
            },
            "feedforward": {
                "description": "Feedforward loop",
                "components": ["input", "intermediate", "output"],
                "logic": "input AND intermediate",
                "complexity": "medium"
            },
            "feedback": {
                "description": "Feedback loop",
                "components": ["input", "output", "feedback"],
                "logic": "input AND feedback",
                "complexity": "high"
            }
        }
    
    def parse_circuit_description(self, description: str, organism: str = "E. coli") -> Dict[str, Any]:
        """Parse a natural language circuit description."""
        logger.info(f"Parsing circuit description: {description}")
        
        # Extract logic from description
        parsed_logic = self.logic_extractor.extract_logic(description)
        
        # Identify circuit template
        template = self._identify_circuit_template(description, parsed_logic)
        
        # Generate circuit structure
        circuit_structure = self._generate_circuit_structure(parsed_logic, template, organism)
        
        # Use AI for advanced parsing if available
        if self.openai_client:
            ai_enhanced = self._enhance_with_ai(description, parsed_logic, circuit_structure)
            circuit_structure.update(ai_enhanced)
        
        result = {
            "parsed_logic": parsed_logic.to_dict(),
            "template": template,
            "circuit_structure": circuit_structure,
            "confidence": parsed_logic.confidence,
            "metadata": {
                "original_description": description,
                "organism": organism,
                "parsing_timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(f"Parsing completed with confidence: {parsed_logic.confidence:.2f}")
        return result
    
    def _identify_circuit_template(self, description: str, parsed_logic: ParsedLogic) -> str:
        """Identify the most suitable circuit template."""
        # Score each template based on logic and description
        template_scores = {}
        
        for template_name, template_info in self.circuit_templates.items():
            score = 0.0
            
            # Score based on logic operators
            logic_expression = parsed_logic.logic_expression.lower()
            template_logic = template_info["logic"].lower()
            
            if "and" in logic_expression and "and" in template_logic:
                score += 0.3
            if "or" in logic_expression and "or" in template_logic:
                score += 0.3
            if "not" in logic_expression and "not" in template_logic:
                score += 0.2
            if "oscillator" in logic_expression and "oscillator" in template_logic:
                score += 0.5
            
            # Score based on description keywords
            description_lower = description.lower()
            if template_name in description_lower:
                score += 0.4
            if template_info["description"].lower() in description_lower:
                score += 0.3
            
            # Score based on complexity
            if len(parsed_logic.variables) <= 2 and template_info["complexity"] == "low":
                score += 0.2
            elif len(parsed_logic.variables) <= 4 and template_info["complexity"] == "medium":
                score += 0.2
            elif len(parsed_logic.variables) > 4 and template_info["complexity"] == "high":
                score += 0.2
            
            template_scores[template_name] = score
        
        # Return the template with highest score
        best_template = max(template_scores.items(), key=lambda x: x[1])
        return best_template[0]
    
    def _generate_circuit_structure(self, parsed_logic: ParsedLogic, 
                                  template: str, organism: str) -> Dict[str, Any]:
        """Generate circuit structure from parsed logic and template."""
        template_info = self.circuit_templates[template]
        
        # Create basic circuit structure
        structure = {
            "template": template,
            "components": [],
            "connections": [],
            "parameters": {},
            "logic_gates": []
        }
        
        # Add components based on template
        for component_type in template_info["components"]:
            component = self._create_component(component_type, organism)
            structure["components"].append(component)
        
        # Add connections based on logic
        connections = self._generate_connections(parsed_logic, template_info)
        structure["connections"] = connections
        
        # Add logic gates
        logic_gates = self._generate_logic_gates(parsed_logic)
        structure["logic_gates"] = logic_gates
        
        # Add parameters
        parameters = self._generate_parameters(parsed_logic, template_info)
        structure["parameters"] = parameters
        
        return structure
    
    def _create_component(self, component_type: str, organism: str) -> Dict[str, Any]:
        """Create a component of the specified type."""
        component_id = str(uuid.uuid4())
        
        if component_type == "promoter":
            return {
                "id": component_id,
                "type": "promoter",
                "name": f"p{component_id[:8]}",
                "organism": organism,
                "properties": {
                    "strength": 1.0,
                    "type": "inducible"
                }
            }
        elif component_type == "gene":
            return {
                "id": component_id,
                "type": "gene",
                "name": f"gene_{component_id[:8]}",
                "organism": organism,
                "properties": {
                    "product": "protein",
                    "expression_level": 1.0
                }
            }
        elif component_type == "repressor":
            return {
                "id": component_id,
                "type": "protein",
                "name": f"rep_{component_id[:8]}",
                "organism": organism,
                "properties": {
                    "function": "repressor",
                    "binding_affinity": 1.0
                }
            }
        elif component_type == "activator":
            return {
                "id": component_id,
                "type": "protein",
                "name": f"act_{component_id[:8]}",
                "organism": organism,
                "properties": {
                    "function": "activator",
                    "binding_affinity": 1.0
                }
            }
        else:
            return {
                "id": component_id,
                "type": component_type,
                "name": f"{component_type}_{component_id[:8]}",
                "organism": organism,
                "properties": {}
            }
    
    def _generate_connections(self, parsed_logic: ParsedLogic, 
                            template_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate connections between components."""
        connections = []
        
        # Generate connections based on logic expression
        logic_expression = parsed_logic.logic_expression
        
        if "AND" in logic_expression:
            # Create AND connection
            connections.append({
                "type": "regulation",
                "source": "input1",
                "target": "output",
                "regulation_type": "activation",
                "strength": 1.0
            })
            connections.append({
                "type": "regulation", 
                "source": "input2",
                "target": "output",
                "regulation_type": "activation",
                "strength": 1.0
            })
        
        if "OR" in logic_expression:
            # Create OR connection
            connections.append({
                "type": "regulation",
                "source": "input1",
                "target": "output",
                "regulation_type": "activation",
                "strength": 1.0
            })
            connections.append({
                "type": "regulation",
                "source": "input2", 
                "target": "output",
                "regulation_type": "activation",
                "strength": 1.0
            })
        
        if "NOT" in logic_expression:
            # Create NOT connection
            connections.append({
                "type": "regulation",
                "source": "input",
                "target": "output",
                "regulation_type": "repression",
                "strength": 1.0
            })
        
        return connections
    
    def _generate_logic_gates(self, parsed_logic: ParsedLogic) -> List[Dict[str, Any]]:
        """Generate logic gates from parsed logic."""
        gates = []
        
        logic_expression = parsed_logic.logic_expression
        
        if "AND" in logic_expression:
            gates.append({
                "type": "AND",
                "inputs": ["input1", "input2"],
                "output": "output",
                "threshold": 0.5
            })
        
        if "OR" in logic_expression:
            gates.append({
                "type": "OR",
                "inputs": ["input1", "input2"],
                "output": "output",
                "threshold": 0.5
            })
        
        if "NOT" in logic_expression:
            gates.append({
                "type": "NOT",
                "inputs": ["input"],
                "output": "output",
                "threshold": 0.5
            })
        
        return gates
    
    def _generate_parameters(self, parsed_logic: ParsedLogic, 
                           template_info: Dict[str, Any]) -> Dict[str, float]:
        """Generate simulation parameters."""
        parameters = {
            "transcription_rate": 1.0,
            "translation_rate": 1.0,
            "degradation_rate": 0.1,
            "binding_affinity": 1.0
        }
        
        # Add temporal parameters if oscillatory
        if "OSCILLATOR" in parsed_logic.logic_expression:
            for constraint in parsed_logic.temporal_constraints:
                if constraint["type"] == "oscillator":
                    parameters["oscillation_period"] = constraint["period"]
                    parameters["oscillation_unit"] = constraint["unit"]
        
        return parameters
    
    def _enhance_with_ai(self, description: str, parsed_logic: ParsedLogic, 
                        circuit_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance parsing results using AI."""
        try:
            prompt = f"""
            Analyze this genetic circuit description and provide enhanced parsing:
            
            Description: {description}
            Extracted Logic: {parsed_logic.logic_expression}
            
            Please provide:
            1. Additional biological components that might be needed
            2. Refined parameter values
            3. Potential issues or improvements
            4. Alternative circuit designs
            
            Respond in JSON format.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            ai_response = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                ai_data = json.loads(ai_response)
                return {
                    "ai_enhancements": ai_data,
                    "ai_confidence": 0.8
                }
            except json.JSONDecodeError:
                return {
                    "ai_enhancements": {"raw_response": ai_response},
                    "ai_confidence": 0.5
                }
                
        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")
            return {
                "ai_enhancements": {},
                "ai_confidence": 0.0
            }
    
    def validate_parsing(self, result: Dict[str, Any]) -> List[str]:
        """Validate parsing results and return error messages."""
        errors = []
        
        # Check confidence
        if result.get("confidence", 0) < 0.3:
            errors.append("Low confidence in parsing results")
        
        # Check required fields
        if "parsed_logic" not in result:
            errors.append("Missing parsed logic")
        
        if "circuit_structure" not in result:
            errors.append("Missing circuit structure")
        
        # Check circuit structure
        structure = result.get("circuit_structure", {})
        if not structure.get("components"):
            errors.append("No components in circuit structure")
        
        if not structure.get("connections"):
            errors.append("No connections in circuit structure")
        
        return errors 