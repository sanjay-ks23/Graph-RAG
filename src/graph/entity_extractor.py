"""Entity extraction and relationship identification for graph construction"""

import spacy
import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class EntityExtractor:
    """Extract therapeutic entities and relationships from text"""
    
    def __init__(self, entity_types: List[str], relationship_types: List[str]):
        self.entity_types = entity_types
        self.relationship_types = relationship_types
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("Downloading spaCy model...")
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Therapeutic domain patterns
        self.therapy_patterns = self._init_therapy_patterns()
        self.indian_cultural_patterns = self._init_cultural_patterns()
    
    def _init_therapy_patterns(self) -> Dict[str, List[str]]:
        """Initialize therapeutic domain patterns"""
        return {
            'therapeutic_technique': [
                r'\b(cognitive behavioral therapy|CBT|mindfulness|meditation|'
                r'breathing exercises?|relaxation|visualization|'
                r'positive affirmations?|journaling|art therapy|play therapy|'
                r'exposure therapy|systematic desensitization)\b',
            ],
            'emotion': [
                r'\b(anxiety|fear|worry|stress|anger|sadness|depression|'
                r'happiness|joy|excitement|frustration|guilt|shame|'
                r'loneliness|confusion|overwhelm)\b',
            ],
            'behavior': [
                r'\b(avoidance|withdrawal|aggression|tantrums?|'
                r'self-harm|isolation|acting out|cooperation|'
                r'participation|engagement)\b',
            ],
            'cognitive_pattern': [
                r'\b(negative thoughts?|catastrophizing|overgeneralization|'
                r'black and white thinking|self-criticism|rumination|'
                r'positive thinking|growth mindset)\b',
            ],
            'coping_strategy': [
                r'\b(deep breathing|counting|grounding|distraction|'
                r'seeking support|problem solving|self-care|'
                r'physical activity|creative expression)\b',
            ],
            'developmental_stage': [
                r'\b(early childhood|preschool|elementary|middle school|'
                r'adolescen[ct]e?|teenager|toddler|infant)\b',
            ]
        }
    
    def _init_cultural_patterns(self) -> Dict[str, List[str]]:
        """Initialize Indian cultural context patterns"""
        return {
            'cultural_context': [
                r'\b(family values?|joint family|extended family|'
                r'respect for elders|academic pressure|'
                r'cultural expectations?|traditional values?|'
                r'festivals?|rituals?|community support)\b',
            ]
        }
    
    def extract_entities(self, text: str, 
                        min_confidence: float = 0.6) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        entities = []
        seen_entities = set()
        
        # Extract using patterns
        for entity_type, patterns in {**self.therapy_patterns, 
                                      **self.indian_cultural_patterns}.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group(0).lower()
                    
                    # Avoid duplicates
                    entity_key = (entity_text, entity_type)
                    if entity_key not in seen_entities:
                        entities.append({
                            'text': entity_text,
                            'type': entity_type,
                            'confidence': 0.8,  # Pattern-based confidence
                            'span': (match.start(), match.end())
                        })
                        seen_entities.add(entity_key)
        
        # Extract using NLP
        doc = self.nlp(text)
        
        # Extract noun phrases as potential entities
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            
            # Check if it's therapy-related
            if self._is_therapy_related(chunk_text):
                entity_type = self._classify_entity(chunk_text)
                entity_key = (chunk_text, entity_type)
                
                if entity_key not in seen_entities and len(chunk_text.split()) <= 4:
                    entities.append({
                        'text': chunk_text,
                        'type': entity_type,
                        'confidence': 0.6,
                        'span': (chunk.start_char, chunk.end_char)
                    })
                    seen_entities.add(entity_key)
        
        # Filter by confidence
        entities = [e for e in entities if e['confidence'] >= min_confidence]
        
        return entities
    
    def extract_relationships(self, text: str, 
                            entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        relationships = []
        doc = self.nlp(text)
        
        # Create entity lookup
        entity_spans = {(e['span'][0], e['span'][1]): e for e in entities}
        
        # Pattern-based relationship extraction
        relationship_patterns = {
            'treats': [
                r'(\w+(?:\s+\w+)*)\s+(?:helps?|treats?|reduces?|alleviates?)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:for|to address|to manage)\s+(\w+(?:\s+\w+)*)',
            ],
            'causes': [
                r'(\w+(?:\s+\w+)*)\s+(?:causes?|leads? to|results? in)\s+(\w+(?:\s+\w+)*)',
            ],
            'relates_to': [
                r'(\w+(?:\s+\w+)*)\s+(?:relates? to|associated with|connected to)\s+(\w+(?:\s+\w+)*)',
            ],
            'appropriate_for_age': [
                r'(\w+(?:\s+\w+)*)\s+(?:for|suitable for|appropriate for)\s+(\w+(?:\s+\w+)*)',
            ]
        }
        
        for rel_type, patterns in relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    source = match.group(1).lower().strip()
                    target = match.group(2).lower().strip()
                    
                    # Check if both are entities
                    source_entity = self._find_entity(source, entities)
                    target_entity = self._find_entity(target, entities)
                    
                    if source_entity and target_entity:
                        relationships.append({
                            'source': source_entity['text'],
                            'target': target_entity['text'],
                            'type': rel_type,
                            'confidence': 0.7
                        })
        
        # Dependency-based relationship extraction
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                head = token.head
                
                # Find entities
                source_entity = self._find_entity_by_token(token, entities)
                target_entity = self._find_entity_by_token(head, entities)
                
                if source_entity and target_entity and source_entity != target_entity:
                    rel_type = self._infer_relationship_type(
                        source_entity, target_entity, head.lemma_
                    )
                    if rel_type:
                        relationships.append({
                            'source': source_entity['text'],
                            'target': target_entity['text'],
                            'type': rel_type,
                            'confidence': 0.6
                        })
        
        # Remove duplicates
        unique_rels = []
        seen = set()
        for rel in relationships:
            key = (rel['source'], rel['target'], rel['type'])
            if key not in seen:
                unique_rels.append(rel)
                seen.add(key)
        
        return unique_rels
    
    def _is_therapy_related(self, text: str) -> bool:
        """Check if text is therapy-related"""
        therapy_keywords = [
            'therapy', 'therapeutic', 'treatment', 'counseling',
            'emotion', 'feeling', 'behavior', 'cognitive', 'mental',
            'psychological', 'coping', 'stress', 'anxiety', 'child'
        ]
        return any(keyword in text.lower() for keyword in therapy_keywords)
    
    def _classify_entity(self, text: str) -> str:
        """Classify entity type"""
        for entity_type, patterns in {**self.therapy_patterns, 
                                      **self.indian_cultural_patterns}.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return entity_type
        
        return 'therapeutic_technique'  # Default
    
    def _find_entity(self, text: str, entities: List[Dict]) -> Dict:
        """Find entity by text"""
        text = text.lower().strip()
        for entity in entities:
            if entity['text'] == text or text in entity['text'] or entity['text'] in text:
                return entity
        return None
    
    def _find_entity_by_token(self, token, entities: List[Dict]) -> Dict:
        """Find entity by spaCy token"""
        for entity in entities:
            if token.text.lower() in entity['text']:
                return entity
        return None
    
    def _infer_relationship_type(self, source: Dict, target: Dict, 
                                verb: str) -> str:
        """Infer relationship type from verb"""
        treat_verbs = {'help', 'treat', 'reduce', 'alleviate', 'manage'}
        cause_verbs = {'cause', 'lead', 'result', 'trigger'}
        
        if verb in treat_verbs:
            return 'treats'
        elif verb in cause_verbs:
            return 'causes'
        else:
            return 'relates_to'
