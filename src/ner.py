"""
Named Entity Recognition for French text extraction
Extracts PERSON and COMPANY entities from newspaper content
"""

import spacy
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import json
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class Entity(BaseModel):
    """Pydantic model for extracted entities"""
    page_num: int
    entity_text: str
    entity_type: str = Field(..., pattern="^(PERSON|COMPANY)$")
    language: str = Field(..., pattern="^(fr|ar|unknown)$")
    confidence: float = Field(..., ge=0.0, le=1.0)

class NERExtractor:
    """
    Extracts named entities (PERSON, COMPANY) from French text
    using spaCy NER model
    """
    
    def __init__(self, model_name: str = "fr_core_news_sm"):
        """
        Initialize NER extractor with spaCy model
        
        Args:
            model_name: Name of spaCy model to use
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"Model {model_name} not found. Please install with: "
                        f"python -m spacy download {model_name}")
            raise
    
    def extract_entities_from_pages(self, pages_data: List[Dict]) -> List[Entity]:
        """
        Extract entities from all pages
        
        Args:
            pages_data: List of page dictionaries
            
        Returns:
            List of extracted entities
        """
        all_entities = []
        
        for page in pages_data:
            page_num = page["page_num"]
            french_text = page["page_content_fr"]
            
            # Only process pages with substantial French text
            if len(french_text.strip()) > 10:
                page_entities = self._extract_from_page(page_num, french_text)
                all_entities.extend(page_entities)
                
                logger.info(f"Page {page_num}: extracted {len(page_entities)} entities")
        
        logger.info(f"Total entities extracted: {len(all_entities)}")
        return all_entities
    
    def _extract_from_page(self, page_num: int, text: str) -> List[Entity]:
        """
        Extract entities from a single page's French text
        
        Args:
            page_num: Page number
            text: French text content
            
        Returns:
            List of entities from this page
        """
        entities = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entity_type = self._map_entity_type(ent.label_)
                
                if entity_type:  # Only process PERSON and COMPANY
                    entity = Entity(
                        page_num=page_num,
                        entity_text=ent.text,
                        entity_type=entity_type,
                        language="fr",
                        confidence=0.8  # spaCy doesn't provide confidence, using default
                    )
                    entities.append(entity)
                    
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {str(e)}")
        
        return entities
    
    def _map_entity_type(self, spacy_label: str) -> str:
        """
        Map spaCy entity labels to our schema
        
        Args:
            spacy_label: spaCy entity label
            
        Returns:
            Mapped entity type or None if not relevant
        """
        mapping = {
            'PER': 'PERSON',
            'PERSON': 'PERSON',
            'ORG': 'COMPANY',
            'MISC': 'COMPANY'  # Map miscellaneous to COMPANY
        }
        return mapping.get(spacy_label)
    
    def save_entities_to_jsonl(self, entities: List[Entity], output_path: str):
        """
        Save entities to JSONL file
        
        Args:
            entities: List of Entity objects
            output_path: Output file path
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entity in entities:
                json_line = json.dumps(entity.dict(), ensure_ascii=False)
                f.write(json_line + '\n')
        
        logger.info(f"Saved {len(entities)} entities to {output_path}")

# Utility function for easy usage
def extract_entities_from_jsonl(input_path: str, output_path: str, model_name: str = "fr_core_news_sm"):
    """
    Convenience function to extract entities from JSONL file
    
    Args:
        input_path: Input JSONL file with pages data
        output_path: Output JSONL file for entities
        model_name: spaCy model name
    """
    # Load pages data
    pages_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            pages_data.append(json.loads(line))
    
    # Extract entities
    extractor = NERExtractor(model_name=model_name)
    entities = extractor.extract_entities_from_pages(pages_data)
    extractor.save_entities_to_jsonl(entities, output_path)
    
    return entities