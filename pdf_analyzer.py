#!/usr/bin/env python3
"""
Universal PDF Analysis Engine
Pure processing engine for extracting and analyzing PDF content
with customizable keywords and domain-specific organization.
"""

import os
import sys
import fitz  # PyMuPDF
import pandas as pd
import re
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import camelot  # For table extraction
# Note: tabula can be added as alternative if camelot fails
import logging
from collections import defaultdict
import textwrap
from typing import Dict, List, Tuple, Optional, Any
import json
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalPDFAnalyzer:
    """
    Pure PDF analysis engine with customizable keywords and domain focus.
    No user interaction - only processing and analysis.
    """
    
    def __init__(self, custom_keywords: List[str] = None, domain_name: str = "Technical"):
        """
        Initialize analyzer with custom settings
        
        Args:
            custom_keywords: List of keywords to focus analysis on
            domain_name: Domain/field name for contextual analysis
        """
        self.domain_name = domain_name
        self.custom_keywords = custom_keywords or []
        
        # Generic content classification patterns
        self.content_patterns = {
            'introduction': ['introduction', 'abstract', 'overview', 'background', 'summary'],
            'standards': ['standard', 'specification', 'guideline', 'regulation', 'policy'],
            'methodology': ['method', 'approach', 'procedure', 'process', 'technique'],
            'data_model': ['data', 'model', 'schema', 'structure', 'framework'],
            'examples': ['example', 'case study', 'illustration', 'demonstration', 'sample'],
            'figures': ['figure', 'diagram', 'illustration', 'graphic', 'chart'],
            'tables': ['table', 'data', 'results', 'comparison', 'analysis'],
            'applications': ['application', 'use case', 'implementation', 'deployment'],
            'conclusions': ['conclusion', 'summary', 'result', 'finding', 'outcome']
        }
        
        # Dynamic topic mapping based on user keywords
        self.topic_map = self._create_dynamic_topic_map()
        
        # Processing statistics
        self.stats = {
            'pages_processed': 0,
            'images_extracted': 0,
            'tables_extracted': 0,
            'keyword_matches': 0,
            'processing_time': 0
        }
    
    def _create_dynamic_topic_map(self) -> Dict[str, List[str]]:
        """Create topic mapping based on user's custom keywords"""
        if not self.custom_keywords:
            return {}
        
        topic_map = {}
        
        for keyword in self.custom_keywords:
            keyword_lower = keyword.lower()
            
            # Categorize keywords into logical groups
            if any(term in keyword_lower for term in ['standard', 'spec', 'guideline', 'regulation']):
                topic_map.setdefault('Standards & Specifications', []).append(keyword)
            elif any(term in keyword_lower for term in ['method', 'process', 'procedure', 'technique']):
                topic_map.setdefault('Methods & Procedures', []).append(keyword)
            elif any(term in keyword_lower for term in ['data', 'model', 'schema', 'structure']):
                topic_map.setdefault('Data & Models', []).append(keyword)
            elif any(term in keyword_lower for term in ['test', 'validation', 'verification', 'quality']):
                topic_map.setdefault('Testing & Validation', []).append(keyword)
            elif any(term in keyword_lower for term in ['design', 'development', 'implementation']):
                topic_map.setdefault('Design & Development', []).append(keyword)
            else:
                topic_map.setdefault(f'{self.domain_name} Concepts', []).append(keyword)
        
        return topic_map
    
    def extract_content_from_pdf(self, pdf_path: str) -> Dict[int, Dict[str, Any]]:
        """
        Extract and organize all content from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dict: Organized content by page number
        """
        logger.info(f"Starting content extraction from: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            organized_content = {}
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Process page content
                cleaned_text = self._clean_text(text)
                paragraphs = self._extract_meaningful_paragraphs(cleaned_text)
                page_images = self._extract_page_images(page, doc, page_num)
                
                # Analyze content
                content_analysis = {
                    'raw_text': text,
                    'cleaned_text': cleaned_text,
                    'paragraphs': paragraphs,
                    'images': page_images,
                    'content_type': self._classify_content_type(cleaned_text),
                    'key_topics': self._extract_key_topics(cleaned_text),
                    'figure_references': self._find_figure_references(cleaned_text),
                    'table_references': self._find_table_references(cleaned_text),
                    'keyword_matches': self._find_keyword_matches(cleaned_text),
                    'importance_score': self._calculate_importance_score(cleaned_text),
                    'has_meaningful_content': len(paragraphs) > 0,
                    'word_count': len(cleaned_text.split()),
                    'keyword_density': self._calculate_keyword_density(cleaned_text)
                }
                
                organized_content[page_num + 1] = content_analysis
                self.stats['pages_processed'] += 1
                
                logger.debug(f"Processed page {page_num + 1}: {len(paragraphs)} paragraphs, {len(page_images)} images")
            
            doc.close()
            logger.info(f"Content extraction complete: {len(organized_content)} pages processed")
            return organized_content
            
        except Exception as e:
            logger.error(f"Error during content extraction: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common headers/footers
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'http://[^\s]+', '', text)
        text = re.sub(r'www\.[^\s]+', '', text)
        
        # Fix common OCR issues
        text = re.sub(r'(\w)\s+([.,;:])', r'\1\2', text)
        text = re.sub(r'([A-Z])\s+([A-Z])\s+([A-Z])', r'\1\2\3', text)
        
        return text.strip()
    
    def _extract_meaningful_paragraphs(self, text: str) -> List[str]:
        """Extract meaningful paragraphs, filtering out technical clutter"""
        paragraphs = re.split(r'\n\n+|(?<=[.!?])\s+(?=[A-Z])', text)
        
        meaningful_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            
            # Skip technical schema and clutter
            if self._is_technical_clutter(para):
                continue
                
            # Filter based on content quality
            if (len(para) > 100 and 
                len(para.split()) > 15 and 
                not para.isupper() and
                not re.match(r'^[\d\s\W]+$', para)):
                meaningful_paragraphs.append(para)
        
        return meaningful_paragraphs[:3]  # Top 3 paragraphs per page
    
    def _extract_page_images(self, page, doc, page_num: int) -> List[Dict[str, Any]]:
        """Extract images from a page with context analysis"""
        page_images = []
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                    
                    # Only keep meaningful images
                    if len(img_data) > 2000:  # Skip tiny images
                        page_images.append({
                            'img_index': img_index,
                            'img_data': img_data,
                            'size': len(img_data),
                            'description': self._analyze_image_context(page.get_text()),
                            'page_number': page_num + 1
                        })
                        self.stats['images_extracted'] += 1
                
                pix = None
                
            except Exception as e:
                logger.warning(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
                continue
        
        return page_images
    
    def _find_keyword_matches(self, text: str) -> List[Dict[str, Any]]:
        """Find matches for custom keywords in text"""
        if not self.custom_keywords:
            return []
        
        text_lower = text.lower()
        matches = []
        
        for keyword in self.custom_keywords:
            # Count exact matches and partial matches
            exact_count = text_lower.count(keyword.lower())
            
            # Also check for word boundaries to avoid partial word matches
            word_pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            word_matches = len(re.findall(word_pattern, text_lower))
            
            if exact_count > 0 or word_matches > 0:
                matches.append({
                    'keyword': keyword,
                    'exact_matches': exact_count,
                    'word_matches': word_matches,
                    'total_count': max(exact_count, word_matches)
                })
                self.stats['keyword_matches'] += max(exact_count, word_matches)
        
        # Sort by relevance
        matches.sort(key=lambda x: x['total_count'], reverse=True)
        return matches
    
    def _classify_content_type(self, text: str) -> str:
        """Classify content type based on patterns and keywords"""
        text_lower = text.lower()
        scores = {}
        
        # Score based on generic patterns
        for content_type, keywords in self.content_patterns.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            if score > 0:
                scores[content_type] = score
        
        # Boost score for custom keyword presence
        if self.custom_keywords:
            custom_score = sum(text_lower.count(keyword.lower()) for keyword in self.custom_keywords)
            if custom_score > 0:
                scores['custom_content'] = custom_score
        
        return max(scores, key=scores.get) if scores else 'general_content'
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics based on custom keywords and domain"""
        text_lower = text.lower()
        topics = []
        
        # Use dynamic topic mapping
        for topic, keywords in self.topic_map.items():
            relevance = sum(text_lower.count(keyword.lower()) for keyword in keywords)
            if relevance >= 1:
                topics.append(topic)
        
        # Fallback to individual keywords if no topics found
        if not topics and self.custom_keywords:
            for keyword in self.custom_keywords:
                if keyword.lower() in text_lower:
                    topics.append(keyword.title())
        
        return topics[:5]  # Limit to top 5 topics
    
    def _calculate_importance_score(self, text: str) -> float:
        """Calculate content importance based on multiple factors"""
        text_lower = text.lower()
        score = 0.0
        
        # Custom keyword weight
        if self.custom_keywords:
            keyword_score = sum(text_lower.count(keyword.lower()) for keyword in self.custom_keywords)
            score += keyword_score * 3.0
        
        # Structural elements weight
        if 'figure' in text_lower or 'table' in text_lower:
            score += 2.0
        if 'example' in text_lower or 'case study' in text_lower:
            score += 1.5
        
        # Length factor
        word_count = len(text.split())
        if word_count > 200:
            score += 1.0
        elif word_count > 100:
            score += 0.5
        
        # Domain-specific terms
        domain_terms = ['analysis', 'method', 'result', 'conclusion', 'approach']
        domain_score = sum(text_lower.count(term) for term in domain_terms)
        score += domain_score * 0.5
        
        return score
    
    def _calculate_keyword_density(self, text: str) -> float:
        """Calculate keyword density as percentage of total words"""
        if not self.custom_keywords or not text:
            return 0.0
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        keyword_count = sum(text.lower().count(keyword.lower()) for keyword in self.custom_keywords)
        return (keyword_count / total_words) * 100
    
    def _is_technical_clutter(self, text: str) -> bool:
        """Identify technical clutter that should be filtered out"""
        text_lower = text.lower()
        
        # Technical schema indicators
        clutter_indicators = [
            'entity', 'end_entity', 'subtype of', 'supertype of',
            'derive', 'inverse', 'where wr', 'sizeof', 'query',
            'typeof', 'end_type', 'enumeration of', 'select (',
            'set [', '*)', '(*', 'logical :=', 'function', 'procedure'
        ]
        
        clutter_count = sum(1 for indicator in clutter_indicators if indicator in text_lower)
        
        # Check for code-like patterns
        return (clutter_count >= 2 or 
                text.count(':') > 5 or 
                text.count(';') > 3 or 
                'END_ENTITY;' in text or
                text.count('(') > 5)
    
    def _find_figure_references(self, text: str) -> List[str]:
        """Find figure references in text"""
        patterns = [
            r'[Ff]igure\s*(\d+)',
            r'[Ff]ig\.\s*(\d+)',
            r'[Ss]ee\s+[Ff]igure\s*(\d+)'
        ]
        
        references = []
        for pattern in patterns:
            references.extend(re.findall(pattern, text))
        
        return list(set(references))
    
    def _find_table_references(self, text: str) -> List[str]:
        """Find table references in text"""
        patterns = [
            r'[Tt]able\s*(\d+)',
            r'[Ss]ee\s+[Tt]able\s*(\d+)'
        ]
        
        references = []
        for pattern in patterns:
            references.extend(re.findall(pattern, text))
        
        return list(set(references))
    
    def _analyze_image_context(self, surrounding_text: str) -> str:
        """Analyze image context based on surrounding text"""
        text_lower = surrounding_text.lower()
        
        # Check custom keywords first
        if self.custom_keywords:
            for keyword in self.custom_keywords:
                if keyword.lower() in text_lower:
                    return f"{self.domain_name} - {keyword.title()}"
        
        # Generic context mapping
        context_map = {
            'Technical Diagram': ['diagram', 'schematic', 'flow', 'process'],
            'Data Visualization': ['chart', 'graph', 'plot', 'visualization'],
            'Example': ['example', 'illustration', 'demonstration'],
            'Design': ['design', 'architecture', 'structure', 'layout'],
            'Analysis': ['analysis', 'result', 'comparison', 'evaluation']
        }
        
        for description, keywords in context_map.items():
            if any(keyword in text_lower for keyword in keywords):
                return f"{self.domain_name} - {description}"
        
        return f"{self.domain_name} Diagram"
    
    def extract_tables(self, pdf_path: str, organized_content: Dict[int, Dict]) -> List[Dict[str, Any]]:
        """Extract tables from PDF with context analysis"""
        logger.info("Starting table extraction...")
        
        tables_with_context = []
        
        try:
            # Use Camelot for table extraction
            camelot_tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            
            for i, table in enumerate(camelot_tables):
                df = table.df
                
                if not df.empty and df.shape[0] > 1 and df.shape[1] > 1:
                    # Calculate relevance score
                    relevance_score = self._calculate_table_relevance(df)
                    
                    if relevance_score > 0:
                        page_num = table.parsing_report['page']
                        
                        # Get context from page content
                        context = ""
                        if page_num in organized_content:
                            page_content = organized_content[page_num]
                            if page_content['paragraphs']:
                                context = page_content['paragraphs'][0][:300]
                        
                        tables_with_context.append({
                            'table_id': i,
                            'page': page_num,
                            'dataframe': df,
                            'description': self._generate_table_description(df),
                            'context': context,
                            'relevance_score': relevance_score,
                            'extraction_method': 'camelot'
                        })
                        
                        self.stats['tables_extracted'] += 1
            
            # Sort by relevance
            tables_with_context.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            logger.info(f"Table extraction complete: {len(tables_with_context)} tables found")
            return tables_with_context[:10]  # Top 10 most relevant tables
            
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
            return []
    
    def _calculate_table_relevance(self, df: pd.DataFrame) -> float:
        """Calculate table relevance based on content and keywords"""
        table_text = ' '.join(df.values.flatten().astype(str)).lower()
        score = 0.0
        
        # Custom keyword relevance
        if self.custom_keywords:
            for keyword in self.custom_keywords:
                score += table_text.count(keyword.lower()) * 2.0
        
        # Generic relevance indicators
        relevance_terms = ['result', 'analysis', 'comparison', 'data', 'value', 'measurement']
        for term in relevance_terms:
            score += table_text.count(term) * 1.0
        
        # Size factor (prefer meaningful tables)
        if df.shape[0] > 3 and df.shape[1] > 2:
            score += 1.0
        
        return score
    
    def _generate_table_description(self, df: pd.DataFrame) -> str:
        """Generate descriptive title for table"""
        columns = [str(col).lower() for col in df.columns]
        content = ' '.join(df.values.flatten().astype(str)).lower()
        
        descriptions = []
        
        # Check for custom keywords
        if self.custom_keywords:
            for keyword in self.custom_keywords:
                if (keyword.lower() in content or 
                    any(keyword.lower() in col for col in columns)):
                    descriptions.append(f"{keyword.title()} Data")
        
        # Generic patterns
        if any('result' in col for col in columns) or 'result' in content:
            descriptions.append("Results")
        if any('comparison' in col for col in columns) or 'comparison' in content:
            descriptions.append("Comparison")
        if any('analysis' in col for col in columns) or 'analysis' in content:
            descriptions.append("Analysis")
        
        if descriptions:
            return f"{self.domain_name} - {' & '.join(descriptions[:3])}"
        else:
            return f"{self.domain_name} Data Table ({df.shape[0]}x{df.shape[1]})"
    
    def create_summary_pdf(self, organized_content: Dict[int, Dict], 
                          tables: List[Dict], pdf_path: str) -> str:
        """Create comprehensive summary PDF"""
        logger.info("Creating summary PDF...")
        
        # Setup paths
        output_dir = os.path.dirname(pdf_path)
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        temp_dir = os.path.join(output_dir, f"temp_{base_name}")
        summary_path = os.path.join(output_dir, f"{base_name}_analysis_summary.pdf")
        
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Create PDF document
            doc = SimpleDocTemplate(
                summary_path, 
                pagesize=letter,
                leftMargin=0.75*inch, 
                rightMargin=0.75*inch,
                topMargin=1*inch, 
                bottomMargin=1*inch
            )
            
            # Create content
            story = []
            styles = getSampleStyleSheet()
            
            # Add title and introduction
            self._add_title_section(story, styles)
            self._add_executive_summary(story, organized_content, styles)
            story.append(PageBreak())
            
            # Add content sections
            self._add_content_sections(story, organized_content, temp_dir, styles)
            
            # Add tables section
            if tables:
                self._add_tables_section(story, tables, styles)
            
            # Build PDF
            doc.build(story)
            
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            logger.info(f"Summary PDF created: {summary_path}")
            return summary_path
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            logger.error(f"Error creating summary PDF: {e}")
            raise
    
    def _add_title_section(self, story: List, styles):
        """Add title section to PDF"""
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1,
            fontName='Helvetica-Bold'
        )
        
        story.append(Paragraph(f"{self.domain_name} Document Analysis", title_style))
        story.append(Paragraph("Comprehensive Content Summary with Custom Keywords", styles['Normal']))
        story.append(Spacer(1, 30))
    
    def _add_executive_summary(self, story: List, organized_content: Dict, styles):
        """Add executive summary section"""
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        # Calculate statistics
        total_pages = len(organized_content)
        meaningful_pages = sum(1 for page in organized_content.values() if page['has_meaningful_content'])
        total_keyword_matches = sum(len(page['keyword_matches']) for page in organized_content.values())
        content_types = set(page['content_type'] for page in organized_content.values())
        
        # Summary text
        summary_text = f"""
        This {self.domain_name.lower()} document contains {total_pages} pages with {meaningful_pages} pages 
        containing meaningful content. The analysis identified {total_keyword_matches} keyword matches 
        and {len(content_types)} distinct content types.
        """
        
        story.append(Paragraph(summary_text.strip(), styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Keyword analysis
        if self.custom_keywords:
            story.append(Paragraph("Keyword Analysis", styles['Heading3']))
            story.append(Spacer(1, 8))
            
            # Aggregate keyword statistics
            keyword_stats = defaultdict(int)
            for page in organized_content.values():
                for match in page['keyword_matches']:
                    keyword_stats[match['keyword']] += match['total_count']
            
            # Display top keywords
            top_keywords = sorted(keyword_stats.items(), key=lambda x: x[1], reverse=True)[:10]
            for keyword, count in top_keywords:
                story.append(Paragraph(f"• {keyword}: {count} mentions", styles['Normal']))
            
            story.append(Spacer(1, 20))
        
        # Content type distribution
        story.append(Paragraph("Content Distribution", styles['Heading3']))
        story.append(Spacer(1, 8))
        
        type_counts = defaultdict(int)
        for page in organized_content.values():
            type_counts[page['content_type']] += 1
        
        for content_type, count in sorted(type_counts.items()):
            formatted_type = content_type.replace('_', ' ').title()
            story.append(Paragraph(f"• {formatted_type}: {count} pages", styles['Normal']))
        
        story.append(Spacer(1, 20))
    
    def _add_content_sections(self, story: List, organized_content: Dict, temp_dir: str, styles):
        """Add main content sections organized by topics"""
        # Group content by topics
        topic_groups = defaultdict(list)
        
        # Sort content by importance
        sorted_content = sorted(
            organized_content.items(),
            key=lambda x: (x[1]['importance_score'], len(x[1]['keyword_matches'])),
            reverse=True
        )
        
        # Group by topics
        for page_num, page_data in sorted_content:
            if page_data['key_topics']:
                for topic in page_data['key_topics']:
                    topic_groups[topic].append((page_num, page_data))
            else:
                topic_groups['General Content'].append((page_num, page_data))
        
        # Create sections
        section_style = ParagraphStyle(
            'CustomSection',
            parent=styles['Heading2'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.darkgreen,
            spaceBefore=25,
            fontName='Helvetica-Bold'
        )
        
        for topic, pages in topic_groups.items():
            if pages:
                story.append(Paragraph(topic, section_style))
                story.append(Spacer(1, 15))
                
                for page_num, page_data in pages[:5]:  # Top 5 pages per topic
                    self._add_page_content(story, page_num, page_data, temp_dir, styles)
                
                story.append(PageBreak())
    
    def _add_page_content(self, story: List, page_num: int, page_data: Dict, temp_dir: str, styles):
        """Add individual page content to PDF"""
        subsection_style = ParagraphStyle(
            'CustomSubsection',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        content_style = ParagraphStyle(
            'ContentStyle',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            spaceBefore=4,
            leftIndent=12,
            rightIndent=12
        )
        
        # Page header
        keyword_info = ""
        if page_data['keyword_matches']:
            top_keywords = [match['keyword'] for match in page_data['keyword_matches'][:3]]
            keyword_info = f" - Keywords: {', '.join(top_keywords)}"
        
        header = f"Page {page_num}{keyword_info}"
        story.append(Paragraph(header, subsection_style))
        story.append(Spacer(1, 8))
        
        # Add content paragraphs
        for paragraph in page_data['paragraphs'][:2]:
            if len(paragraph) > 100:
                clean_paragraph = self._clean_paragraph_for_display(paragraph)
                if clean_paragraph:
                    highlighted_paragraph = self._highlight_keywords_in_text(clean_paragraph)
                    story.append(Paragraph(highlighted_paragraph, content_style))
                    story.append(Spacer(1, 6))
        
        # Add images
        for img_data in page_data['images']:
            if img_data['size'] > 5000:  # Only meaningful images
                self._add_image_to_story(story, img_data, page_num, temp_dir, styles)
        
        story.append(Spacer(1, 15))
    
    def _clean_paragraph_for_display(self, paragraph: str) -> Optional[str]:
        """Clean paragraph for display in PDF"""
        if self._is_technical_clutter(paragraph):
            return None
        
        # Remove technical artifacts
        cleaned = re.sub(r'\([*][)].*?[(][*]', '', paragraph, flags=re.DOTALL)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned if len(cleaned) > 50 else None
    
    def _highlight_keywords_in_text(self, text: str) -> str:
        """Highlight custom keywords in text for PDF display"""
        if not self.custom_keywords:
            return text
        
        highlighted_text = text
        
        # Sort keywords by length (longest first) to avoid partial replacements
        sorted_keywords = sorted(self.custom_keywords, key=len, reverse=True)
        
        for keyword in sorted_keywords:
            # Use word boundaries to avoid partial word matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            highlighted_text = re.sub(pattern, f"<b>{keyword}</b>", highlighted_text, flags=re.IGNORECASE)
        
        return highlighted_text
    
    def _add_image_to_story(self, story: List, img_data: Dict, page_num: int, temp_dir: str, styles):
        """Add image to PDF story"""
        img_filename = f"page{page_num}_img{img_data['img_index']}.png"
        temp_img_path = os.path.join(temp_dir, img_filename)
        
        try:
            # Save image temporarily
            with open(temp_img_path, "wb") as img_file:
                img_file.write(img_data['img_data'])
            
            # Add image description
            story.append(Paragraph(f"<b>{img_data['description']}</b>", styles['Normal']))
            story.append(Spacer(1, 4))
            
            # Add image with proper sizing
            img = Image(temp_img_path)
            max_width, max_height = 4.5*inch, 3*inch
            
            # Smart resizing
            if img.drawWidth > max_width:
                img.drawHeight = img.drawHeight * max_width / img.drawWidth
                img.drawWidth = max_width
            if img.drawHeight > max_height:
                img.drawWidth = img.drawWidth * max_height / img.drawHeight
                img.drawHeight = max_height
            
            story.append(img)
            story.append(Spacer(1, 12))
            
        except Exception as e:
            story.append(Paragraph(f"<i>Image unavailable: {e}</i>", styles['Normal']))
            story.append(Spacer(1, 8))
    
    def _add_tables_section(self, story: List, tables: List[Dict], styles):
        """Add tables section to PDF"""
        section_style = ParagraphStyle(
            'CustomSection',
            parent=styles['Heading2'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.darkgreen,
            spaceBefore=25,
            fontName='Helvetica-Bold'
        )
        
        subsection_style = ParagraphStyle(
            'CustomSubsection',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        story.append(Paragraph("Data Tables and Analysis", section_style))
        story.append(Spacer(1, 15))
        
        for i, table in enumerate(tables, 1):
            story.append(Paragraph(f"Table {i}: {table['description']}", subsection_style))
            story.append(Spacer(1, 8))
            
            # Add context if available
            if table.get('context'):
                context_text = table['context'][:200] + "..." if len(table['context']) > 200 else table['context']
                story.append(Paragraph(f"<i>Context: {context_text}</i>", styles['Normal']))
                story.append(Spacer(1, 8))
            
            # Add the table
            self._add_formatted_table(story, table, styles)
            story.append(Spacer(1, 20))
    
    def _add_formatted_table(self, story: List, table_data: Dict, styles):
        """Add formatted table to PDF"""
        df = table_data['dataframe']
        
        if df.empty:
            story.append(Paragraph("<i>Table contains no data</i>", styles['Normal']))
            return
        
        # Clean and prepare data
        df_clean = df.fillna('')
        
        # Limit table size for readability
        max_rows, max_cols = 15, 6
        truncated_rows = truncated_cols = False
        
        if df_clean.shape[0] > max_rows:
            df_display = df_clean.head(max_rows)
            truncated_rows = True
        else:
            df_display = df_clean
        
        if df_display.shape[1] > max_cols:
            df_display = df_display.iloc[:, :max_cols]
            truncated_cols = True
        
        # Create table data
        table_data_list = [df_display.columns.tolist()] + df_display.values.tolist()
        
        # Clean cell content
        for i, row in enumerate(table_data_list):
            for j, cell in enumerate(row):
                cell_str = str(cell).strip()
                if len(cell_str) > 40:
                    cell_str = cell_str[:37] + "..."
                table_data_list[i][j] = cell_str
        
        # Create table with styling
        table_obj = Table(table_data_list)
        table_obj.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8)
        ]))
        
        story.append(table_obj)
        
        # Add truncation notes
        if truncated_rows or truncated_cols:
            story.append(Spacer(1, 4))
            notes = []
            if truncated_rows:
                notes.append(f"showing first {max_rows} of {df_clean.shape[0]} rows")
            if truncated_cols:
                notes.append(f"showing first {max_cols} of {df_clean.shape[1]} columns")
            
            story.append(Paragraph(f"<i>Note: {', '.join(notes)}</i>", styles['Normal']))
    
    def generate_analysis_report(self, organized_content: Dict[int, Dict], 
                               tables: List[Dict], pdf_path: str) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        
        # Calculate final statistics
        total_pages = len(organized_content)
        meaningful_pages = sum(1 for page in organized_content.values() if page['has_meaningful_content'])
        total_images = sum(len(page['images']) for page in organized_content.values())
        total_keyword_matches = sum(len(page['keyword_matches']) for page in organized_content.values())
        content_types = set(page['content_type'] for page in organized_content.values())
        
        # Calculate keyword statistics
        keyword_stats = defaultdict(int)
        for page in organized_content.values():
            for match in page['keyword_matches']:
                keyword_stats[match['keyword']] += match['total_count']
        
        # Find most important pages
        important_pages = sorted(
            organized_content.items(),
            key=lambda x: x[1]['importance_score'],
            reverse=True
        )[:5]
        
        # Create comprehensive report
        report = {
            'input_file': pdf_path,
            'domain': self.domain_name,
            'custom_keywords': self.custom_keywords,
            'processing_stats': {
                'total_pages': total_pages,
                'meaningful_pages': meaningful_pages,
                'processing_success_rate': (meaningful_pages / total_pages * 100) if total_pages > 0 else 0
            },
            'content_analysis': {
                'content_types': list(content_types),
                'keyword_matches': total_keyword_matches,
                'keyword_statistics': dict(keyword_stats),
                'top_keywords': sorted(keyword_stats.items(), key=lambda x: x[1], reverse=True)[:10]
            },
            'extracted_elements': {
                'images': total_images,
                'tables': len(tables),
                'figures_referenced': sum(len(page['figure_references']) for page in organized_content.values()),
                'tables_referenced': sum(len(page['table_references']) for page in organized_content.values())
            },
            'quality_metrics': {
                'average_importance_score': sum(page['importance_score'] for page in organized_content.values()) / total_pages if total_pages > 0 else 0,
                'keyword_density': sum(page['keyword_density'] for page in organized_content.values()) / total_pages if total_pages > 0 else 0,
                'content_richness': meaningful_pages / total_pages if total_pages > 0 else 0
            },
            'important_pages': [
                {
                    'page_number': page_num,
                    'importance_score': page_data['importance_score'],
                    'key_topics': page_data['key_topics'],
                    'keyword_matches': len(page_data['keyword_matches'])
                }
                for page_num, page_data in important_pages
            ]
        }
        
        return report
    
    def process_pdf(self, pdf_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Main processing function - orchestrates the entire analysis pipeline
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Optional output directory (defaults to PDF directory)
            
        Returns:
            Dict: Comprehensive analysis results
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Starting PDF analysis: {os.path.basename(pdf_path)}")
            
            # Extract content
            organized_content = self.extract_content_from_pdf(pdf_path)
            
            # Extract tables
            tables = self.extract_tables(pdf_path, organized_content)
            
            # Create summary PDF
            summary_pdf_path = self.create_summary_pdf(organized_content, tables, pdf_path)
            
            # Generate analysis report
            analysis_report = self.generate_analysis_report(organized_content, tables, pdf_path)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.stats['processing_time'] = processing_time
            
            # Create final result
            result = {
                'input_file': pdf_path,
                'summary_pdf': summary_pdf_path,
                'domain': self.domain_name,
                'custom_keywords': self.custom_keywords,
                'total_pages': len(organized_content),
                'meaningful_pages': sum(1 for page in organized_content.values() if page['has_meaningful_content']),
                'extracted_figures': sum(len(page['images']) for page in organized_content.values()),
                'extracted_tables': len(tables),
                'content_types': list(set(page['content_type'] for page in organized_content.values())),
                'keyword_matches': sum(len(page['keyword_matches']) for page in organized_content.values()),
                'processing_time': processing_time,
                'processing_status': 'success',
                'analysis_report': analysis_report,
                'statistics': self.stats
            }
            
            logger.info(f"PDF analysis completed successfully in {processing_time:.2f} seconds")
            logger.info(f"Summary PDF created: {os.path.basename(summary_pdf_path)}")
            
            return result
            
        except Exception as e:
            logger.error(f"PDF analysis failed: {e}")
            raise
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analysis statistics"""
        return {
            'analyzer_info': {
                'domain': self.domain_name,
                'custom_keywords_count': len(self.custom_keywords),
                'topic_categories': len(self.topic_map)
            },
            'processing_stats': self.stats,
            'capabilities': {
                'content_extraction': True,
                'keyword_analysis': bool(self.custom_keywords),
                'table_extraction': True,
                'image_extraction': True,
                'pdf_generation': True
            }
        }


# Utility functions for standalone use
def analyze_pdf_with_keywords(pdf_path: str, keywords: List[str], 
                            domain: str = "Technical") -> Dict[str, Any]:
    """
    Convenience function for quick PDF analysis
    
    Args:
        pdf_path: Path to PDF file
        keywords: List of keywords to focus on
        domain: Domain name for analysis
        
    Returns:
        Dict: Analysis results
    """
    analyzer = UniversalPDFAnalyzer(keywords, domain)
    return analyzer.process_pdf(pdf_path)


def batch_analyze_pdfs(pdf_paths: List[str], keywords: List[str], 
                      domain: str = "Technical") -> List[Dict[str, Any]]:
    """
    Analyze multiple PDFs with the same keywords
    
    Args:
        pdf_paths: List of PDF file paths
        keywords: List of keywords to focus on
        domain: Domain name for analysis
        
    Returns:
        List[Dict]: Analysis results for each PDF
    """
    analyzer = UniversalPDFAnalyzer(keywords, domain)
    results = []
    
    for pdf_path in pdf_paths:
        try:
            result = analyzer.process_pdf(pdf_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            results.append({
                'input_file': pdf_path,
                'processing_status': 'failed',
                'error': str(e)
            })
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Universal PDF Analyzer - Analysis Engine")
    print("This module should be imported and used by main.py")
    print("For standalone testing, use the analyze_pdf_with_keywords function")