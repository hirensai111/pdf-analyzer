#!/usr/bin/env python3
"""
Universal PDF Content Extractor - Main Interface
Handles user input, validation, and orchestrates the PDF analysis process.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class PDFExtractorInterface:
    """Main interface for the Universal PDF Content Extractor"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
        self.min_keywords = 0
        self.max_keywords = 50
        
    def display_welcome(self):
        """Display welcome message and tool description"""
        print("🎯" + "="*70)
        print("    UNIVERSAL PDF CONTENT EXTRACTOR")
        print("="*70)
        print("📄 Extract and organize content from any technical PDF")
        print("🔍 Use your custom keywords for targeted analysis")
        print("🎯 Works across all domains: Engineering, Medical, Legal, etc.")
        print("="*70)
    
    def get_pdf_path(self, provided_path: Optional[str] = None) -> str:
        """
        Get and validate PDF file path
        
        Args:
            provided_path: Optional path provided via command line
            
        Returns:
            str: Valid PDF file path
        """
        if provided_path:
            pdf_path = provided_path.strip('\'"')
        else:
            print("\n📂 PDF File Selection")
            print("-" * 30)
            pdf_path = input("Enter PDF file path: ").strip('\'"')
        
        # Normalize path
        pdf_path = os.path.normpath(pdf_path)
        
        # Enhanced file finding logic
        if not os.path.exists(pdf_path):
            found_path = self._find_pdf_file(pdf_path)
            if found_path:
                pdf_path = found_path
                print(f"✅ Found file: {os.path.basename(pdf_path)}")
            else:
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Validate file format
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError("File must be a PDF (.pdf extension)")
        
        # Check file size (warn if very large)
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
        if file_size > 100:
            print(f"⚠️  Large file detected: {file_size:.1f} MB - processing may take longer")
        
        return pdf_path
    
    def _find_pdf_file(self, pdf_path: str) -> Optional[str]:
        """
        Try to find PDF file in common locations
        
        Args:
            pdf_path: Original path provided by user
            
        Returns:
            Optional[str]: Found file path or None
        """
        possible_paths = [
            pdf_path,
            os.path.join(os.getcwd(), pdf_path),
            os.path.join(os.getcwd(), os.path.basename(pdf_path)),
            # Remove potential shell escape characters
            pdf_path.replace('\\(', '(').replace('\\)', ')'),
            pdf_path.replace('\\ ', ' ')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Show available PDF files in current directory
        current_dir = os.getcwd()
        pdf_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.pdf')]
        
        if pdf_files:
            print(f"\n📁 Available PDF files in current directory:")
            for i, file in enumerate(pdf_files[:10], 1):
                print(f"   {i}. {file}")
            
            if len(pdf_files) > 10:
                print(f"   ... and {len(pdf_files) - 10} more files")
        
        return None
    
    def get_domain_name(self, provided_domain: Optional[str] = None) -> str:
        """
        Get domain/field name from user
        
        Args:
            provided_domain: Optional domain provided via command line
            
        Returns:
            str: Domain name
        """
        if provided_domain:
            return provided_domain.strip()
        
        print("\n🎯 Domain/Field Selection")
        print("-" * 30)
        print("Examples: Engineering, Medical, Legal, Data Science, Finance, etc.")
        
        domain = input("Enter your domain/field (or press Enter for 'Technical'): ").strip()
        return domain if domain else "Technical"
    
    def get_keywords_interactive(self) -> List[str]:
        """
        Get keywords from user interactively
        
        Returns:
            List[str]: List of keywords
        """
        print("\n🔍 Custom Keywords Entry")
        print("-" * 30)
        print("Enter keywords that are important for your analysis.")
        print("These will be used to identify and highlight relevant content.")
        print("Examples: 'machine learning', 'neural networks', 'data visualization'")
        print("\nInstructions:")
        print("• Enter one keyword per line")
        print("• Keywords can be single words or phrases")
        print("• Press Enter on empty line when done")
        print("• Type 'skip' to use generic analysis")
        print()
        
        keywords = []
        keyword_count = 0
        
        while keyword_count < self.max_keywords:
            prompt = f"Keyword {keyword_count + 1}: " if keyword_count < 9 else f"Keyword {keyword_count + 1}: "
            keyword = input(prompt).strip()
            
            if not keyword:
                break
            
            if keyword.lower() == 'skip':
                keywords = []
                print("⏭️  Skipping custom keywords - will use generic analysis")
                break
            
            # Basic validation
            if len(keyword) < 2:
                print("⚠️  Keyword too short (minimum 2 characters)")
                continue
            
            if keyword in keywords:
                print("⚠️  Keyword already added")
                continue
            
            keywords.append(keyword)
            keyword_count += 1
            
            # Show progress
            if keyword_count % 5 == 0:
                print(f"📊 Added {keyword_count} keywords so far...")
        
        return keywords
    
    def parse_keywords_from_string(self, keywords_string: str) -> List[str]:
        """
        Parse keywords from comma-separated string
        
        Args:
            keywords_string: Comma-separated keywords
            
        Returns:
            List[str]: List of cleaned keywords
        """
        keywords = []
        
        for keyword in keywords_string.split(','):
            keyword = keyword.strip()
            if keyword and len(keyword) >= 2:
                keywords.append(keyword)
        
        return keywords[:self.max_keywords]  # Limit to max
    
    def confirm_settings(self, pdf_path: str, domain: str, keywords: List[str]) -> bool:
        """
        Show settings summary and get user confirmation
        
        Args:
            pdf_path: Path to PDF file
            domain: Domain name
            keywords: List of keywords
            
        Returns:
            bool: True if user confirms, False otherwise
        """
        print("\n" + "="*50)
        print("📋 ANALYSIS SETTINGS SUMMARY")
        print("="*50)
        print(f"📄 PDF File: {os.path.basename(pdf_path)}")
        print(f"📁 Full Path: {pdf_path}")
        print(f"🎯 Domain: {domain}")
        print(f"🔍 Keywords: {len(keywords)} keywords")
        
        if keywords:
            print("   Keywords List:")
            for i, keyword in enumerate(keywords[:10], 1):
                print(f"   {i:2d}. {keyword}")
            if len(keywords) > 10:
                print(f"   ... and {len(keywords) - 10} more")
        else:
            print("   📝 Using generic analysis (no custom keywords)")
        
        print("="*50)
        
        while True:
            confirm = input("\n✅ Proceed with analysis? (y/n/edit): ").lower().strip()
            if confirm in ['y', 'yes']:
                return True
            elif confirm in ['n', 'no']:
                return False
            elif confirm in ['e', 'edit']:
                return False
            else:
                print("Please enter 'y' for yes, 'n' for no, or 'edit' to modify settings")
    
    def display_progress(self, stage: str, message: str = ""):
        """
        Display progress information
        
        Args:
            stage: Current processing stage
            message: Optional additional message
        """
        stages = {
            'validation': '🔍 Validating inputs',
            'extraction': '📖 Extracting content from PDF',
            'analysis': '🧠 Analyzing content with keywords',
            'tables': '📊 Extracting tables',
            'images': '🖼️  Processing images',
            'summary': '📝 Generating summary PDF',
            'complete': '✅ Analysis complete'
        }
        
        stage_message = stages.get(stage, f"🔄 {stage}")
        full_message = f"{stage_message}{': ' + message if message else '...'}"
        print(full_message)
    
    def display_results(self, results: dict):
        """
        Display analysis results in a formatted way
        
        Args:
            results: Dictionary containing analysis results
        """
        print("\n" + "="*70)
        print("🎉 ANALYSIS RESULTS")
        print("="*70)
        
        # Basic stats
        print(f"📄 Input file: {os.path.basename(results.get('input_file', 'Unknown'))}")
        print(f"📋 Summary PDF: {os.path.basename(results.get('summary_pdf', 'Unknown'))}")
        print(f"🎯 Domain: {results.get('domain', 'Unknown')}")
        print(f"🔍 Custom keywords: {len(results.get('custom_keywords', []))}")
        print(f"📊 Total pages: {results.get('total_pages', 0)}")
        print(f"📄 Meaningful pages: {results.get('meaningful_pages', 0)}")
        print(f"🎯 Keyword matches: {results.get('keyword_matches', 0)}")
        print(f"🖼️  Images extracted: {results.get('extracted_figures', 0)}")
        print(f"📈 Tables extracted: {results.get('extracted_tables', 0)}")
        
        # Content types
        content_types = results.get('content_types', [])
        if content_types:
            print(f"🏷️  Content types: {', '.join(content_types)}")
        
        print("\n🎯 FEATURES APPLIED:")
        print("✓ Custom keyword analysis")
        print("✓ Domain-specific organization")
        print("✓ Intelligent content classification")
        print("✓ Keyword highlighting in output")
        print("✓ Professional PDF summary")
        print("✓ Image and table extraction")
        
        # File locations
        print(f"\n📁 Files created:")
        if results.get('summary_pdf'):
            print(f"   📋 Summary: {results['summary_pdf']}")
        
        print("="*70)
    
    def display_error(self, error: Exception, stage: str = ""):
        """
        Display error information in a user-friendly way
        
        Args:
            error: Exception object
            stage: Stage where error occurred
        """
        print(f"\n❌ ERROR{' in ' + stage if stage else ''}")
        print("-" * 50)
        print(f"Error type: {type(error).__name__}")
        print(f"Message: {str(error)}")
        
        # Provide helpful suggestions based on error type
        if isinstance(error, FileNotFoundError):
            print("\n💡 Suggestions:")
            print("• Check if the file path is correct")
            print("• Make sure the file exists")
            print("• Try using quotes around the file path")
            print("• Check if you have permission to read the file")
        
        elif isinstance(error, ValueError):
            print("\n💡 Suggestions:")
            print("• Make sure you're using a valid PDF file")
            print("• Check if the file is corrupted")
            print("• Try with a different PDF file")
        
        elif isinstance(error, PermissionError):
            print("\n💡 Suggestions:")
            print("• Check if you have read permissions for the file")
            print("• Make sure the file is not open in another program")
            print("• Try running with appropriate permissions")
        
        print("-" * 50)
    
    def save_session(self, pdf_path: str, domain: str, keywords: List[str], 
                    output_dir: str) -> str:
        """
        Save session information for future reference
        
        Args:
            pdf_path: Path to PDF file
            domain: Domain name
            keywords: List of keywords
            output_dir: Output directory
            
        Returns:
            str: Path to saved session file
        """
        session_data = {
            'pdf_file': pdf_path,
            'domain': domain,
            'keywords': keywords,
            'timestamp': datetime.now().isoformat(),
            'tool_version': '1.0.0'
        }
        
        session_file = os.path.join(output_dir, 'extraction_session.json')
        
        try:
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            return session_file
        except Exception as e:
            logger.warning(f"Could not save session: {e}")
            return ""
    
    def run_interactive_mode(self):
        """Run the tool in interactive mode"""
        try:
            self.display_welcome()
            
            # Get PDF file
            pdf_path = self.get_pdf_path()
            
            # Get domain
            domain = self.get_domain_name()
            
            # Get keywords
            keywords = self.get_keywords_interactive()
            
            # Confirm settings
            if not self.confirm_settings(pdf_path, domain, keywords):
                print("🔄 Restarting or exiting...")
                return False
            
            # Import and run analysis
            self.display_progress('validation')
            
            try:
                from pdf_analyzer import UniversalPDFAnalyzer
            except ImportError:
                print("❌ Error: pdf_analyzer.py not found in the same directory")
                print("💡 Make sure pdf_analyzer.py is in the same folder as main.py")
                return False
            
            analyzer = UniversalPDFAnalyzer(keywords, domain)
            
            self.display_progress('extraction')
            results = analyzer.process_pdf(pdf_path)
            
            self.display_progress('complete')
            self.display_results(results)
            
            # Save session
            output_dir = os.path.dirname(pdf_path)
            session_file = self.save_session(pdf_path, domain, keywords, output_dir)
            if session_file:
                print(f"💾 Session saved: {os.path.basename(session_file)}")
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Process interrupted by user")
            return False
        except Exception as e:
            self.display_error(e, "processing")
            return False
    
    def run_command_line_mode(self, args):
        """Run the tool in command line mode"""
        try:
            # Get PDF path
            pdf_path = self.get_pdf_path(args.pdf_path)
            
            # Get domain
            domain = self.get_domain_name(args.domain)
            
            # Get keywords
            keywords = []
            if args.keywords:
                keywords = self.parse_keywords_from_string(args.keywords)
            
            # Show settings if not in quiet mode
            if not args.quiet:
                self.display_welcome()
                if not self.confirm_settings(pdf_path, domain, keywords):
                    print("❌ Analysis cancelled by user")
                    return False
            
            if not args.quiet:
                self.display_progress('validation')
            
            try:
                from pdf_analyzer import UniversalPDFAnalyzer
            except ImportError:
                print("❌ Error: pdf_analyzer.py not found in the same directory")
                return False
                
            analyzer = UniversalPDFAnalyzer(keywords, domain)
            
            if not args.quiet:
                self.display_progress('extraction')
            
            results = analyzer.process_pdf(pdf_path)
            
            if not args.quiet:
                self.display_progress('complete')
                self.display_results(results)
            else:
                print(f"✅ Analysis complete: {os.path.basename(results.get('summary_pdf', ''))}")
            
            return True
            
        except Exception as e:
            self.display_error(e, "command line processing")
            return False


def create_argument_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="Universal PDF Content Extractor - Extract and analyze PDF content with custom keywords",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python main.py
  
  Command line mode:
    python main.py document.pdf
    python main.py document.pdf --keywords "AI,machine learning,neural networks"
    python main.py document.pdf --domain "Data Science" --keywords "python,pandas,sklearn"
  
  Quiet mode:
    python main.py document.pdf --keywords "keyword1,keyword2" --quiet
        """
    )
    
    parser.add_argument(
        'pdf_path',
        nargs='?',
        help='Path to the PDF file to analyze'
    )
    
    parser.add_argument(
        '--keywords',
        type=str,
        help='Comma-separated list of keywords to focus on'
    )
    
    parser.add_argument(
        '--domain',
        type=str,
        help='Domain/field name (e.g., "Engineering", "Medical")'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Run in quiet mode with minimal output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Universal PDF Content Extractor 1.0.0'
    )
    
    return parser


def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create interface
    interface = PDFExtractorInterface()
    
    # Run in appropriate mode
    if args.pdf_path:
        # Command line mode
        success = interface.run_command_line_mode(args)
    else:
        # Interactive mode
        success = interface.run_interactive_mode()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()