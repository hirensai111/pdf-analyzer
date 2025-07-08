#!/usr/bin/env python3
"""
PDF Analyzer GUI Frontend
Modern dark theme with purple gradients using CustomTkinter
3-column layout: Upload | Domain & Keywords | Analysis Control
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import threading
import os
import sys
from typing import List, Optional, Callable
import json
from pathlib import Path
import time

# Try to import the analyzer
try:
    from pdf_analyzer import UniversalPDFAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    print("Warning: pdf_analyzer.py not found. GUI will run in demo mode.")

# Configure CustomTkinter for dark theme with purple accents
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class KeywordTag(ctk.CTkFrame):
    """Custom keyword tag widget with purple gradient styling"""
    
    def __init__(self, parent, text: str, on_remove: Callable = None, **kwargs):
        super().__init__(parent, **kwargs)
        self.text = text
        self.on_remove = on_remove
        
        self.configure(
            fg_color=("#8b5cf6", "#7c3aed"),  # Purple gradient
            corner_radius=15,
            height=28
        )
        
        # Keyword label
        self.label = ctk.CTkLabel(
            self, 
            text=text,
            font=("Arial", 11),
            text_color="white"
        )
        self.label.pack(side="left", padx=(10, 5), pady=4)
        
        # Remove button
        self.remove_btn = ctk.CTkButton(
            self,
            text="×",
            width=20,
            height=20,
            font=("Arial", 12, "bold"),
            fg_color="#ef4444",
            hover_color="#dc2626",
            command=self._remove_clicked
        )
        self.remove_btn.pack(side="right", padx=(0, 5), pady=4)
    
    def _remove_clicked(self):
        if self.on_remove:
            self.on_remove(self.text)
        self.destroy()

class FileUploadFrame(ctk.CTkFrame):
    """File upload widget with drag-and-drop styling"""
    
    def __init__(self, parent, on_file_selected: Callable, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_file_selected = on_file_selected
        self.selected_file = None
        
        self.configure(
            fg_color=("#8b5cf6", "#7c3aed"),  # Purple gradient
            corner_radius=12,
            border_width=2,
            border_color="#a78bfa"
        )
        
        # Upload icon and text
        self.upload_label = ctk.CTkLabel(
            self,
            text="📄\n\nDrop PDF file here\nor click to browse",
            font=("Arial", 16, "bold"),
            text_color="white",
            justify="center"
        )
        self.upload_label.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Bind click event
        self.bind("<Button-1>", self._on_click)
        self.upload_label.bind("<Button-1>", self._on_click)
        
        # Configure hover effects
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
    
    def _on_click(self, event=None):
        self._browse_file()
    
    def _on_enter(self, event=None):
        self.configure(border_color="#c4b5fd")
    
    def _on_leave(self, event=None):
        self.configure(border_color="#a78bfa")
    
    def _browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if file_path:
            self.set_file(file_path)
    
    def set_file(self, file_path: str):
        """Set the selected file and update display"""
        self.selected_file = file_path
        filename = os.path.basename(file_path)
        
        # Update display to show selected file
        self.upload_label.configure(
            text=f"✅ {filename}\n\nClick to change file",
            font=("Arial", 12, "bold")
        )
        
        if self.on_file_selected:
            self.on_file_selected(file_path)

class ProgressPanel(ctk.CTkFrame):
    """Progress tracking panel with status messages"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.configure(
            fg_color=("#1e1b4b", "#312e81"),  # Dark purple gradient
            corner_radius=8,
            border_width=1,
            border_color="#7c3aed"
        )
        
        # Progress title
        self.title_label = ctk.CTkLabel(
            self,
            text="Progress",
            font=("Arial", 14, "bold"),
            text_color="white"
        )
        self.title_label.pack(anchor="w", padx=15, pady=(15, 5))
        
        # Progress bar
        self.progress_var = ctk.DoubleVar(value=0)
        self.progress_bar = ctk.CTkProgressBar(
            self,
            variable=self.progress_var,
            progress_color="#8b5cf6",
            fg_color="#1e1b4b"
        )
        self.progress_bar.pack(fill="x", padx=15, pady=5)
        
        # Progress percentage
        self.percentage_label = ctk.CTkLabel(
            self,
            text="0%",
            font=("Arial", 10),
            text_color="#c4b5fd"
        )
        self.percentage_label.pack(anchor="w", padx=15)
        
        # Status messages frame
        self.status_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.status_frame.pack(fill="both", expand=True, padx=15, pady=10)
        
        # Status messages
        self.status_messages = []
        self._create_status_messages()
    
    def _create_status_messages(self):
        """Create initial status messages"""
        messages = [
            ("PDF loaded successfully", "pending"),
            ("Text extraction complete", "pending"),
            ("Processing keywords", "pending"),
            ("Extracting tables", "pending"),
            ("Generating summary", "pending")
        ]
        
        for i, (message, status) in enumerate(messages):
            label = ctk.CTkLabel(
                self.status_frame,
                text=f"⏸ {message}",
                font=("Arial", 11),
                text_color="#64748b",
                anchor="w"
            )
            label.pack(anchor="w", pady=2)
            self.status_messages.append(label)
    
    def update_progress(self, percentage: float, current_step: int = -1):
        """Update progress bar and percentage"""
        self.progress_var.set(percentage / 100)
        self.percentage_label.configure(text=f"{percentage:.0f}%")
        
        # Update status messages
        for i, label in enumerate(self.status_messages):
            if i < current_step:
                # Completed
                message = label.cget("text")[2:]  # Remove emoji
                label.configure(text=f"✓ {message}", text_color="#22c55e")
            elif i == current_step:
                # Current
                message = label.cget("text")[2:]  # Remove emoji
                label.configure(text=f"⏳ {message}", text_color="#f59e0b")
            else:
                # Pending
                message = label.cget("text")[2:]  # Remove emoji
                label.configure(text=f"⏸ {message}", text_color="#64748b")
    
    def reset(self):
        """Reset progress to initial state"""
        self.update_progress(0, -1)

class PDFAnalyzerGUI:
    """Main GUI application for PDF Analyzer"""
    
    def __init__(self):
        # Initialize main window
        self.root = ctk.CTk()
        self.root.title("Universal PDF Analyzer")
        self.root.geometry("1200x700")
        self.root.configure(fg_color="#000000")  # Pure black background
        
        # Variables
        self.selected_file = None
        self.keywords = []
        self.domain = "Engineering"  # Default domain
        self.analysis_options = {
            "extract_tables": True,
            "generate_summary": True,
            "advanced_ocr": False,
            "keyword_highlighting": True
        }
        
        # Analysis state
        self.is_analyzing = False
        self.analyzer = None
        
        # Initialize UI components
        self.load_domain_btn = None
        self.keywords_scroll = None
        self.keywords_count_label = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the main user interface"""
        # Title section
        self.setup_header()
        
        # Main content in 3 columns
        self.setup_main_content()
        
        # Footer
        self.setup_footer()
    
    def setup_header(self):
        """Setup header with title and logo"""
        header_frame = ctk.CTkFrame(self.root, fg_color="transparent", height=80)
        header_frame.pack(fill="x", padx=20, pady=20)
        header_frame.pack_propagate(False)
        
        # Logo and title
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.pack(side="left", fill="y")
        
        # Logo
        logo_label = ctk.CTkLabel(
            title_frame,
            text="📄",
            font=("Arial", 24),
            width=40,
            height=40,
            fg_color=("#8b5cf6", "#7c3aed"),
            corner_radius=20
        )
        logo_label.pack(side="left", padx=(0, 15), pady=5)
        
        # Title text
        title_text_frame = ctk.CTkFrame(title_frame, fg_color="transparent")
        title_text_frame.pack(side="left", fill="y")
        
        title_label = ctk.CTkLabel(
            title_text_frame,
            text="Universal PDF Analyzer",
            font=("Arial", 24, "bold"),
            text_color="white"
        )
        title_label.pack(anchor="w")
        
        subtitle_label = ctk.CTkLabel(
            title_text_frame,
            text="Extract and analyze PDF content with custom keywords",
            font=("Arial", 14),
            text_color="#c4b5fd"
        )
        subtitle_label.pack(anchor="w")
    
    def setup_main_content(self):
        """Setup main 3-column layout"""
        main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Configure grid
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_columnconfigure(2, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Column 1: PDF Upload
        self.setup_upload_column(main_frame)
        
        # Column 2: Domain & Keywords
        self.setup_keywords_column(main_frame)
        
        # Column 3: Analysis Control
        self.setup_analysis_column(main_frame)
    
    def setup_upload_column(self, parent):
        """Setup PDF upload column"""
        upload_frame = ctk.CTkFrame(parent, fg_color="transparent")
        upload_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Column title
        title_label = ctk.CTkLabel(
            upload_frame,
            text="📄 Upload PDF",
            font=("Arial", 18, "bold"),
            text_color="white"
        )
        title_label.pack(anchor="w", pady=(0, 20))
        
        # File upload area
        self.upload_widget = FileUploadFrame(
            upload_frame,
            on_file_selected=self.on_file_selected,
            height=200
        )
        self.upload_widget.pack(fill="x", pady=(0, 20))
        
        # File info panel
        self.file_info_frame = ctk.CTkFrame(
            upload_frame,
            fg_color=("#1e1b4b", "#312e81"),
            corner_radius=8,
            border_width=1,
            border_color="#7c3aed"
        )
        self.file_info_frame.pack(fill="x", pady=(0, 20))
        
        # File info content
        self.setup_file_info_panel()
        
        # Upload tips panel
        self.setup_upload_tips_panel(upload_frame)
    
    def setup_file_info_panel(self):
        """Setup file information display"""
        info_title = ctk.CTkLabel(
            self.file_info_frame,
            text="File Information",
            font=("Arial", 14, "bold"),
            text_color="white"
        )
        info_title.pack(anchor="w", padx=15, pady=(15, 5))
        
        self.file_name_label = ctk.CTkLabel(
            self.file_info_frame,
            text="No file selected",
            font=("Arial", 12),
            text_color="#c4b5fd"
        )
        self.file_name_label.pack(anchor="w", padx=15)
        
        self.file_size_label = ctk.CTkLabel(
            self.file_info_frame,
            text="",
            font=("Arial", 10),
            text_color="#94a3b8"
        )
        self.file_size_label.pack(anchor="w", padx=15, pady=(0, 15))
    
    def setup_upload_tips_panel(self, parent):
        """Setup upload tips panel instead of statistics"""
        tips_frame = ctk.CTkFrame(
            parent,
            fg_color=("#1e1b4b", "#312e81"),
            corner_radius=8,
            border_width=1,
            border_color="#7c3aed"
        )
        tips_frame.pack(fill="both", expand=True)
        
        tips_title = ctk.CTkLabel(
            tips_frame,
            text="💡 Tips for Better Analysis",
            font=("Arial", 14, "bold"),
            text_color="white"
        )
        tips_title.pack(anchor="w", padx=15, pady=(15, 10))
        
        tips = [
            "• Use specific technical keywords",
            "• Choose the right domain field",
            "• PDFs with clear text work best",
            "• Enable OCR for scanned documents",
            "• Use domain suggestions for quick setup"
        ]
        
        for tip in tips:
            tip_label = ctk.CTkLabel(
                tips_frame,
                text=tip,
                font=("Arial", 11),
                text_color="#e2e8f0"
            )
            tip_label.pack(anchor="w", padx=15, pady=2)
        
        # Spacer
        ctk.CTkLabel(tips_frame, text="", height=10).pack()
    
    def setup_keywords_column(self, parent):
        """Setup domain and keywords column"""
        keywords_frame = ctk.CTkFrame(parent, fg_color="transparent")
        keywords_frame.grid(row=0, column=1, sticky="nsew", padx=10)
        
        # Column title
        title_label = ctk.CTkLabel(
            keywords_frame,
            text="🎯 Domain & Keywords",
            font=("Arial", 18, "bold"),
            text_color="white"
        )
        title_label.pack(anchor="w", pady=(0, 20))
        
        # Domain selection
        domain_label = ctk.CTkLabel(
            keywords_frame,
            text="Domain/Field",
            font=("Arial", 14, "bold"),
            text_color="white"
        )
        domain_label.pack(anchor="w", pady=(0, 5))
        
        self.domain_combo = ctk.CTkComboBox(
            keywords_frame,
            values=["Engineering", "Medical", "Legal", "Data Science", "Finance", "Technical", "Other"],
            fg_color=("#1e1b4b", "#312e81"),
            border_color="#7c3aed",
            button_color="#8b5cf6",
            command=self.on_domain_changed,
            state="readonly"  # Make dropdown non-editable
        )
        self.domain_combo.set("Engineering")
        self.domain_combo.pack(fill="x", pady=(0, 20))
        
        # Keywords section
        keywords_label = ctk.CTkLabel(
            keywords_frame,
            text="Custom Keywords",
            font=("Arial", 14, "bold"),
            text_color="white"
        )
        keywords_label.pack(anchor="w", pady=(0, 5))
        
        keywords_desc = ctk.CTkLabel(
            keywords_frame,
            text="Enter keywords to focus your analysis",
            font=("Arial", 12),
            text_color="#c4b5fd"
        )
        keywords_desc.pack(anchor="w", pady=(0, 10))
        
        # Keyword input
        self.keyword_entry = ctk.CTkEntry(
            keywords_frame,
            placeholder_text="Enter keyword and press Enter",
            fg_color=("#1e1b4b", "#312e81"),
            border_color="#7c3aed"
        )
        self.keyword_entry.pack(fill="x", pady=(0, 15))
        self.keyword_entry.bind("<Return>", self.add_keyword)
        
        # Keywords suggestions from domain files
        self.setup_keyword_suggestions(keywords_frame)
    
    def setup_keyword_suggestions(self, parent):
        """Setup keyword suggestions and container"""
        # Domain suggestions
        suggestions_frame = ctk.CTkFrame(
            parent,
            fg_color=("#1e1b4b", "#312e81"),
            corner_radius=8,
            border_width=1,
            border_color="#7c3aed"
        )
        suggestions_frame.pack(fill="x", pady=(0, 15))
        
        suggestions_title = ctk.CTkLabel(
            suggestions_frame,
            text="Quick Add:",
            font=("Arial", 12, "bold"),
            text_color="white"
        )
        suggestions_title.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Buttons frame
        buttons_frame = ctk.CTkFrame(suggestions_frame, fg_color="transparent")
        buttons_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        # Load domain keywords button
        self.load_domain_btn = ctk.CTkButton(
            buttons_frame,
            text=f"Load {self.domain} Keywords",
            font=("Arial", 11),
            height=28,
            fg_color="#8b5cf6",
            hover_color="#7c3aed",
            command=self.load_domain_keywords
        )
        self.load_domain_btn.pack(side="left", padx=(0, 10))
        
        # Clear all button
        clear_btn = ctk.CTkButton(
            buttons_frame,
            text="Clear All",
            font=("Arial", 11),
            height=28,
            fg_color="#ef4444",
            hover_color="#dc2626",
            command=self.clear_all_keywords
        )
        clear_btn.pack(side="left")
        
        # Keywords container
        self.setup_keywords_container(parent)
    
    def setup_keywords_container(self, parent):
        """Setup keywords container with tags"""
        container_frame = ctk.CTkFrame(
            parent,
            fg_color=("#1e1b4b", "#312e81"),
            corner_radius=8,
            border_width=1,
            border_color="#7c3aed"
        )
        container_frame.pack(fill="both", expand=True)
        
        container_title = ctk.CTkLabel(
            container_frame,
            text="Added Keywords:",
            font=("Arial", 12, "bold"),
            text_color="white"
        )
        container_title.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Scrollable frame for keywords
        self.keywords_scroll = ctk.CTkScrollableFrame(
            container_frame,
            fg_color="transparent",
            height=120
        )
        self.keywords_scroll.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # Keywords stats
        self.keywords_stats_frame = ctk.CTkFrame(container_frame, fg_color="transparent")
        self.keywords_stats_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        self.keywords_count_label = ctk.CTkLabel(
            self.keywords_stats_frame,
            text="0 keywords added",
            font=("Arial", 10),
            text_color="#94a3b8"
        )
        self.keywords_count_label.pack(anchor="w")
    
    def setup_analysis_column(self, parent):
        """Setup analysis control column"""
        analysis_frame = ctk.CTkFrame(parent, fg_color="transparent")
        analysis_frame.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        
        # Column title
        title_label = ctk.CTkLabel(
            analysis_frame,
            text="⚙️ Analysis Control",
            font=("Arial", 18, "bold"),
            text_color="white"
        )
        title_label.pack(anchor="w", pady=(0, 20))
        
        # Analysis options
        self.setup_analysis_options(analysis_frame)
        
        # Analysis button
        self.analyze_button = ctk.CTkButton(
            analysis_frame,
            text="🚀 Start Analysis",
            font=("Arial", 16, "bold"),
            height=50,
            fg_color=("#8b5cf6", "#7c3aed"),
            hover_color=("#7c3aed", "#6d28d9"),
            command=self.start_analysis
        )
        self.analyze_button.pack(fill="x", pady=20)
        
        # Progress panel
        self.progress_panel = ProgressPanel(analysis_frame)
        self.progress_panel.pack(fill="both", expand=True)
    
    def setup_analysis_options(self, parent):
        """Setup analysis options checkboxes"""
        options_frame = ctk.CTkFrame(
            parent,
            fg_color=("#1e1b4b", "#312e81"),
            corner_radius=8,
            border_width=1,
            border_color="#7c3aed"
        )
        options_frame.pack(fill="x", pady=(0, 20))
        
        options_title = ctk.CTkLabel(
            options_frame,
            text="Options",
            font=("Arial", 14, "bold"),
            text_color="white"
        )
        options_title.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Checkboxes
        self.option_vars = {}
        options = [
            ("extract_tables", "Extract tables and figures"),
            ("generate_summary", "Generate summary PDF"),
            ("advanced_ocr", "Advanced OCR processing"),
            ("keyword_highlighting", "Keyword highlighting")
        ]
        
        for key, text in options:
            var = ctk.BooleanVar(value=self.analysis_options[key])
            self.option_vars[key] = var
            
            checkbox = ctk.CTkCheckBox(
                options_frame,
                text=text,
                variable=var,
                fg_color="#8b5cf6",
                hover_color="#7c3aed",
                command=lambda k=key: self.update_option(k)
            )
            checkbox.pack(anchor="w", padx=15, pady=5)
        
        # Spacer
        ctk.CTkLabel(options_frame, text="", height=10).pack()
    
    def setup_footer(self):
        """Setup footer with version info"""
        footer_frame = ctk.CTkFrame(self.root, fg_color="transparent", height=30)
        footer_frame.pack(fill="x", padx=20, pady=(0, 10))
        footer_frame.pack_propagate(False)
        
        footer_text = ctk.CTkLabel(
            footer_frame,
            text="Universal PDF Analyzer v1.0 • Supports all PDF types • Powered by PyMuPDF & Camelot",
            font=("Arial", 10),
            text_color="#8b5cf6"
        )
        footer_text.pack(side="left")
        
        help_text = ctk.CTkLabel(
            footer_frame,
            text="Need help? Check documentation",
            font=("Arial", 10),
            text_color="#8b5cf6"
        )
        help_text.pack(side="right")
    
    # Event handlers
    def on_file_selected(self, file_path: str):
        """Handle file selection"""
        self.selected_file = file_path
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        # Update file info
        self.file_name_label.configure(text=f"✅ {filename}")
        self.file_size_label.configure(text=f"{file_size / (1024*1024):.1f} MB")
    
    def on_domain_changed(self, value):
        """Handle domain selection change"""
        self.domain = value
        # Update the load button text
        if self.load_domain_btn:
            self.load_domain_btn.configure(text=f"Load {self.domain} Keywords")
    
    def add_keyword(self, event=None):
        """Add keyword from entry field"""
        keyword = self.keyword_entry.get().strip()
        if keyword and keyword not in self.keywords:
            self.keywords.append(keyword)
            self.keyword_entry.delete(0, 'end')
            self.update_keywords_display()
    
    def remove_keyword(self, keyword: str):
        """Remove keyword"""
        if keyword in self.keywords:
            self.keywords.remove(keyword)
            self.update_keywords_display()
    
    def update_keywords_display(self):
        """Update keywords display"""
        if not self.keywords_scroll:
            return
            
        # Clear existing tags
        for widget in self.keywords_scroll.winfo_children():
            widget.destroy()
        
        # Add keyword tags
        for keyword in self.keywords:
            tag = KeywordTag(
                self.keywords_scroll,
                text=keyword,
                on_remove=self.remove_keyword
            )
            tag.pack(anchor="w", pady=2, padx=5, fill="x")
        
        # Update stats
        count = len(self.keywords)
        if self.keywords_count_label:
            self.keywords_count_label.configure(text=f"{count} keywords added")
    
    def update_option(self, key: str):
        """Update analysis option"""
        self.analysis_options[key] = self.option_vars[key].get()
    
    def load_domain_keywords(self):
        """Load keywords from domain-specific file"""
        domain_file = f"keywords/{self.domain.lower().replace(' ', '_')}.txt"
        
        if not os.path.exists(domain_file):
            # Create keywords directory and files if they don't exist
            self.create_keyword_files_if_needed()
        
        try:
            with open(domain_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Parse keywords (skip comments and empty lines)
            new_keywords = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    if line not in self.keywords:
                        new_keywords.append(line)
            
            # Add new keywords
            self.keywords.extend(new_keywords)
            self.update_keywords_display()
            
            # Show notification
            messagebox.showinfo(
                "Keywords Loaded", 
                f"Added {len(new_keywords)} new keywords from {self.domain} domain!"
            )
            
        except FileNotFoundError:
            messagebox.showerror(
                "File Not Found", 
                f"Could not find keyword file for {self.domain} domain.\nPlease create keywords/{domain_file}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Error loading keywords: {str(e)}")
    
    def clear_all_keywords(self):
        """Clear all keywords"""
        if self.keywords:
            self.keywords.clear()
            self.update_keywords_display()
            messagebox.showinfo("Keywords Cleared", "All keywords have been removed.")
    
    def create_keyword_files_if_needed(self):
        """Create keyword files if they don't exist"""
        keywords_dir = Path("keywords")
        keywords_dir.mkdir(exist_ok=True)
        
        # Domain-specific keywords
        domain_keywords = {
            "Engineering": [
                "algorithm", "analysis", "architecture", "automation", "calibration",
                "circuit", "computation", "control", "design", "development",
                "efficiency", "embedded", "framework", "implementation", "integration",
                "manufacture", "measurement", "modeling", "optimization", "performance",
                "processing", "protocol", "reliability", "simulation", "specification",
                "system", "testing", "validation", "verification", "workflow"
            ],
            
            "Medical": [
                "diagnosis", "treatment", "patient", "clinical", "therapy",
                "medication", "disease", "symptoms", "pathology", "anatomy",
                "physiology", "surgery", "procedure", "healthcare", "medical",
                "pharmaceutical", "epidemiology", "radiology", "cardiology", "neurology",
                "oncology", "pediatrics", "geriatrics", "immunology", "microbiology",
                "biochemistry", "pharmacology", "toxicology", "genetics", "biomarker"
            ],
            
            "Legal": [
                "contract", "agreement", "litigation", "compliance", "regulation",
                "statute", "jurisdiction", "precedent", "case law", "judicial",
                "attorney", "counsel", "defendant", "plaintiff", "evidence",
                "testimony", "verdict", "settlement", "arbitration", "mediation",
                "intellectual property", "copyright", "trademark", "patent", "license",
                "liability", "negligence", "damages", "breach", "constitution"
            ],
            
            "Data Science": [
                "algorithm", "machine learning", "deep learning", "neural network", "artificial intelligence",
                "data mining", "big data", "analytics", "statistics", "regression",
                "classification", "clustering", "prediction", "modeling", "dataset",
                "feature engineering", "cross validation", "overfitting", "underfitting", "bias",
                "variance", "supervised learning", "unsupervised learning", "reinforcement learning", "natural language processing",
                "computer vision", "time series", "dimensionality reduction", "ensemble methods", "optimization"
            ],
            
            "Finance": [
                "investment", "portfolio", "risk management", "asset", "liability",
                "equity", "debt", "derivatives", "options", "futures",
                "valuation", "financial modeling", "cash flow", "revenue", "profit",
                "loss", "balance sheet", "income statement", "financial statement", "audit",
                "compliance", "regulation", "market", "trading", "securities",
                "bond", "stock", "dividend", "interest rate", "inflation"
            ],
            
            "Technical": [
                "software", "hardware", "programming", "development", "coding",
                "database", "server", "network", "security", "encryption",
                "protocol", "interface", "API", "framework", "library",
                "algorithm", "data structure", "performance", "scalability", "optimization",
                "debugging", "testing", "deployment", "version control", "documentation",
                "architecture", "design pattern", "microservices", "cloud computing", "DevOps"
            ],
            
            "Other": [
                "analysis", "research", "methodology", "framework", "approach",
                "evaluation", "assessment", "comparison", "study", "investigation",
                "process", "procedure", "standard", "guideline", "best practice",
                "implementation", "application", "solution", "strategy", "planning",
                "management", "organization", "structure", "development", "improvement",
                "quality", "efficiency", "effectiveness", "innovation", "technology"
            ]
        }
        
        # Create file for current domain if it doesn't exist
        domain_file = keywords_dir / f"{self.domain.lower().replace(' ', '_')}.txt"
        if not domain_file.exists() and self.domain in domain_keywords:
            with open(domain_file, 'w', encoding='utf-8') as f:
                f.write(f"# {self.domain} Domain Keywords\n")
                f.write(f"# Generated keyword list for {self.domain} domain analysis\n\n")
                
                for keyword in sorted(domain_keywords[self.domain]):
                    f.write(f"{keyword}\n")
    
    def start_analysis(self):
        """Start PDF analysis"""
        if not self.selected_file:
            messagebox.showerror("Error", "Please select a PDF file first.")
            return
        
        if self.is_analyzing:
            messagebox.showwarning("Warning", "Analysis is already in progress.")
            return
        
        # Disable analyze button
        self.analyze_button.configure(state="disabled", text="Analyzing...")
        self.is_analyzing = True
        
        # Reset progress
        self.progress_panel.reset()
        
        # Start analysis in background thread
        analysis_thread = threading.Thread(target=self._run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def _run_analysis(self):
        """Run analysis in background thread"""
        try:
            if ANALYZER_AVAILABLE:
                # Real analysis
                self._run_real_analysis()
            else:
                # Demo analysis
                self._run_demo_analysis()
            
        except Exception as e:
            self.root.after(0, lambda: self._on_analysis_error(str(e)))
        finally:
            self.root.after(0, self._on_analysis_complete)
    
    def _run_real_analysis(self):
        """Run real PDF analysis"""
        try:
            # Initialize analyzer
            analyzer = UniversalPDFAnalyzer(self.keywords, self.domain)
            
            # Update progress
            self.root.after(0, lambda: self.progress_panel.update_progress(10, 0))
            
            # Run analysis
            results = analyzer.process_pdf(self.selected_file)
            
            # Update progress
            self.root.after(0, lambda: self.progress_panel.update_progress(100, 5))
            
            # Show results
            self.root.after(0, lambda: self._show_results(results))
            
        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}")
    
    def _run_demo_analysis(self):
        """Run demo analysis with simulated progress"""
        steps = [
            (20, 0, "Loading PDF..."),
            (40, 1, "Extracting text..."),
            (60, 2, "Processing keywords..."),
            (80, 3, "Extracting tables..."),
            (100, 4, "Generating summary...")
        ]
        
        for progress, step, message in steps:
            time.sleep(1)  # Simulate processing time
            self.root.after(0, lambda p=progress, s=step: self.progress_panel.update_progress(p, s))
        
        # Simulate results
        demo_results = {
            'input_file': self.selected_file,
            'summary_pdf': self.selected_file.replace('.pdf', '_analysis_summary.pdf'),
            'domain': self.domain,
            'custom_keywords': self.keywords,
            'total_pages': 45,
            'meaningful_pages': 42,
            'extracted_figures': 12,
            'extracted_tables': 8,
            'keyword_matches': len(self.keywords) * 5,
            'processing_time': 5.2,
            'processing_status': 'success'
        }
        
        self.root.after(0, lambda: self._show_results(demo_results))
    
    def _show_results(self, results: dict):
        """Show analysis results"""
        result_text = f"""Analysis Complete! ✅

📄 File: {os.path.basename(results.get('input_file', 'Unknown'))}
🎯 Domain: {results.get('domain', 'Unknown')}
📊 Pages processed: {results.get('total_pages', 0)}
🔍 Keywords found: {results.get('keyword_matches', 0)} matches
🖼️ Images extracted: {results.get('extracted_figures', 0)}
📈 Tables extracted: {results.get('extracted_tables', 0)}
⏱️ Processing time: {results.get('processing_time', 0):.1f} seconds

Summary PDF: {os.path.basename(results.get('summary_pdf', 'Not generated'))}"""
        
        messagebox.showinfo("Analysis Complete", result_text)
    
    def _on_analysis_error(self, error_message: str):
        """Handle analysis error"""
        messagebox.showerror("Analysis Error", f"Analysis failed:\n\n{error_message}")
    
    def _on_analysis_complete(self):
        """Handle analysis completion"""
        self.analyze_button.configure(state="normal", text="🚀 Start Analysis")
        self.is_analyzing = False
    
    def run(self):
        """Run the GUI application"""
        self.root.mainloop()

def main():
    """Main entry point"""
    if not ANALYZER_AVAILABLE:
        print("⚠️  Warning: Running in demo mode")
        print("To enable full functionality, ensure pdf_analyzer.py is in the same directory")
        print("=" * 60)
    
    app = PDFAnalyzerGUI()
    app.run()

if __name__ == "__main__":
    main()