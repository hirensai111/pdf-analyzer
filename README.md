# Universal PDF Analyzer 📄

A powerful, domain-specific PDF analysis tool with a modern GUI and command-line interface. Extract and analyze PDF content using custom keywords tailored to your field of expertise.

## ✨ Features

- 🎯 **Domain-Specific Analysis** - Pre-built keyword sets for Engineering, Medical, Legal, Data Science, Finance, and more
- 🖥️ **Modern GUI** - Dark theme with purple gradients and intuitive 3-column layout
- ⚡ **CLI Interface** - Powerful command-line tool for batch processing and automation
- 📊 **Smart Content Extraction** - Automatically extracts tables, figures, and meaningful text
- 🔍 **Keyword Highlighting** - Highlights your custom keywords in the generated summary
- 📝 **Professional Reports** - Generates comprehensive PDF summaries with analysis results
- 🚀 **Real-time Progress** - Live progress tracking with detailed status updates

## 🖼️ Screenshots

### Main GUI Interface
![PDF Analyzer GUI](C:\Users\hvellanki\Desktop\Pdf_Summarizer\gui.png)
*Modern 3-column interface: Upload | Domain & Keywords | Analysis Control*

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/hirensai111/pdf-analyzer.git
   cd pdf-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the GUI**
   ```bash
   python gui.py
   ```

### Using the GUI

1. **Upload PDF** - Drag and drop or click to browse
2. **Select Domain** - Choose from Engineering, Medical, Legal, etc.
3. **Add Keywords** - Enter custom keywords or load domain-specific ones
4. **Configure Options** - Enable table extraction, OCR, summary generation
5. **Start Analysis** - Click "🚀 Start Analysis" and watch the progress

### Command Line Usage

```bash
# Basic analysis
python main.py document.pdf

# With custom keywords and domain
python main.py document.pdf --keywords "machine learning,AI,neural networks" --domain "Data Science"

# Quiet mode
python main.py document.pdf --keywords "engineering,design" --quiet
```

## 📁 Project Structure

```
pdf-analyzer/
├── gui.py                  # Modern GUI interface
├── pdf_analyzer.py         # Core analysis engine
├── main.py                 # Command-line interface
├── requirements.txt        # Dependencies
├── LICENSE                 # MIT License
├── keywords/              # Domain-specific keyword files
│   ├── engineering.txt
│   ├── medical.txt
│   ├── legal.txt
│   ├── data_science.txt
│   ├── finance.txt
│   ├── technical.txt
│   └── other.txt
└── examples/              # Example PDF files for testing
    ├── ADVANCED_DATABASE_MANAGEMENT_SYSTEMS.pdf
    ├── AI_planning.pdf
    └── AI_planning_analysis_summary.pdf
```

## 📚 Example PDFs

The `examples/` directory contains sample PDFs for testing:

- **`ADVANCED_DATABASE_MANAGEMENT_SYSTEMS.pdf`** (3.6 MB) - Database management systems documentation
- **`AI_planning.pdf`** (742 KB) - AI planning and algorithms research paper
- **`AI_planning_analysis_summary.pdf`** (1.7 MB) - Generated analysis summary example

These examples demonstrate the analyzer's capabilities across different domains and document types.

## 🎯 Domain Keywords

The analyzer includes pre-built keyword sets for various domains:

| Domain | Keywords Count | Examples |
|--------|---------------|----------|
| **Engineering** | 30+ | algorithm, circuit, design, testing, simulation |
| **Medical** | 30+ | diagnosis, treatment, clinical, therapy, pathology |
| **Legal** | 30+ | contract, litigation, compliance, regulation, evidence |
| **Data Science** | 30+ | machine learning, neural network, classification, optimization |
| **Finance** | 30+ | investment, portfolio, risk management, derivatives |
| **Technical** | 30+ | software, API, framework, debugging, architecture |

## ⚙️ Configuration

### Analysis Options

- **Extract Tables & Figures** - Automatically detect and extract visual elements
- **Generate Summary PDF** - Create professional analysis reports
- **Advanced OCR Processing** - Handle scanned documents (requires Tesseract)
- **Keyword Highlighting** - Highlight keywords in the output

### Custom Keywords

Add your own keywords in two ways:

1. **GUI Method**: Type keywords and press Enter, or load domain-specific sets
2. **File Method**: Edit keyword files in the `keywords/` directory

## 📋 Requirements

- Python 3.7+
- CustomTkinter 5.2.0+
- PyMuPDF (fitz)
- Camelot (table extraction)
- ReportLab (PDF generation)
- Pandas & NumPy

See `requirements.txt` for complete list.

## 🔧 Advanced Usage

### Batch Processing

```bash
# Process multiple files
for file in *.pdf; do
    python main.py "$file" --keywords "your,keywords" --domain "YourDomain"
done
```

### Custom Keyword Files

Create domain-specific keyword files:

```bash
# Create custom keyword file
echo "keyword1\nkeyword2\nkeyword3" > keywords/my_domain.txt
```

### Integration

```python
from pdf_analyzer import UniversalPDFAnalyzer

# Use in your own code
analyzer = UniversalPDFAnalyzer(["keyword1", "keyword2"], "MyDomain")
results = analyzer.process_pdf("document.pdf")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) for the modern GUI framework
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) for PDF processing
- [Camelot](https://github.com/camelot-dev/camelot) for table extraction
- [ReportLab](https://www.reportlab.com/) for PDF generation