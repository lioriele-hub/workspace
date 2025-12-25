#!/bin/bash
# Script to convert theoretical_answers.md to PDF
# Requires pandoc to be installed: brew install pandoc

echo "Converting theoretical_answers.md to PDF..."

if ! command -v pandoc &> /dev/null
then
    echo "Error: pandoc is not installed."
    echo "Please install it using: brew install pandoc"
    echo ""
    echo "Alternative: You can convert the markdown file to PDF using:"
    echo "  - Online converters (markdown-to-pdf.com)"
    echo "  - VS Code with Markdown PDF extension"
    echo "  - Copy content to Google Docs and export as PDF"
    exit 1
fi

# Convert markdown to PDF
pandoc theoretical_answers.md -o theoretical_answers.pdf \
    --from markdown \
    --pdf-engine=pdflatex \
    --variable geometry:margin=1in \
    --variable fontsize=11pt \
    --toc

if [ $? -eq 0 ]; then
    echo "✓ Successfully created theoretical_answers.pdf"
else
    echo "✗ Conversion failed"
    echo ""
    echo "Alternative methods:"
    echo "1. Open theoretical_answers.md in VS Code and use 'Markdown PDF' extension"
    echo "2. Use an online converter like markdown-to-pdf.com"
    echo "3. Copy content to a word processor and export as PDF"
fi

