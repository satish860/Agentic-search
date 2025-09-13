"""
On-demand PDF downloader for FinanceBench dataset.
Downloads PDFs one at a time from GitHub raw URLs and caches in .finance/ folder.
"""

import os
import json
import requests
from pathlib import Path
from typing import Optional, Dict, Any


class FinancePDFDownloader:
    def __init__(self):
        """Initialize the PDF downloader with cache directory setup."""
        self.base_dir = Path(__file__).parent.parent
        self.cache_dir = self.base_dir / ".finance"
        self.cache_dir.mkdir(exist_ok=True)

        # Load document metadata
        self.metadata_file = self.base_dir / "src" / "datasets" / "finance_bench" / "financebench_document_information.jsonl"
        self._load_metadata()

    def _load_metadata(self):
        """Load document metadata from JSONL file."""
        self.doc_metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                for line in f:
                    doc = json.loads(line.strip())
                    self.doc_metadata[doc['doc_name']] = doc
        else:
            print(f"Warning: Metadata file not found at {self.metadata_file}")

    def get_pdf_path(self, doc_name: str) -> Path:
        """Get the local path where PDF should be cached."""
        return self.cache_dir / f"{doc_name}.pdf"

    def is_cached(self, doc_name: str) -> bool:
        """Check if PDF is already downloaded and cached."""
        pdf_path = self.get_pdf_path(doc_name)
        return pdf_path.exists() and pdf_path.stat().st_size > 0

    def get_github_url(self, doc_name: str) -> str:
        """Get the raw GitHub URL for the PDF."""
        base_url = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"
        return f"{base_url}/{doc_name}.pdf"

    def download_pdf(self, doc_name: str, force_redownload: bool = False) -> Optional[Path]:
        """
        Download a single PDF on-demand.

        Args:
            doc_name: Document name (e.g., "3M_2018_10K")
            force_redownload: If True, re-download even if cached

        Returns:
            Path to downloaded PDF or None if failed
        """
        pdf_path = self.get_pdf_path(doc_name)

        # Check if already cached
        if not force_redownload and self.is_cached(doc_name):
            print(f"PDF already cached: {pdf_path}")
            return pdf_path

        # Get metadata for this document
        if doc_name not in self.doc_metadata:
            print(f"No metadata found for document: {doc_name}")
            return None

        doc_info = self.doc_metadata[doc_name]
        github_url = self.get_github_url(doc_name)

        print(f"Downloading {doc_name} from GitHub...")
        print(f"   Company: {doc_info['company']}")
        print(f"   Type: {doc_info['doc_type'].upper()}")
        print(f"   Year: {doc_info['doc_period']}")
        print(f"   URL: {github_url}")

        try:
            # Download the PDF
            response = requests.get(github_url, stream=True, timeout=30)
            response.raise_for_status()

            # Save to cache
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            file_size = pdf_path.stat().st_size
            print(f"Downloaded successfully: {file_size:,} bytes")
            print(f"   Saved to: {pdf_path}")

            return pdf_path

        except requests.RequestException as e:
            print(f"Download failed: {e}")
            # Clean up partial file if it exists
            if pdf_path.exists():
                pdf_path.unlink()
            return None

    def get_available_documents(self) -> Dict[str, Dict[str, Any]]:
        """Get list of all available documents with metadata."""
        return self.doc_metadata.copy()

    def search_documents(self, company: str = None, doc_type: str = None, year: int = None) -> Dict[str, Dict[str, Any]]:
        """
        Search for documents by criteria.

        Args:
            company: Company name filter
            doc_type: Document type filter ("10k", "10q", "8k")
            year: Year filter

        Returns:
            Dictionary of matching documents
        """
        results = {}
        for doc_name, doc_info in self.doc_metadata.items():
            # Apply filters
            if company and company.upper() not in doc_info['company'].upper():
                continue
            if doc_type and doc_info['doc_type'] != doc_type.lower():
                continue
            if year and doc_info['doc_period'] != year:
                continue

            results[doc_name] = doc_info

        return results


def main():
    """Demo usage of the PDF downloader."""
    downloader = FinancePDFDownloader()

    # Example: Download 3M 2018 10-K
    doc_name = "3M_2018_10K"
    print(f"Testing download for: {doc_name}")

    pdf_path = downloader.download_pdf(doc_name)
    if pdf_path:
        print(f"\nSuccess! PDF available at: {pdf_path}")
    else:
        print(f"\nFailed to download {doc_name}")

    # Show search capabilities
    print(f"\nAvailable 3M documents:")
    m_docs = downloader.search_documents(company="3M")
    for doc_name, info in list(m_docs.items())[:5]:  # Show first 5
        cached = "[CACHED]" if downloader.is_cached(doc_name) else "[NOT CACHED]"
        print(f"  {cached} {doc_name} ({info['doc_type'].upper()} {info['doc_period']})")


if __name__ == "__main__":
    main()