import os
import sys
import tempfile
import time
import warnings
import concurrent.futures
from typing import List, Optional, Dict, Any, Tuple, Union
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# Suppress specific deprecation warnings
import logging
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("pdf2image").setLevel(logging.ERROR)
logging.getLogger("easyocr").setLevel(logging.ERROR)
logging.getLogger("pytesseract").setLevel(logging.ERROR)

# Import after setting up logging
from pdf2image import convert_from_path
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredFileLoader,
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import easyocr
import numpy as np
from config import settings
import base64
from io import BytesIO
import google.generativeai as genai
from google.api_core import retry as google_retry
from google.api_core.exceptions import (
    GoogleAPIError,
    ResourceExhausted,
    ServiceUnavailable,
    DeadlineExceeded
)


class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )
        self.reader = easyocr.Reader(['en'])
        
        # Initialize Google's Generative AI
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        
        self.max_workers = min(4, os.cpu_count() or 2)  # Limit parallel workers
        self.batch_size = 3  # Reduced batch size for Gemini's rate limits

    def load_document(self, file_path: str) -> List[Document]:
        """Load and process document based on file type"""
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                return self._load_pdf(file_path)
            elif file_extension == '.docx':
                return self._load_docx(file_path)
            elif file_extension == '.txt':
                return self._load_text(file_path)
            elif file_extension == '.csv':
                return self._load_csv(file_path)
            elif file_extension in ['.jpg', '.jpeg', '.png']:
                return self._load_image(file_path)
            else:
                return self._load_unstructured(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return []

    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF file with both regular text and image text extraction"""
        try:
            # Extract regular text
            loader = PyPDFLoader(file_path)
            documents = loader.load_and_split()
            
            # Extract text from images in the PDF
            image_text = self._extract_text_from_pdf_images(file_path)
            if image_text:
                # Add image text as an additional document
                documents.append(Document(
                    page_content=image_text,
                    metadata={
                        "source": file_path,
                        "page": "images",
                        "type": "extracted_images"
                    }
                ))
                
            return documents
            
        except Exception as e:
            print(f"Error loading PDF {file_path}: {str(e)}")
            return []

    def _is_blank_image(self, image, threshold=0.95):
        """Check if an image is mostly blank/white"""
        if image.mode != 'L':
            image = image.convert('L')
        
        # Calculate the fraction of white pixels
        histogram = image.histogram()
        pixels = float(image.size[0] * image.size[1])
        white_pixels = histogram[-1]  # Last bin contains white pixels (255)
        white_ratio = white_pixels / pixels
        
        return white_ratio > threshold

    @google_retry.Retry(
        predicate=google_retry.if_exception_type(
            ResourceExhausted, ServiceUnavailable, DeadlineExceeded, GoogleAPIError
        ),
        initial=1.0,
        maximum=10.0,
        multiplier=2.0,
        deadline=60.0  # 60 seconds deadline
    )
    def _call_gemini_vision(self, prompt: str, images: List[Image.Image]) -> str:
        """Make API call to Google's Gemini Vision with retry logic"""
        try:
            # Convert images to base64 for Gemini
            image_parts = []
            for img in images:
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                image_parts.append({
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(buffered.getvalue()).decode("utf-8")
                })
            
            # Prepare the content for Gemini
            content = [{"text": prompt}]
            for img_part in image_parts:
                content.append({"inline_data": img_part})
            
            # Make the API call
            response = self.gemini_model.generate_content(
                contents=[{"role": "user", "parts": content}],
                generation_config={
                    "max_output_tokens": 8192,
                    "temperature": 0.0,
                },
                request_options={"timeout": 60}  # 60 seconds timeout
            )
            
            if response.text:
                return response.text
            else:
                raise GoogleAPIError("No text in Gemini response")
                
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            raise

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image to improve OCR accuracy"""
        # Convert to grayscale
        image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Apply slight sharpening
        image = image.filter(ImageFilter.SHARPEN)
        
        # Apply adaptive thresholding
        image = image.point(lambda p: 255 if p > 200 else 0)
        
        return image

    def _image_to_base64(self, image: Image.Image, temp_dir: str, idx: int) -> str:
        """Convert image to base64 with optimized settings"""
        try:
            # Optimize image for OCR
            image = self._preprocess_image(image)
            
            # Convert to RGB mode for JPEG
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Save with optimized settings
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=85, optimize=True, progressive=True)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return ""

    def _process_single_page(self, args):
        """Process a single page image."""
        i, image, temp_dir = args
        try:
            # Skip if image is mostly blank
            if self._is_blank_image(image):
                return f"[Skipped blank page {i+1}]"
                
            # Convert image to base64 for API call
            img_base64 = self._image_to_base64(image, temp_dir, i)
            if not img_base64:
                return ""
                
            # Prepare message for API call
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this image. Be thorough and include all visible text."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }]
            
            # Make API call with retry logic
            response = self._call_openai_vision(messages)
            text = response.choices[0].message.content if response and response.choices else ""
                
            if text and text.strip():
                return f"[From image on page {i+1}]: {text}"
                
        except Exception as e:
            print(f"Error processing page {i+1}: {str(e)}")
            
        return ""

    def _process_image_batch(self, batch: List[Tuple[int, Image.Image, str]]) -> List[Tuple[int, str]]:
        """Process a batch of images using Gemini Vision API"""
        results = []
        
        try:
            # Extract images from batch
            images = [img for _, img, _ in batch]
            
            # Create a prompt for Gemini
            prompt = """Extract all text from these images. Be thorough and include all visible text. 
            For each image, start with [PAGE_X] where X is the page number (starting from 1).
            Ensure you extract all text including headers, footers, tables, and any other visible content.
            """
            
            # Call Gemini Vision API
            response_text = self._call_gemini_vision(prompt, images)
            
            # Split the response by page markers
            page_texts = self._split_response_by_pages(response_text, len(batch))
            
            # Process results
            for (page_num, _, _), page_text in zip(batch, page_texts):
                if page_text and page_text.strip():
                    results.append((page_num, f"[From image on page {page_num+1}]: {page_text}"))
                    
        except Exception as e:
            print(f"Error processing batch with Gemini Vision: {str(e)}")
            
        return results

    def _split_response_by_pages(self, text: str, expected_pages: int) -> List[str]:
        """Split the API response into individual page texts"""
        if not text:
            return [""] * expected_pages
            
        # Split by page markers if present
        if "[PAGE_" in text:
            pages = []
            current_page = 0
            lines = text.split('\n')
            
            for line in lines:
                if line.strip().startswith('[PAGE_'):
                    # Extract page number
                    try:
                        current_page = int(line.split('_')[1].rstrip(']')) - 1
                        pages.append("")
                    except (IndexError, ValueError):
                        pass
                elif pages:
                    pages[current_page] += line + '\n'
            # Ensure we have the expected number of pages
            while len(pages) < expected_pages:
                pages.append("")
                
            return pages[:expected_pages]
            
        # If no page markers, split equally (fallback)
        avg_len = len(text) // expected_pages
        return [text[i*avg_len:(i+1)*avg_len].strip() for i in range(expected_pages)]

    def _extract_text_from_pdf_images(self, file_path: str) -> str:
        """Extract text from images in PDF using OpenAI Vision with batch processing"""
        try:
            # Convert PDF to images with reduced DPI and grayscale
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Converting PDF to images: {file_path}")
                start_time = time.time()
                
                images = convert_from_path(
                    file_path,
                    dpi=150,  # Reduced from default 300 DPI
                    grayscale=True,  # Process in grayscale
                    output_folder=temp_dir,
                    poppler_path='/opt/homebrew/bin',
                    fmt='jpeg',
                    thread_count=4  # Use multiple threads for conversion
                )
                
                print(f"Converted {len(images)} pages in {time.time() - start_time:.2f} seconds")
                
                if not images:
                    return ""
                
                all_text = [""] * len(images)
                
                # Prepare batches of images to process
                batches = []
                for i in range(0, len(images), self.batch_size):
                    batch = images[i:i + self.batch_size]
                    batch_tuples = [(i + j, img, temp_dir) for j, img in enumerate(batch)]
                    batches.append(batch_tuples)
                
                print(f"Processing {len(batches)} batches with {self.max_workers} workers")
                start_time = time.time()
                
                # Process batches in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    
                    # Submit all batches for processing
                    for batch in batches:
                        future = executor.submit(self._process_image_batch, batch)
                        futures.append(future)
                    
                    # Process results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            results = future.result()
                            for page_num, text in results:
                                if text and text.strip():
                                    all_text[page_num] = text
                        except Exception as e:
                            print(f"Error processing batch: {str(e)}")
                
                print(f"Processed {len(images)} pages in {time.time() - start_time:.2f} seconds")
                
                # Combine all non-empty texts
                return '\n\n'.join(filter(None, all_text))
                
        except Exception as e:
            print(f"Error extracting text from PDF images: {str(e)}")
            return ""

    def _load_docx(self, file_path: str) -> List[Document]:
        """Load Word document"""
        loader = Docx2txtLoader(file_path)
        return loader.load()

    def _load_text(self, file_path: str) -> List[Document]:
        """Load text file"""
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()

    def _load_csv(self, file_path: str) -> List[Document]:
        """Load CSV file"""
        loader = CSVLoader(file_path=file_path)
        return loader.load()

    def _extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image using Gemini Vision API with fallback to other OCR methods"""
        try:
            # First try with Gemini Vision
            with Image.open(image_path) as img:
                # Call Gemini Vision API
                prompt = "Extract all text from this image. Be thorough and include all visible text, including headers, footers, tables, and any other content."
                text = self._call_gemini_vision(prompt, [img])
                
                if text and text.strip():
                    return text
                
                # If we get here, Gemini Vision failed to return text
                raise Exception("Gemini Vision API returned no results")
                
        except Exception as e:
            print(f"Gemini Vision failed for {image_path}, trying fallback OCR: {str(e)}")
            
            # Fallback to EasyOCR
            try:
                result = self.reader.readtext(image_path, detail=0)
                if result:
                    return '\n'.join(result)
            except Exception as ocr_error:
                print(f"EasyOCR failed: {str(ocr_error)}")
            
            # Final fallback to Tesseract
            try:
                return pytesseract.image_to_string(Image.open(image_path))
            except Exception as tesseract_error:
                print(f"Tesseract failed: {str(tesseract_error)}")
                return ""

    def _load_image(self, file_path: str) -> List[Document]:
        """Extract text from image using OCR with multiple fallback methods"""
        try:
            # First try with our enhanced method which includes fallbacks
            text = self._extract_text_from_image(file_path)
            
            if not text or not text.strip():
                print(f"All OCR methods failed for {file_path}")
                return []
                
            return [Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "type": "image",
                    "processing_method": "gemini_vision"
                }
            )]
            
        except Exception as e:
            print(f"Error processing image {file_path}: {str(e)}")
            return []

    def _load_unstructured(self, file_path: str) -> List[Document]:
        """Load other file types using UnstructuredFileLoader"""
        try:
            loader = UnstructuredFileLoader(file_path)
            return loader.load()
        except Exception as e:
            print(f"Failed to load {file_path} with UnstructuredFileLoader: {str(e)}")
            return []

    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """Process multiple documents"""
        all_docs = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
                
            print(f"Processing: {file_path}")
            docs = self.load_document(file_path)
            if docs:
                all_docs.extend(docs)
        
        # Split documents into chunks
        if all_docs:
            return self.text_splitter.split_documents(all_docs)
        return []
