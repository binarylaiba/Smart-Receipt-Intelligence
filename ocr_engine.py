"""OCR Engine Module for AI Receipt Analyser.

This module provides image preprocessing pipelines tailored for receipts
and text extraction using EasyOCR.
"""

from __future__ import annotations

import io
from typing import Tuple, Union, Optional
import cv2
import numpy as np
from PIL import Image
import easyocr


# Global cached instance of EasyOCR Reader for efficiency
_EASYOCR_READER: Optional[easyocr.Reader] = None


def get_ocr_reader(lang_list: Optional[list[str]] = None, gpu: bool = True) -> easyocr.Reader:
    """Retrieve or initialize a singleton instance of the EasyOCR Reader.

    Args:
        lang_list: List of language codes to load (defaults to ['en']).
        gpu: Whether to enable GPU acceleration if available.

    Returns:
        easyocr.Reader: Initialized EasyOCR reader.
    """
    global _EASYOCR_READER
    if lang_list is None:
        lang_list = ["en"]

    if _EASYOCR_READER is None:
        try:
            _EASYOCR_READER = easyocr.Reader(lang_list, gpu=gpu)
        except Exception:
            # Fallback to CPU if GPU initialization fails
            _EASYOCR_READER = easyocr.Reader(lang_list, gpu=False)

    return _EASYOCR_READER


def preprocess_image(
    image_input: Union[str, bytes, io.BytesIO, Image.Image, np.ndarray],
    block_size: int = 15,
    c_constant: int = 8,
) -> Tuple[np.ndarray, Image.Image]:
    """Preprocess receipt images to enhance OCR character recognition.

    The pipeline performs:
    1. Conversion to Grayscale
    2. Bilateral Filtering (d=9, sigmaColor=75, sigmaSpace=75) to eliminate
       background paper texture / noise while preserving crisp character edges.
    3. Adaptive Gaussian Thresholding to normalize uneven mobile lighting and shadows.

    Args:
        image_input: Filepath (str), raw image bytes, BytesIO buffer,
                     PIL Image, or NumPy ndarray.
        block_size: Size of a pixel neighborhood used to calculate threshold value (must be odd, > 1).
        c_constant: Constant subtracted from the mean or weighted mean.

    Returns:
        Tuple[np.ndarray, Image.Image]:
            - Preprocessed binary image as a 2D NumPy array (uint8).
            - Preprocessed binary image as a PIL Image object (mode 'L').

    Raises:
        ValueError: If input format is invalid or image cannot be decoded.
        RuntimeError: If OpenCV image processing fails.
    """
    try:
        # Step 1: Normalize input to OpenCV BGR / Grayscale ndarray
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Unable to read image from path: {image_input}")
        elif isinstance(image_input, (bytes, bytearray)):
            np_arr = np.frombuffer(image_input, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image from provided bytes.")
        elif isinstance(image_input, io.BytesIO):
            image_input.seek(0)
            np_arr = np.frombuffer(image_input.read(), np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to decode image from BytesIO stream.")
        elif isinstance(image_input, Image.Image):
            rgb_img = image_input.convert("RGB")
            image = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Step 2: Convert to Grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif len(image.shape) == 2:
            gray = image
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        # Step 3: Bilateral Filtering (d=9, sigmaColor=75, sigmaSpace=75)
        # Smooths paper texture and noise while keeping character edges sharp
        filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Step 4: Adaptive Gaussian Thresholding for handling uneven illumination / shadows
        # Ensures block size is odd and >= 3
        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3

        binarized = cv2.adaptiveThreshold(
            filtered,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=c_constant,
        )

        # Step 5: Convert NumPy array to PIL Image
        pil_image = Image.fromarray(binarized, mode="L")

        return binarized, pil_image

    except Exception as exc:
        if isinstance(exc, (ValueError, TypeError)):
            raise
        raise RuntimeError(f"Error during image preprocessing: {exc}") from exc


def extract_raw_text(
    preprocessed_image: Union[np.ndarray, Image.Image, str],
    reader: Optional[easyocr.Reader] = None,
    detail: int = 0,
    paragraph: bool = False,
) -> str:
    """Extract and join raw text line-by-line from a preprocessed receipt image using EasyOCR.

    Args:
        preprocessed_image: 2D/3D NumPy array, PIL Image, or path to binarized image.
        reader: Optional pre-configured EasyOCR Reader instance. If None,
                the default singleton English reader is used.
        detail: EasyOCR detail level (0 returns text strings only, 1 returns bounding boxes + confidences).
        paragraph: Whether to combine nearby text into paragraphs.

    Returns:
        str: Extracted receipt text joined line-by-line.

    Raises:
        ValueError: If input image is empty or invalid.
        RuntimeError: If OCR extraction encounters an error.
    """
    try:
        # Ensure OCR reader is available
        ocr_reader = reader if reader is not None else get_ocr_reader()

        # Convert PIL Image to NumPy array if necessary
        if isinstance(preprocessed_image, Image.Image):
            img_matrix = np.array(preprocessed_image)
        elif isinstance(preprocessed_image, np.ndarray):
            img_matrix = preprocessed_image
        elif isinstance(preprocessed_image, str):
            img_matrix = cv2.imread(preprocessed_image, cv2.IMREAD_GRAYSCALE)
            if img_matrix is None:
                raise ValueError(f"Unable to read image from path: {preprocessed_image}")
        else:
            raise TypeError(f"Unsupported image input type: {type(preprocessed_image)}")

        if img_matrix.size == 0:
            raise ValueError("Provided image matrix is empty.")

        # Run OCR detection
        results = ocr_reader.readtext(
            img_matrix,
            detail=detail,
            paragraph=paragraph,
        )

        # Process extracted results
        if detail == 0:
            # results is a list of strings
            raw_text = "\n".join([str(line).strip() for line in results if str(line).strip()])
        else:
            # results is a list of tuples: (bbox, text, confidence)
            raw_text = "\n".join([str(item[1]).strip() for item in results if len(item) > 1 and str(item[1]).strip()])

        return raw_text

    except Exception as exc:
        if isinstance(exc, (ValueError, TypeError)):
            raise
        raise RuntimeError(f"Error during OCR text extraction: {exc}") from exc


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"--- Testing OCR Engine on: {test_file} ---")
        try:
            np_img, pil_img = preprocess_image(test_file)
            print(f"Preprocessed successfully! Shape: {np_img.shape}, PIL size: {pil_img.size}")
            
            print("Extracting text with EasyOCR...")
            extracted_text = extract_raw_text(np_img)
            print("\n=== Extracted Receipt Text ===")
            print(extracted_text)
            print("==============================")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
    else:
        print("ocr_engine.py is ready. Run with an image path to test: python ocr_engine.py <path_to_receipt_image>")
