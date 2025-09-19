"""
Machine Translation (MT) module for the subtitle translator package.
"""
from typing import List, Dict, Optional, Union, Any
import threading
import time
from dataclasses import dataclass

from rich.console import Console

from .asr import Segment

console = Console()

class TranslationEngine:
    """Handles translation of subtitle segments using various MT backends"""
    
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "cuda",
        batch_size: int = 8,
        max_length: int = 200,
        beam_size: int = 5,
    ):
        """Initialize the translation engine
        
        Args:
            model_name: Name or path of the translation model to use
            device: Device to run on ("cuda" or "cpu")
            batch_size: Batch size for translation
            max_length: Maximum output sequence length
            beam_size: Beam size for decoding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.beam_size = beam_size
        
        # Lazy loading - model will be initialized on first use
        self._model = None
        self._tokenizer = None
        
    def _ensure_model_loaded(self):
        """Ensure the model and tokenizer are loaded"""
        if self._model is None:
            try:
                # Lazy import to avoid dependency if not used
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                
                console.print(f"Loading translation model: {self.model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                
                # Move to appropriate device
                if self.device == "cuda" and torch.cuda.is_available():
                    self._model = self._model.half().to("cuda")
                else:
                    if self.device == "cuda" and not torch.cuda.is_available():
                        console.print("[yellow]CUDA requested but not available. Using CPU instead.[/yellow]")
                    self._model = self._model.to("cpu")
                    
            except ImportError:
                console.print("[yellow]Cannot load translation model: torch or transformers not installed.[/yellow]")
                console.print("[yellow]Install optional dependencies with: pip install torch transformers[/yellow]")
                raise
            except Exception as e:
                console.print(f"[red]Failed to load translation model: {e}[/red]")
                raise
    
    def translate_segments(
        self,
        segments: List[Segment],
        source_lang: str,
        target_lang: str,
        progress_callback: Optional[Any] = None,
    ) -> List[Segment]:
        """Translate a list of subtitle segments
        
        Args:
            segments: List of segments to translate
            source_lang: Source language code
            target_lang: Target language code
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of translated segments
        """
        try:
            self._ensure_model_loaded()
            
            # Prepare batches
            batches = [segments[i:i+self.batch_size] for i in range(0, len(segments), self.batch_size)]
            result = []
            
            # Process each batch
            for i, batch in enumerate(batches):
                texts = [seg.text for seg in batch]
                
                # Tokenize
                inputs = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Add language tokens if using a multilingual model like NLLB or M2M100
                if "nllb" in self.model_name.lower():
                    # NLLB style forced BOS token
                    self._tokenizer.src_lang = source_lang
                    forced_bos_token_id = self._tokenizer.lang_code_to_id[target_lang]
                elif "m2m" in self.model_name.lower():
                    # M2M100 style
                    self._tokenizer.src_lang = source_lang
                    forced_bos_token_id = self._tokenizer.get_lang_id(target_lang)
                else:
                    forced_bos_token_id = None
                
                # Generate translations
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=self.max_length,
                        num_beams=self.beam_size,
                        early_stopping=True,
                    )
                
                # Decode translations
                translations = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Create new segments with translations
                for j, (seg, translation) in enumerate(zip(batch, translations)):
                    new_seg = Segment(
                        id=seg.id,
                        start=seg.start,
                        end=seg.end,
                        text=translation.strip(),
                        source_text=seg.text,
                    )
                    result.append(new_seg)
                    
                    # Update progress if callback provided
                    if progress_callback:
                        progress_callback(i * self.batch_size + j + 1, len(segments))
            
            return result
            
        except Exception as e:
            # Fallback to internal Whisper translation if external MT fails
            console.print(f"[yellow]Translation failed: {e}. Use Whisper's internal translation instead.[/yellow]")
            
            # Simply return the original segments as fallback
            # In a real implementation, you might want to call Whisper's translate functionality
            return segments
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single text string
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        segments = [Segment(id=1, start=0.0, end=1.0, text=text)]
        translated = self.translate_segments(segments, source_lang, target_lang)
        return translated[0].text if translated else text