import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMManager:
    """Manages the local LLM (Mistral-7B-Instruct) for question answering."""
    
    def __init__(self, model_name: str = config.MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.embedding_model = None
        
        # Create cache directory
        Path(config.CACHE_DIR).mkdir(exist_ok=True)
        
    def load_models(self) -> None:
        """Load the LLM and embedding models."""
        logger.info(f"Loading models on device: {self.device}")
        
        try:
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=config.CACHE_DIR,
                trust_remote_code=True
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ) if self.device == "cuda" else None
            
            # Load model
            logger.info("Loading LLM model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=config.CACHE_DIR,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # Optimize memory usage
                use_flash_attention_2=False  # Disable for compatibility
            )
            
            # Load embedding model
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer(
                config.EMBEDDING_MODEL,
                cache_folder=config.CACHE_DIR
            )
            
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def generate_response(self, prompt: str, max_new_tokens: int = config.MAX_NEW_TOKENS) -> str:
        """Generate response from the LLM."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        try:
            logger.info(f"Generating response for prompt length: {len(prompt)} chars")
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096 - max_new_tokens,
                padding=True
            ).to(self.device)
            
            logger.info(f"Input tokens: {inputs['input_ids'].shape[1]}")
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Reduce repetition
                    use_cache=True,  # Use key-value cache for speed
                    stopping_criteria=None  # Let it generate naturally
                )
            
            logger.info(f"Generated tokens: {outputs.shape[1]}")
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Clean up response - remove any remaining special tokens
            response = response.replace("<|im_end|>", "").strip()
            
            logger.info(f"Response length: {len(response)} chars")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your question."
    
    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Generate embeddings for texts."""
        if self.embedding_model is None:
            raise ValueError("Embedding model not loaded.")
        
        try:
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def create_rag_prompt(self, question: str, context_chunks: List[str]) -> str:
        """Create a RAG prompt with context and question."""
        context = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])
        
        # OpenHermes format (ChatML style)
        prompt = f"""<|im_start|>system
You are a helpful AI assistant. Answer the question based ONLY on the provided context. If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."<|im_end|>
<|im_start|>user
Context:
{context}

Question: {question}<|im_end|>
<|im_start|>assistant
"""
        
        return prompt
