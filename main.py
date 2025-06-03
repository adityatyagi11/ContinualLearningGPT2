import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
import json
import pickle
from pathlib import Path
import logging
import warnings
from collections import defaultdict
import math

#logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    memory_size: int = 100
    memory_dim: int = 768  # Default GPT-2 hidden size
    retrieval_top_k: int = 3
    update_threshold: float = 0.3
    fusion_weight: float = 0.08  
    memory_decay: float = 0.995
    similarity_threshold: float = 0.7
    consolidation_interval: int = 2000
    max_sequence_length: int = 128
    batch_size: int = 1
    gradient_clip: float = 1.0
    semantic_threshold: float = 0.6
    category_diversity_bonus: float = 0.05

class MemoryBank:
    
    
    def __init__(self, config: MemoryConfig, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"MemoryBank initialized on device: {self.device}")
        
        
        self.memory_vectors = torch.zeros(
            config.memory_size, config.memory_dim, device=self.device, dtype=torch.float32
        )
        self.memory_keys = torch.zeros(
            config.memory_size, config.memory_dim, device=self.device, dtype=torch.float32
        )
        self.memory_usage = torch.zeros(config.memory_size, device=self.device, dtype=torch.float32)
        self.memory_timestamps = torch.zeros(config.memory_size, device=self.device, dtype=torch.float32)
        self.memory_importance = torch.zeros(config.memory_size, device=self.device, dtype=torch.float32)
        self.memory_texts = [""] * config.memory_size
        self.memory_categories = [""] * config.memory_size
        
        # Tracking
        self.memory_index = 0
        self.current_time = 0
        self.access_count = defaultdict(int)
        self.category_counts = defaultdict(int)
        
        # Statistics
        self.hit_count = 0
        self.total_queries = 0
        
    def _ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        # For ensuring tensor is on the correct device
        if tensor is None:
            return tensor
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=torch.float32)
        if tensor.device != self.device:
            return tensor.to(self.device)
        return tensor
        
    def _compute_similarity(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        #Compute cosine similarity between query and keys
        query = self._ensure_device(query)
        keys = self._ensure_device(keys)
        
        #Ensure both tensors are 2D
        if query.dim() == 1:
            query = query.unsqueeze(0)
        if keys.dim() == 1:
            keys = keys.unsqueeze(0)
            
        #Normalize vectors
        query_norm = F.normalize(query, dim=-1, eps=1e-8)
        keys_norm = F.normalize(keys, dim=-1, eps=1e-8)
        
        #Compute similarity
        similarity = torch.matmul(query_norm, keys_norm.T)
        if similarity.dim() > 1:
            similarity = similarity.squeeze(0)
            
        return self._ensure_device(similarity)
    
    def _is_duplicate(self, new_key: torch.Tensor, text: str = "", threshold: float = None) -> Tuple[bool, int]:
        #For ensuring that memory does not contain duplicates
        if self.memory_index == 0:
            return False, -1

        threshold = threshold or self.config.similarity_threshold
        new_key = self._ensure_device(new_key)

        #Only check exact text matches for very short texts
        if text and len(text.strip()) > 20:  
            text_clean = text.lower().strip()
            for i in range(self.memory_index):
                stored_text = self.memory_texts[i].lower().strip()
                #Only flag as duplicate if texts are very similar
                if text_clean == stored_text or (len(text_clean) > 30 and stored_text in text_clean):
                    logger.info(f"Text duplicate detected at index {i}")
                    return True, i

        #Vector similarity check with higher threshold
        active_keys = self.memory_keys[:self.memory_index]
        similarities = self._compute_similarity(new_key, active_keys)
    
        if similarities.numel() > 0:
            max_similarity, best_idx = similarities.max(dim=0)
            similarity_value = max_similarity.item()
            logger.info(f"Max vector similarity: {similarity_value:.3f} (threshold: {threshold})")
        
            if similarity_value > threshold:
                logger.info(f"Vector duplicate detected at index {best_idx.item()}")
                return True, best_idx.item()

        return False, -1
    def debug_memory_state(self):
        """Debug current memory state"""
        print(f"\n=== Memory Bank Debug ===")
        print(f"Total memories stored: {self.memory_index}")
        print(f"Memory texts:")
        for i in range(self.memory_index):
            print(f"  [{i}] {self.memory_categories[i]}: {self.memory_texts[i][:50]}...")
            print(f"      Importance: {self.memory_importance[i].item():.3f}")
        print(f"Hit rate: {self.hit_count}/{self.total_queries}")
        print("========================\n")

    def retrieve_memories(self, query: torch.Tensor, top_k: int = None, query_text: str = "") -> Tuple[torch.Tensor, torch.Tensor, List[str], List[int]]:
        #retrieve top-k memories based on similarity to the query
        if top_k is None:
            top_k = self.config.retrieval_top_k
            
        self.total_queries += 1
        query = self._ensure_device(query)
        
        if self.memory_index == 0:
            empty_memories = torch.zeros(top_k, self.config.memory_dim, device=self.device)
            empty_scores = torch.zeros(top_k, device=self.device)
            empty_texts = [""] * top_k
            empty_indices = [-1] * top_k
            return empty_memories, empty_scores, empty_texts, empty_indices
        
        #Get active memories
        active_keys = self.memory_keys[:self.memory_index]
        active_vectors = self.memory_vectors[:self.memory_index]
        
        #Compute similarities
        similarities = self._compute_similarity(query, active_keys)
        
        #Simple scoring based on similarity and importance
        importance_weights = self.memory_importance[:self.memory_index]
        combined_scores = similarities * 0.7 + importance_weights * 0.3
        
        #Get top-k
        actual_k = min(top_k, self.memory_index)
        if combined_scores.numel() > 0:
            top_scores, top_indices = torch.topk(combined_scores, actual_k, dim=-1)
            
            #Update usage
            for idx in top_indices:
                self.memory_usage[idx] += 0.1
            
            retrieved_memories = active_vectors[top_indices]
            retrieved_texts = [self.memory_texts[idx.item()] for idx in top_indices]
            retrieved_indices = top_indices.tolist()
            
            if actual_k > 0:
                self.hit_count += 1
        else:
            retrieved_memories = torch.zeros(actual_k, self.config.memory_dim, device=self.device)
            top_scores = torch.zeros(actual_k, device=self.device)
            retrieved_texts = [""] * actual_k
            retrieved_indices = [-1] * actual_k
        
        #Pad if necessary
        if actual_k < top_k:
            padding_needed = top_k - actual_k
            zero_padding = torch.zeros(padding_needed, self.config.memory_dim, device=self.device)
            score_padding = torch.zeros(padding_needed, device=self.device)
            
            retrieved_memories = torch.cat([retrieved_memories, zero_padding], dim=0)
            top_scores = torch.cat([top_scores, score_padding])
            retrieved_texts.extend([""] * padding_needed)
            retrieved_indices.extend([-1] * padding_needed)
        
        return retrieved_memories, top_scores, retrieved_texts, retrieved_indices
    
    def update_memory(self, key: torch.Tensor, value: torch.Tensor, 
                 improvement: float, context_text: str = "", category: str = "") -> bool:
        key = self._ensure_device(key).detach()
        value = self._ensure_device(value).detach()
    
        #Debug logging
        logger.info(f"Attempting to store: '{context_text[:50]}...' (category: {category})")
    
        #Check for duplicates with more specific logging
        is_duplicate, duplicate_idx = self._is_duplicate(key, context_text)
    
        if is_duplicate and duplicate_idx >= 0:
            logger.info(f"Found duplicate at index {duplicate_idx}, updating existing memory")
            #Update existing memory
            alpha = 0.2
            self.memory_vectors[duplicate_idx] = (1-alpha) * self.memory_vectors[duplicate_idx] + alpha * value
            self.memory_keys[duplicate_idx] = (1-alpha) * self.memory_keys[duplicate_idx] + alpha * key
            self.memory_importance[duplicate_idx] += improvement * 0.5
            self.memory_timestamps[duplicate_idx] = self.current_time
        
            if len(context_text) > len(self.memory_texts[duplicate_idx]):
                self.memory_texts[duplicate_idx] = context_text
            
            return True
    
        #Check improvement threshold
        if improvement <= self.config.update_threshold:
            logger.info(f"Improvement {improvement} below threshold {self.config.update_threshold}, skipping")
            return False
    
        #Find slot for new memory
        if self.memory_index < self.config.memory_size:
            slot_idx = self.memory_index
            self.memory_index += 1
            logger.info(f"Using new slot {slot_idx}, memory_index now {self.memory_index}")
        else:
            #Replace least important memory
            importance_scores = self.memory_importance[:self.memory_index]
            slot_idx = torch.argmin(importance_scores).item()
            logger.info(f"Replacing memory at slot {slot_idx} (least important)")
    
        #Store new memory with detailed logging
        self.memory_keys[slot_idx] = key
        self.memory_vectors[slot_idx] = value
        self.memory_importance[slot_idx] = improvement
        self.memory_usage[slot_idx] = 1.0
        self.memory_timestamps[slot_idx] = self.current_time
        self.memory_texts[slot_idx] = context_text
        self.memory_categories[slot_idx] = category
    
        if category:
            self.category_counts[category] += 1
    
        self.current_time += 1
    
        logger.info(f"Successfully stored memory at index {slot_idx}")
        logger.info(f"Memory bank now contains {self.memory_index} memories")
    
        return True
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics"""
        hit_rate = self.hit_count / max(self.total_queries, 1)
        avg_importance = self.memory_importance[:self.memory_index].mean().item() if self.memory_index > 0 else 0
        
        return {
            'memory_count': self.memory_index,
            'hit_rate': hit_rate,
            'avg_importance': avg_importance,
            'total_queries': self.total_queries,
            'category_distribution': dict(self.category_counts)
        }

class SimpleMemoryAttention(nn.Module):
    
    def __init__(self, hidden_size: int, memory_config: MemoryConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_config = memory_config
        
        #Simple query generation
        self.query_proj = nn.Linear(hidden_size, memory_config.memory_dim)
        
        #Memory integration
        self.memory_proj = nn.Linear(memory_config.memory_dim, hidden_size)
        
        #Gate for controlling memory influence
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        #Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor, memory_bank: MemoryBank, 
                input_text: str = "") -> torch.Tensor:
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        #Move to correct device if needed
        if not all(p.device == device for p in self.parameters()):
            self.to(device)
        
        enhanced_states = hidden_states.clone()
        
        #Process each batch item
        for b in range(batch_size):
            #Use the last token's hidden state as query
            query_hidden = hidden_states[b, -1]  
            memory_query = self.query_proj(query_hidden) 
            
            #Retrieve memories
            memories, scores, texts, indices = memory_bank.retrieve_memories(memory_query, query_text=input_text)
            
            #Check if we have meaningful memories
            if scores.sum() > 0.01:
                #Weight memories by scores
                valid_mask = scores > 0.01
                if valid_mask.any():
                    valid_memories = memories[valid_mask]
                    valid_scores = scores[valid_mask]
                    
                    #Weighted average of memories
                    weights = F.softmax(valid_scores, dim=0)
                    memory_context = torch.sum(valid_memories * weights.unsqueeze(-1), dim=0)
                    
                    #Project memory to hidden size
                    memory_output = self.memory_proj(memory_context)  
                    
                    #Apply gating only to the last few tokens
                    for t in range(max(0, seq_len - 3), seq_len):  #Last 3 tokens
                        original = enhanced_states[b, t]
                        
                       
                        gate_input = torch.cat([original, memory_output])
                        gate_weight = self.gate(gate_input) * self.memory_config.fusion_weight
                        
                        #Apply memory influence
                        enhanced_states[b, t] = original + gate_weight * memory_output
        
        #Layer normalization
        enhanced_states = self.layer_norm(enhanced_states)
        
        return enhanced_states

class ContinualLearningGPT2(nn.Module):
    def __init__(self, model_name: str = 'gpt2', memory_config: MemoryConfig = None):
        super().__init__()
        
        self.memory_config = memory_config or MemoryConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #Load base model
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.config = self.gpt2.config
        
        #Ensure memory_dim matches hidden_size
        self.memory_config.memory_dim = self.config.hidden_size
        
        #Initialize memory components
        self.memory_bank = MemoryBank(self.memory_config, device=self.device)
        self.memory_attention = SimpleMemoryAttention(self.config.hidden_size, self.memory_config)
        
        #memory encoder 
        self.memory_encoder = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.Tanh()
        )
        
        #Add a content aware pooling mechanism
        self.content_attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        #Move everything to device
        self.to(self.device)
    
    def _extract_content_representation(self, hidden_states: torch.Tensor, 
                                      attention_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if attention_mask is not None:
            #Use attention mask to focus on actual content
            expanded_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            masked_hidden = hidden_states * expanded_mask.float()
            
            #Get sequence lengths
            seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
            seq_lengths = torch.clamp(seq_lengths, min=1.0)
            
            #Weighted average based on position (later tokens get more weight)
            position_weights = torch.arange(seq_len, device=self.device).float()
            position_weights = position_weights.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
            position_weights = position_weights * attention_mask.unsqueeze(-1).float()
            
            #Normalize position weights
            weight_sum = position_weights.sum(dim=1, keepdim=True)
            weight_sum = torch.clamp(weight_sum, min=1.0)
            position_weights = position_weights / weight_sum
            
            #Weighted sum
            content_repr = (masked_hidden * position_weights).sum(dim=1)
        else:
            #Simple mean pooling fallback
            content_repr = hidden_states.mean(dim=1)
        
        return content_repr
        
    def _calculate_improvement_score(self, loss: float, text: str, category: str) -> float:
        """Calculate improvement score for memory update"""
        #Simple loss-based scoring
        base_score = max(0.1, min(2.0, loss))
        
        #Category bonus
        category_weights = {
            'geography': 1.2, 'science': 1.3, 'programming': 1.1, 
            'literature': 1.1, 'history': 1.1, 'astronomy': 1.2
        }
        category_multiplier = category_weights.get(category.lower(), 1.0)
        
        return base_score * category_multiplier
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, use_memory: bool = True, 
                input_text: str = "") -> Dict:
        
        #Ensure inputs are on correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        #Get GPT-2 hidden states
        transformer_outputs = self.gpt2.transformer(
            input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = transformer_outputs.last_hidden_state
        
        #Store original for comparison
        original_hidden = hidden_states.clone()
        
        #Apply memory augmentation with much lighter touch
        if use_memory and self.memory_bank.memory_index > 0:
            #Only apply memory to a subset of the sequence
            memory_enhanced = self.memory_attention(hidden_states, self.memory_bank, input_text)
            #Blend very conservatively
            alpha = 0.1  #Very small memory influence
            hidden_states = (1 - alpha) * hidden_states + alpha * memory_enhanced
        
        #Generate logits using the (possibly) enhanced hidden states
        logits = self.gpt2.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states,
            'original_hidden_states': original_hidden
        }
    
    def learn_from_example(self, text: str, tokenizer, device, category: str = "") -> Dict[str, float]:
        """Enhanced learning with better content representation"""
        try:
            #Tokenize
            encoding = tokenizer(
                text, 
                return_tensors='pt', 
                max_length=self.memory_config.max_sequence_length,
                truncation=True, 
                padding=True
            )
        
            for key in encoding:
                encoding[key] = encoding[key].to(self.device)
        
            with torch.no_grad():
                #Get baseline performance
                outputs_baseline = self.forward(
                    encoding['input_ids'], 
                    attention_mask=encoding['attention_mask'],
                    labels=encoding['input_ids'], 
                    use_memory=False,
                    input_text=text
                )
                baseline_loss = outputs_baseline['loss'].item() if outputs_baseline['loss'] is not None else 2.0
            
                #Extract better content representation
                hidden_states = outputs_baseline['original_hidden_states']
                content_repr = self._extract_content_representation(
                    hidden_states, encoding['attention_mask']
                )
            
                #Enhanced memory encoding with more randomness/distinctiveness
                memory_repr = self.memory_encoder(content_repr.squeeze(0))
                
                #Add category specific noise to make representations more distinct
                category_seed = hash(category) % 1000
                torch.manual_seed(category_seed)
                category_noise = torch.randn_like(memory_repr) * 0.01
                memory_repr = memory_repr + category_noise
            
                #Calculate improvement score
                improvement = self._calculate_improvement_score(baseline_loss, text, category)
            
                logger.info(f"Learning - Category: {category}, Loss: {baseline_loss:.4f}, Improvement: {improvement:.4f}")
                
                #Debug: Print memory representation stats
                logger.info(f"Memory repr stats - Mean: {memory_repr.mean().item():.4f}, "
                           f"Std: {memory_repr.std().item():.4f}, "
                           f"Norm: {memory_repr.norm().item():.4f}")
            
                #Add to memory
                added = self.memory_bank.update_memory(
                    memory_repr, memory_repr.clone(), improvement, text, category
                )
            
                return {
                    'added': added,
                    'improvement': improvement,
                    'baseline_loss': baseline_loss,
                    'memory_count': self.memory_bank.memory_index,
                    'category': category,
                }
        except Exception as e:
            logger.error(f"Learning failed for text '{text[:50]}...': {e}")
            return {'added': False, 'error': str(e)}
    
    @torch.no_grad()
    def generate_with_memory(self, input_ids: torch.Tensor, tokenizer, 
                           max_length: int = 50, temperature: float = 0.8, 
                           top_p: float = 0.9, do_sample: bool = True) -> torch.Tensor:
        """Generate text with memory"""
        
        self.eval()
        input_ids = input_ids.to(self.device)
        current_ids = input_ids.clone()
        
        try:
            for step in range(max_length - input_ids.size(1)):
                #Forward pass with memory
                outputs = self.forward(current_ids, use_memory=True)
                logits = outputs['logits']
                
                #Get next token logits
                next_token_logits = logits[0, -1, :] / max(temperature, 0.1)
                
                if do_sample:
                    #Apply top-p sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = -float('inf')
                    
                    #Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    #Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                #Append to sequence
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
                
                #Stop if EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            return current_ids
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return current_ids

def test_fixed_system():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        #Initialize tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        
        #Enhanced configuration with higher similarity threshold
        memory_config = MemoryConfig(
            memory_size=50,
            retrieval_top_k=3,
            update_threshold=0.3,
            fusion_weight=0.05,
            similarity_threshold=0.99,  #Much higher - only nearly identical vectors are duplicates
            max_sequence_length=64
        )
        
        #Initialize model with enhanced encoder
        model = ContinualLearningGPT2('gpt2', memory_config)
        
        #Freeze base model parameters to prevent catastrophic forgetting
        for param in model.gpt2.parameters():
            param.requires_grad = False
        
        logger.info(f"Model initialized on device: {model.device}")
        
        #Training examples
        training_examples = [
            ("geography", "Paris is the capital city of France."),
            ("programming", "Python is a programming language used for AI."),
            ("astronomy", "The Earth orbits around the Sun."),
            ("physics", "Water freezes at 0 degrees Celsius."),
            ("literature", "Shakespeare wrote Romeo and Juliet."),
            ("science", "Einstein developed the theory of relativity."),
        ]
        
        #Learning phase
        logger.info("\n--- Learning Phase ---")
        successful_learns = 0
        
        for category, text in training_examples:
            result = model.learn_from_example(text, tokenizer, device, category)
            if result.get('added', False):
                successful_learns += 1
                logger.info(f"✓ [{category}] Learned: {text}")
            else:
                logger.info(f"✗ [{category}] Skipped: {text}")
        
        logger.info(f"Successfully learned {successful_learns}/{len(training_examples)} examples")
        
        #Memory statistics
        stats = model.memory_bank.get_memory_stats()
        logger.info(f"Memory stats: {stats}")
        
        #Test generation
        logger.info("\n--- Testing Generation ---")
        test_prompts = [
            "The capital of France is",
            "Python programming language",
            "Shakespeare wrote",
            "Einstein's theory"
        ]
        
        for prompt in test_prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            #Generate with memory
            generated_ids = model.generate_with_memory(
                input_ids, tokenizer, 
                max_length=input_ids.size(1) + 15,
                temperature=0.7, 
                top_p=0.9
            )
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            logger.info(f"Prompt: '{prompt}'")
            logger.info(f"Generated: '{generated_text}'")
            logger.info("")

            #Debug memory state
        model.memory_bank.debug_memory_state()

        #Test memory retrieval specifically
        print("\n--- Testing Memory Retrieval ---")
        test_query = "What is the capital of France?"
        encoding = tokenizer(test_query, return_tensors='pt', max_length=64, truncation=True)
        with torch.no_grad():
            outputs = model.forward(encoding['input_ids'].to(device), use_memory=True)
            query_hidden = outputs['original_hidden_states'].mean(dim=1).squeeze(0)
            memory_repr = model.memory_encoder(query_hidden)
    
            memories, scores, texts, indices = model.memory_bank.retrieve_memories(memory_repr, query_text=test_query)
            print(f"Retrieved {len([s for s in scores if s > 0.01])} relevant memories")
            for i, (score, text, idx) in enumerate(zip(scores, texts, indices)):
                if score > 0.01:
                    print(f"  Memory {i}: Score={score:.3f}, Text='{text[:50]}...', Index={idx}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    try:
        model, tokenizer = test_fixed_system()
        logger.info("\n Continual learning system test completed successfully!")
       
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
