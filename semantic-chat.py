import json
import os
import yaml
import numpy as np
from ollama import Client
from datetime import datetime
from typing import List, Dict, Any

class SemanticMemory:
    def __init__(self, model='all-minilm', memory_file='semantic_memory.json'):
        self.client = Client()
        self.model = model
        self.memory_file = memory_file
        self.memories = self.load_memories()
    
    def load_memories(self) -> List[Dict[str, Any]]:
        """Load existing memories from file."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading semantic memories: {e}")
                return []
        return []
    
    def save_memories(self):
        """Save memories to file."""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memories, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving semantic memories: {e}")
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding vector for text using Ollama."""
        try:
            response = self.client.embeddings(model=self.model, prompt=text)
            return response['embedding']
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return []
    
    def add_memory(self, text: str, metadata: Dict[str, Any] = None):
        """Add a new memory with its embedding."""
        embedding = self.create_embedding(text)
        if not embedding:
            return False
        
        memory = {
            'text': text,
            'embedding': embedding,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.memories.append(memory)
        self.save_memories()
        return True
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def search_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar memories using semantic similarity."""
        if not self.memories:
            return []
        
        query_embedding = self.create_embedding(query)
        if not query_embedding:
            return []
        
        # memory embedding math
        similarities = []
        for i, memory in enumerate(self.memories):
            if memory['text'] != query: 
                similarity = self.cosine_similarity(query_embedding, memory['embedding'])
                similarities.append((i, similarity))
        
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
       
        results = []
        for i, similarity in similarities[:top_k]:
            memory = self.memories[i].copy()
            memory['similarity'] = similarity
            memory.pop('embedding', None)
            results.append(memory)
        
        return results

def load_conversation_history(filename='chat_history.json'):
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                history = json.load(f)
             
                for msg in history:
                    msg['role'] = msg['role'].lower()
                
                # Save the updated history back to file
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
                
                return history
        return []
    except Exception as e:
        print(f"Error loading conversation history: {e}")
        return []

def load_system_prompt(filename='prompt.yaml'):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            prompt_data = yaml.safe_load(f)
          
            self_code = get_self_code()
            if self_code:
                codebase = prompt_data['system']['codebase'].replace('{codebase}', self_code)
            else:
                codebase = ''
            
            system_message = {
                'role': 'system',
                'content': f"{prompt_data['system']['role']}\n\n{prompt_data['system']['description']}\n\n{codebase}\n\n{prompt_data['system']['instructions']}"
            }
            return system_message
    except Exception as e:
        print(f"Error loading system prompt: {e}")
        return None

def save_conversation_history(conversation, filename='chat_history.json'):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving conversation history: {e}")

def get_recent_conversation(conversation, max_messages=10):
    """Get the most recent messages while keeping the system prompt."""
    if not conversation:
        return []
    
    # Always keep the system prompt if it exists
    if conversation and conversation[0]['role'] == 'system':
        recent = [conversation[0]] + conversation[-max_messages:]
    else:
        recent = conversation[-max_messages:]
    
    
    formatted_messages = []
    for msg in recent:
      
        role = msg['role'].lower()
        api_role = 'user' if role == 'user' else 'assistant' if role in ['assistant', 'nora'] else role
        
        if api_role in ['user', 'assistant', 'system', 'tool']:
            formatted_msg = {
                'role': api_role,
                'content': msg['content'].strip()
            }
            formatted_messages.append(formatted_msg)
    
    return formatted_messages

def get_self_code():
    """Get the contents of semantic-chat.py to provide self-awareness of implementation."""
    try:
        with open(__file__, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading own source code: {e}")
        return None


def chat_loop():
    client = Client()
    print("Chat started! (Type 'quit' to exit, 'memory search <query>' to search memories)")

    conversation = load_conversation_history()
    if conversation:
        for msg in conversation:
            if msg['role'] != 'system':
                display_role = 'User' if msg['role'] == 'user' else 'Nora'
                print(f"\n{display_role}: {msg['content']}")
    
    system_message = load_system_prompt()
    if system_message and (not conversation or conversation[0].get('role') != 'system'):
        conversation.insert(0, system_message)
    
    memory = SemanticMemory()
    
    # ANSI color codes
    PINK = "\033[38;2;255;192;203m" 
    CYAN = "\033[36m"  
    
    while True:
        try:
            
            user_input = input(f"\n{PINK}You: {PINK}").strip()
            print(end='') 
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye', 'adios',]:
                print("Goodbye!")
                save_conversation_history(conversation)
                break
            
            
            if user_input.lower().startswith('memory search '):
                query = user_input[14:].strip()
                results = memory.search_memories(query)
                print("\nMemory search results:")
                for i, result in enumerate(results):
                    print(f"{i+1}. [{result['similarity']:.4f}] {result['text']}")
                continue
            
           
            memory.add_memory(user_input, {'role': 'user', 'timestamp': datetime.now().isoformat()})
            
            conversation.append({
                'role': 'user',
                'content': user_input
            })
            

            relevant_memories = memory.search_memories(user_input, top_k=4)
            
            # Add memory context to system message if there are relevant memories
            # TODO : let's seperate out the priors as seperate headers, so longterm memories are not group with the previous conversation , instead as "Relevent Priors:"
            recent_conversation = get_recent_conversation(conversation)
            if relevant_memories:
                memory_context = "\n\nRelevant previous conversations:\n"
                for i, mem in enumerate(relevant_memories):
                    memory_context += f"{i+1}. {mem['text']}\n"
                
                # Clone the system message and add memory context
                system_with_memory = recent_conversation[0].copy() if recent_conversation and recent_conversation[0]['role'] == 'system' else {
                    'role': 'system', 
                    'content': 'You are Nora, a helpful assistant.'
                }
                system_with_memory['content'] += memory_context
                
                # Replace system message with enhanced version
                if recent_conversation and recent_conversation[0]['role'] == 'system':
                    recent_conversation[0] = system_with_memory
                else:
                    recent_conversation.insert(0, system_with_memory)
            
            # Let's add a print of what the ollama agent sees, the memory, and system context
            print("\n=== Agent Context ===")
            print(json.dumps(recent_conversation, indent=2))
            print("==================\n")
            
    
            import sys
            print(f"\n{CYAN}Nora is thinking...", flush=True)
            sys.stdout.flush()
            
            response = client.chat(model='qwen2.5:3b', messages=recent_conversation)
 
            print("\033[A\033[K", end='') 
            print("\nNora: ", end='')
            
            assistant_message = response.message.content
            print(assistant_message)
            
            memory.add_memory(assistant_message, {'role': 'assistant', 'timestamp': datetime.now().isoformat()})
            
            conversation.append({
                'role': 'assistant',
                'content': assistant_message
            })
            
            save_conversation_history(conversation)
            
        except Exception as e:
            print("\nError occurred during chat:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nPlease try again.")

if __name__ == "__main__":
    chat_loop()