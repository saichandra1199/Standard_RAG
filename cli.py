import os
import sys
import argparse
import warnings
from rag_system import RAGSystem
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class RAGCLI:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY environment variable not found.")
            print("Please create a .env file with your OpenAI API key.")
            sys.exit(1)
        
        # Initialize RAG system
        self.rag = RAGSystem()
    
    def add_documents(self, files):
        print(f"Processing {len(files)} files...")
        self.rag.add_documents(files)
        print("✓ Documents added successfully!")
    
    def query(self, question):
        result = self.rag.query(question)
        print("\nAnswer:")
        print("-" * 50)
        print(result["answer"])
        print("-" * 50)
        
        if result["sources"]:
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source}")
        print()
    
    def clear(self):
        confirm = input("⚠️  Are you sure you want to clear the knowledge base? (y/n): ")
        if confirm.lower() == 'y':
            self.rag.clear_knowledge_base()
            print("✓ Knowledge base cleared!")
    
    def interactive_shell(self):
        print("\n" + "="*60)
        print("🤖 RAG System - Interactive Shell")
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'clear' to clear the knowledge base")
        print("Type 'add <file1> <file2> ...' to add documents")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\n❓ Your question: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ('exit', 'quit'):
                    print("👋 Goodbye!")
                    break
                    
                if user_input.lower() == 'clear':
                    self.clear()
                    continue
                    
                if user_input.startswith('add '):
                    files = user_input[4:].split()
                    if files:
                        self.add_documents(files)
                    else:
                        print("Please specify files to add")
                    continue
                
                # Process as a query
                print("\nProcessing...")
                self.query(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="RAG System CLI")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Interactive mode (default)
    parser.set_defaults(func=lambda args: RAGCLI().interactive_shell())
    
    # Add document command
    add_parser = subparsers.add_parser('add', help='Add documents to the knowledge base')
    add_parser.add_argument('files', nargs='+', help='Files to add to the knowledge base')
    add_parser.set_defaults(func=lambda args: RAGCLI().add_documents(args.files))
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the knowledge base')
    query_parser.add_argument('question', help='Question to ask the knowledge base')
    query_parser.set_defaults(func=lambda args: RAGCLI().query(args.question))
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear the knowledge base')
    clear_parser.set_defaults(func=lambda args: RAGCLI().clear())
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate function
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
