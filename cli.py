import os
import argparse
from rag_system import RAGSystem
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not found.")
        print("Please create a .env file with your OpenAI API key.")
        return
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="RAG System CLI")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Add document command
    add_parser = subparsers.add_parser('add', help='Add documents to the knowledge base')
    add_parser.add_argument('files', nargs='+', help='Files to add to the knowledge base')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the knowledge base')
    query_parser.add_argument('question', help='Question to ask the knowledge base')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear the knowledge base')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute commands
    if args.command == 'add':
        print(f"Processing {len(args.files)} files...")
        rag.add_documents(args.files)
        print("Documents added successfully!")
        
    elif args.command == 'query':
        if not args.question:
            print("Error: Please provide a question.")
            return
            
        print("\nProcessing your question...")
        result = rag.query(args.question)
        
        print("\nAnswer:")
        print(result["answer"])
        
        if result["sources"]:
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source}")
    
    elif args.command == 'clear':
        confirm = input("Are you sure you want to clear the knowledge base? (y/n): ")
        if confirm.lower() == 'y':
            rag.clear_knowledge_base()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
