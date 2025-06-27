#!/usr/bin/env python3
"""
Command-line interface for the Advanced RAG AI System.
This provides a simple CLI for testing the RAG functionality.
"""

import argparse
import sys
from pathlib import Path
from rag_system import RAGSystem

def main():
    parser = argparse.ArgumentParser(description="Advanced RAG AI System CLI")
    parser.add_argument("--add-docs", nargs="+", help="Add documents to the system")
    parser.add_argument("--query", type=str, help="Query the system with a question")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    parser.add_argument("--clear", action="store_true", help="Clear the database")
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Initialize RAG system
    print("🤖 Initializing RAG system...")
    try:
        rag_system = RAGSystem()
        rag_system.initialize()
        print("✅ System initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        return
    
    # Handle commands
    if args.add_docs:
        print(f"📚 Adding {len(args.add_docs)} documents...")
        result = rag_system.add_documents(args.add_docs)
        if result['success']:
            print(f"✅ Successfully processed {result['files_processed']} files")
            print(f"📄 Created {result['chunks_added']} chunks")
        else:
            print(f"❌ Error: {result['message']}")
    
    if args.query:
        print(f"🔍 Processing query: {args.query}")
        result = rag_system.query(args.query)
        if result['success']:
            print(f"\n🤖 Answer: {result['answer']}")
            print(f"\n📚 Sources used: {len(result['sources'])}")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['source']} (similarity: {source['similarity_score']:.3f})")
        else:
            print(f"❌ Error: {result['answer']}")
    
    if args.stats:
        stats = rag_system.get_system_stats()
        print("\n📊 System Statistics:")
        print(f"Status: {stats.get('status', 'Unknown')}")
        print(f"Model: {stats.get('model_name', 'Unknown')}")
        print(f"Device: {stats.get('device', 'Unknown')}")
        if 'database' in stats:
            db = stats['database']
            print(f"Database: {db.get('type', 'Unknown')}")
            print(f"Documents: {db.get('total_documents', 0)}")
    
    if args.clear:
        result = rag_system.clear_database()
        if result['success']:
            print("✅ Database cleared successfully!")
        else:
            print(f"❌ Error: {result['message']}")

if __name__ == "__main__":
    main()
