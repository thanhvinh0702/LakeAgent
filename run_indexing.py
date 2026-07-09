# Disable HuggingFace symlinks warning and requirement on Windows
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys
import asyncio
import platform

# Fix ProactorEventLoop error with psycopg on Windows
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from lake_agent.cli import (
    inventory,
    index_text,
    index_tabular,
    index_document,
    index_image,
    index_web,
    index_sql_script,
    index_database
)

def main():
    print("=========================================")
    print("STARTING DATA LAKE INDEXING PIPELINE")
    print("=========================================")
    
    print("\n[1/8] Running Inventory Scanner...")
    inventory.main([])
    
    print("\n[2/8] Indexing Text Files...")
    index_text.main([])
    
    print("\n[3/8] Indexing Tabular Files...")
    index_tabular.main([])
    
    print("\n[4/8] Indexing Documents (PDF, etc.)...")
    index_document.main([])
    
    print("\n[5/8] Indexing Images (with --no-ocr and --no-vlm for fast metadata)...")
    index_image.main(["--no-ocr", "--no-vlm"])
    
    print("\n[6/8] Indexing Web Pages (.html, .htm)...")
    index_web.main([])
    
    print("\n[7/8] Indexing SQL Scripts...")
    index_sql_script.main([])
    
    print("\n[8/8] Indexing Local Databases (.db, .sqlite)...")
    index_database.main([])
    
    print("\n=========================================")
    print("INDEXING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=========================================")

if __name__ == "__main__":
    main()
