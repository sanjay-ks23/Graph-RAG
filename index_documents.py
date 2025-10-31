import argparse
from src.utils.config_loader import ConfigLoader
from src.pipeline.indexing_pipeline import IndexingPipeline
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """Main indexing script"""
    parser = argparse.ArgumentParser(
        description='Index therapy books into Graph RAG system'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--force-reindex',
        action='store_true',
        help='Force reindexing even if data exists'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    
    # Create and run pipeline
    pipeline = IndexingPipeline(config_loader.config)
    pipeline.run(force_reindex=args.force_reindex)
    
    # Print statistics
    stats = pipeline.get_statistics()
    logger.info(f"Graph: {stats['graph']}")
    logger.info(f"Vector Store: {stats['vector_store']}")

if __name__ == '__main__':
    main()
