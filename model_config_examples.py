#!/usr/bin/env python3
"""
Examples of how to configure different models for the chess app.

This file shows how to easily switch between different model configurations
including Falcon, Stockfish, and other options.
"""

# Example 1: Switch to Falcon for move prediction
def setup_falcon_moves():
    """Configure Falcon model for move prediction"""
    from app import configure_falcon_for_moves, switch_model_preset
    
    # Option 1: Use the built-in preset
    switch_model_preset("falcon_setup")
    
    # Option 2: Configure manually
    configure_falcon_for_moves("tiiuae/falcon-7b-instruct", use_local=True)  # FIXME: Choose specific Falcon model variant and size
    
    print("Falcon configured for move prediction")

# Example 2: Use Stockfish for move prediction (default)
def setup_stockfish_moves():
    """Configure Stockfish for move prediction"""
    from app import switch_model_preset
    
    switch_model_preset("default_cnn_stockfish")
    print("Stockfish configured for move prediction")

# Example 3: Use API for everything
def setup_full_api():
    """Configure API models for both piece classification and moves"""
    from app import switch_model_preset
    
    switch_model_preset("full_api")
    print("Full API configuration enabled")

# Example 4: Custom configuration
def setup_custom_config():
    """Example of custom model configuration"""
    from app import MODEL_CONFIG
    
    # Custom configuration example
    MODEL_CONFIG["move_prediction"]["local"] = {
        "type": "falcon",
        "model_name": "tiiuae/falcon-40b-instruct"  # FIXME: Select optimal Falcon model size for performance/accuracy balance
    }
    
    MODEL_CONFIG["move_prediction"]["api"] = {
        "model": "tiiuae/falcon-7b-instruct",  # FIXME: Configure API model for production use
        "max_tokens": 15,  # FIXME: Adjust token limit based on move format requirements
        "fallback_to_local": True  # FIXME: Set fallback strategy for production reliability
    }
    
    print("Custom Falcon configuration applied")

if __name__ == "__main__":
    print("Chess App Model Configuration Examples")
    print("="*50)
    
    print("\n1. Setting up Falcon for moves...")
    setup_falcon_moves()
    
    print("\n2. Setting up Stockfish for moves...")
    setup_stockfish_moves()
    
    print("\n3. Setting up full API...")
    setup_full_api()
    
    print("\n4. Setting up custom config...")
    setup_custom_config()
    
    print("\n" + "="*50)
    print("Configuration examples completed!")
    print("Edit app.py and uncomment the desired configuration lines.")  # FIXME: Update instructions for your production deployment workflow
