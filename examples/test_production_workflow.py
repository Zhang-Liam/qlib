#!/usr/bin/env python3
"""
Test script for the production workflow.
This script demonstrates how to use the production components with mock data.
"""

import os
import sys
import logging
from pathlib import Path

# Add qlib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qlib.production.workflow import ProductionWorkflow


def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_production.log')
        ]
    )


def test_single_cycle():
    """Test a single trading cycle."""
    print("Testing single trading cycle...")
    
    # Create workflow with mock configuration
    config_path = "examples/production_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        print("Please ensure the production_config.yaml file exists.")
        return False
    
    workflow = ProductionWorkflow(config_path)
    
    # Initialize components
    if not workflow.initialize_components():
        print("Failed to initialize components")
        return False
    
    # Run single cycle
    success = workflow.run_single_cycle()
    
    # Shutdown
    workflow.shutdown()
    
    if success:
        print("Single cycle test completed successfully!")
        return True
    else:
        print("Single cycle test failed!")
        return False


def test_continuous_mode():
    """Test continuous mode for a few cycles."""
    print("Testing continuous mode (3 cycles)...")
    
    config_path = "examples/production_config.yaml"
    workflow = ProductionWorkflow(config_path)
    
    if not workflow.initialize_components():
        print("Failed to initialize components")
        return False
    
    try:
        # Run for 3 cycles with 10-second intervals
        cycle_count = 0
        max_cycles = 3
        
        while cycle_count < max_cycles:
            print(f"\n--- Cycle {cycle_count + 1}/{max_cycles} ---")
            
            success = workflow.run_single_cycle()
            if success:
                print(f"Cycle {cycle_count + 1} completed successfully")
            else:
                print(f"Cycle {cycle_count + 1} failed")
            
            cycle_count += 1
            
            if cycle_count < max_cycles:
                print("Waiting 10 seconds before next cycle...")
                import time
                time.sleep(10)
        
        print("Continuous mode test completed!")
        return True
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return True
    finally:
        workflow.shutdown()


def test_component_initialization():
    """Test individual component initialization."""
    print("Testing component initialization...")
    
    config_path = "examples/production_config.yaml"
    workflow = ProductionWorkflow(config_path)
    
    # Test broker initialization
    print("Testing broker initialization...")
    workflow.broker = workflow.config.broker_config
    if hasattr(workflow.broker, 'connect'):
        print("✓ Broker interface available")
    else:
        print("✗ Broker interface not available")
    
    # Test risk manager initialization
    print("Testing risk manager initialization...")
    workflow.risk_manager = workflow.config.risk_config
    if hasattr(workflow.risk_manager, 'check_order'):
        print("✓ Risk manager interface available")
    else:
        print("✗ Risk manager interface not available")
    
    # Test live data initialization
    print("Testing live data initialization...")
    workflow.live_data = workflow.config.data_config
    if hasattr(workflow.live_data, 'get_latest_price'):
        print("✓ Live data interface available")
    else:
        print("✗ Live data interface not available")
    
    print("Component initialization test completed!")


def main():
    """Main test function."""
    print("=" * 60)
    print("QLIB PRODUCTION WORKFLOW TEST")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    
    # Test component initialization
    test_component_initialization()
    print()
    
    # Test single cycle
    if test_single_cycle():
        print()
        
        # Ask user if they want to test continuous mode
        response = input("Test continuous mode? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            test_continuous_mode()
    else:
        print("Single cycle test failed, skipping continuous mode test")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main() 