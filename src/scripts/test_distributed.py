"""
Simplified unit tests for the distributed training module.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.ml.distributed import AMDFuturesTensorParallel


class TestDistributed(unittest.TestCase):
    """Simplified tests for distributed module."""
    
    def test_initialization(self):
        """Test basic initialization."""
        trainer = AMDFuturesTensorParallel(session_type="RTH", contract="ES")
        self.assertEqual(trainer.session_type, "RTH")
        self.assertEqual(trainer.contract, "ES")
        self.assertIsNone(trainer.model)
    
    @patch('os.environ')
    def test_env_setup(self, mock_environ):
        """Test environment setup."""
        trainer = AMDFuturesTensorParallel()
        trainer._setup_distributed_env()
        
        # Just verify that environment variables were set
        self.assertTrue(mock_environ.__setitem__.called)
    
    @patch('torch.save')
    @patch('os.makedirs')
    def test_save_final_model(self, mock_makedirs, mock_save):
        """Test final model saving."""
        trainer = AMDFuturesTensorParallel()
        trainer.feature_columns = ['col1', 'col2']
        trainer.scaler = MagicMock()
        
        model = MagicMock()
        trainer._save_final_model(model)
        
        self.assertTrue(mock_save.called)
        self.assertTrue(mock_makedirs.called)


if __name__ == '__main__':
    unittest.main()