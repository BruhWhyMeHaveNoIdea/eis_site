"""
Security-related tests for the retrospective analysis system.

This module tests security aspects including path traversal, injection attacks,
large payloads, sensitive data leakage, and other security considerations.
"""

import unittest
import numpy as np
import tempfile
import os
import json
import sys
from unittest.mock import patch, MagicMock
import warnings

# Import modules to test
from core.preprocessing import parse_russian_number, parse_russian_date, clean_russian_text
from core.embeddings import EmbeddingGenerator
from select_k_best import SelectKBest
from ml_retrospective import MLRetrospectiveAnalyzer
from config import get_config


class TestPathTraversal(unittest.TestCase):
    """Test path traversal vulnerabilities."""
    
    def test_file_path_traversal(self):
        """Test that file operations are not vulnerable to path traversal."""
        # Test with paths containing ../ sequences
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config",
            "normal/path/../../../etc/passwd",
            "/absolute/path/../../../etc/passwd",
            "C:\\Windows\\..\\..\\Autoexec.bat"
        ]
        
        # Test with load_tenders_from_json if it exists
        try:
            from core.preprocessing import load_tenders_from_json
            
            for path in malicious_paths:
                with self.assertRaises((FileNotFoundError, ValueError, OSError)):
                    # Should not allow reading outside allowed directories
                    load_tenders_from_json(path)
        except ImportError:
            pass  # Function may not exist
        
        # Test with embedding generator cache folder
        for path in malicious_paths:
            try:
                # Should validate path or raise error
                generator = EmbeddingGenerator(cache_folder=path)
                # If no error, ensure path is sanitized
                self.assertNotIn("..", generator.cache_folder)
            except (ValueError, OSError):
                pass  # Expected to fail
    
    def test_save_results_path_traversal(self):
        """Test saving results with malicious paths."""
        algorithm = SelectKBest()
        
        malicious_paths = [
            "../../../etc/passwd",
            "results/../../../../tmp/test.json",
            "/var/www/../../../etc/shadow"
        ]
        
        for path in malicious_paths:
            try:
                algorithm.save_results([], path)
                # If succeeds, verify file was created in safe location
                self.assertFalse(os.path.exists(path))  # Should not create in parent dir
            except (ValueError, OSError, PermissionError):
                pass  # Expected to fail
    
    def test_model_save_path_traversal(self):
        """Test model saving with malicious paths."""
        config = MLRetrospectiveConfig()
        analyzer = MLRetrospectiveAnalyzer(config)
        
        malicious_paths = [
            "../../../etc/passwd",
            "models/../../../../root/.ssh/authorized_keys",
            "C:\\Windows\\..\\..\\Autoexec.bat"
        ]
        
        for path in malicious_paths:
            try:
                analyzer.save_model(path)
                # If succeeds, verify path is safe
                self.assertNotIn("..", path)
            except (ValueError, OSError, PermissionError):
                pass  # Expected to fail


class TestInjectionAttacks(unittest.TestCase):
    """Test injection attack vulnerabilities."""
    
    def test_sql_injection_prevention(self):
        """Test that no SQL injection is possible (if using SQL)."""
        # The system doesn't use SQL directly, but test any database-like operations
        malicious_inputs = [
            "'; DROP TABLE tenders; --",
            "' OR '1'='1",
            "1; INSERT INTO users VALUES ('hacker', 'password')",
            "admin'--",
            "`rm -rf /`"
        ]
        
        # Test preprocessing functions
        for text in malicious_inputs:
            # Should handle as regular text, not execute
            cleaned = clean_russian_text(text)
            self.assertIsInstance(cleaned, str)
            
            # Should not contain SQL keywords after cleaning (if remove_punctuation)
            if ';' in text:
                # Semicolon might be removed
                pass
    
    def test_command_injection(self):
        """Test that no command injection is possible in system calls."""
        malicious_inputs = [
            "$(rm -rf /)",
            "| cat /etc/passwd",
            "; ls -la",
            "`id`",
            "&& shutdown -h now"
        ]
        
        # Test with parse_russian_number (might call system functions?)
        for text in malicious_inputs:
            try:
                result = parse_russian_number(text)
                # Should return 0.0 or handle as invalid number
                self.assertIsInstance(result, float)
            except Exception:
                pass  # May raise ValueError
        
        # Test with parse_russian_date
        for text in malicious_inputs:
            try:
                result = parse_russian_date(text)
                # Should return None or handle gracefully
                if result is not None:
                    self.assertIsInstance(result, datetime)
            except Exception:
                pass  # May raise ValueError
    
    def test_json_injection(self):
        """Test JSON injection/deserialization attacks."""
        malicious_json_strings = [
            '{"__class__": "os.system", "__args__": ["rm -rf /"]}',
            '{"malicious": true, "__reduce__": "evil"}',
            '{"key": "value", "sub": {"__import__": "os"}}'
        ]
        
        # Test with load_tenders_from_json if it exists
        try:
            from core.preprocessing import load_tenders_from_json
            
            for json_str in malicious_json_strings:
                # Write to temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    f.write(json_str)
                    temp_path = f.name
                
                try:
                    # Should raise JSONDecodeError or ValueError, not execute code
                    with self.assertRaises((json.JSONDecodeError, ValueError, KeyError)):
                        load_tenders_from_json(temp_path)
                finally:
                    os.unlink(temp_path)
        except ImportError:
            pass
        
        # Test direct json.loads in algorithm
        algorithm = SelectKBest()
        for json_str in malicious_json_strings:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(json_str)
                temp_path = f.name
            
            try:
                # Should fail to load as valid tender data
                with self.assertRaises((json.JSONDecodeError, ValueError, KeyError)):
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    algorithm.load_tenders(data)
            finally:
                os.unlink(temp_path)


class TestLargePayloads(unittest.TestCase):
    """Test handling of large payloads (DoS prevention)."""
    
    def test_large_text_fields(self):
        """Test with extremely large text fields."""
        # Generate very large text
        large_text = "A" * 10_000_000  # 10 MB string
        
        # Test preprocessing
        start = time.time()
        cleaned = clean_russian_text(large_text)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        self.assertLess(elapsed, 5.0)  # Less than 5 seconds
        self.assertIsInstance(cleaned, str)
        self.assertLess(len(cleaned), len(large_text))  # May be trimmed or same
    
    def test_large_number_of_tenders(self):
        """Test with very large number of tenders."""
        # Create many tenders (but not too many for test performance)
        n_tenders = 10000
        tenders = [
            {
                "Наименование объекта закупки": f"Тендер {i}",
                "Начальная (максимальная) цена контракта": "1 000 000,00"
            }
            for i in range(n_tenders)
        ]
        
        # Test Algorithm 1
        algorithm = SelectKBest()
        
        import time
        start = time.time()
        algorithm.load_tenders(tenders)
        load_time = time.time() - start
        
        # Loading should be efficient
        self.assertLess(load_time, 10.0)  # Less than 10 seconds
        
        # Test similarity search with large dataset
        target = {"Наименование объекта закупки": "Запрос"}
        start = time.time()
        results = algorithm.find_similar(target, k=10)
        search_time = time.time() - start
        
        self.assertLess(search_time, 15.0)  # Reasonable time
        self.assertLessEqual(len(results), 10)
    
    def test_large_embedding_vectors(self):
        """Test with very high-dimensional embeddings."""
        # Create high-dimensional vectors
        high_dim = 10000
        n_vectors = 100
        
        # Mock embedding generator to return high-dim vectors
        with patch.object(EmbeddingGenerator, 'encode') as mock_encode:
            mock_encode.return_value = np.random.randn(n_vectors, high_dim)
            
            generator = EmbeddingGenerator()
            
            # Test encoding
            texts = ["test"] * n_vectors
            start = time.time()
            embeddings = generator.encode(texts)
            elapsed = time.time() - start
            
            # Should handle high dimensions (though may be slow)
            self.assertLess(elapsed, 30.0)
            self.assertEqual(embeddings.shape, (n_vectors, high_dim))
    
    def test_memory_usage_large_batch(self):
        """Test memory usage with large batch processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large dataset
        n_tenders = 5000
        tenders = [
            {
                "Наименование объекта закупки": "A" * 1000,  # 1KB per tender
                "Начальная (максимальная) цена контракта": "1 000 000,00",
                "Регион": "Москва",
                "Описание": "B" * 5000  # 5KB
            }
            for i in range(n_tenders)
        ]
        
        # Load into algorithm
        algorithm = SelectKBest()
        algorithm.load_tenders(tenders)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500MB for 5000 tenders)
        self.assertLess(memory_increase, 500 * 1024 * 1024)  # 500 MB


class TestSensitiveDataLeakage(unittest.TestCase):
    """Test for sensitive data leakage in logs, errors, or outputs."""
    
    def test_error_messages_no_sensitive_data(self):
        """Test that error messages don't leak sensitive data."""
        # Test with invalid file path
        try:
            from core.preprocessing import load_tenders_from_json
            load_tenders_from_json("/etc/passwd")
        except Exception as e:
            error_msg = str(e)
            # Should not contain sensitive path in error message
            self.assertNotIn("passwd", error_msg)
            # Should use generic error
            self.assertIn("file", error_msg.lower())
        
        # Test with invalid tender data containing sensitive info
        sensitive_tender = {
            "Наименование объекта закупки": "Поставка",
            "Пароль": "secret123",
            "API ключ": "sk_live_1234567890",
            "Номер карты": "4111 1111 1111 1111"
        }
        
        algorithm = SelectKBest()
        try:
            algorithm.load_tenders([sensitive_tender])
            # If no error, check that sensitive data isn't logged
        except Exception as e:
            error_msg = str(e)
            # Should not leak sensitive data in error
            self.assertNotIn("secret123", error_msg)
            self.assertNotIn("sk_live", error_msg)
            self.assertNotIn("4111", error_msg)
    
    def test_logging_no_sensitive_data(self):
        """Test that logging doesn't include sensitive data."""
        import logging
        
        # Capture log messages
        log_capture = []
        
        class TestHandler(logging.Handler):
            def emit(self, record):
                log_capture.append(self.format(record))
        
        logger = logging.getLogger('core.preprocessing')
        handler = TestHandler()
        logger.addHandler(handler)
        
        # Process tender with sensitive data
        sensitive_tender = {
            "Наименование объекта закупки": "Поставка",
            "Пароль администратора": "admin123",
            "Токен": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        }
        
        from core.preprocessing import preprocess_tender
        preprocess_tender(sensitive_tender)
        
        # Check logs don't contain sensitive data
        for log_msg in log_capture:
            self.assertNotIn("admin123", log_msg)
            self.assertNotIn("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", log_msg)
        
        logger.removeHandler(handler)
    
    def test_output_no_sensitive_data(self):
        """Test that output results don't include unexpected sensitive fields."""
        algorithm = SelectKBest()
        
        tenders = [
            {
                "Идентификационный код закупки (ИКЗ)": "001",
                "Наименование объекта закупки": "Поставка",
                "Внутренний комментарий": "Конфиденциально: бюджет 2024",
                "Контактный телефон": "+7 900 123-45-67",
                "Email": "tender@example.com"
            }
        ]
        
        algorithm.load_tenders(tenders)
        
        target = {"Наименование объекта закупки": "Запрос"}
        results = algorithm.find_similar(target, k=5)
        
        # Results should not include sensitive fields unless explicitly requested
        for result in results:
            tender = result['tender']
            # Should not include internal comments, contact info in default output
            self.assertNotIn("Внутренний комментарий", tender)
            self.assertNotIn("Контактный телефон", tender)
            self.assertNotIn("Email", tender)
            # Should include non-sensitive fields
            self.assertIn("Наименование объекта закупки", tender)
    
    def test_configuration_no_secrets(self):
        """Test that configuration doesn't contain hardcoded secrets."""
        config = get_config()
        
        # Check for common secret patterns
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, str):
                    # Should not contain passwords, API keys, tokens
                    self.assertNotRegex(value, r'password\s*[:=]', 
                                       f"Config key {key} may contain password")
                    self.assertNotRegex(value, r'api[_-]?key\s*[:=]', 
                                       f"Config key {key} may contain API key")
                    self.assertNotRegex(value, r'sk_[live|test]_', 
                                       f"Config key {key} may contain Stripe key")
                    self.assertNotRegex(value, r'eyJhbGciOiJ', 
                                       f"Config key {key} may contain JWT token")


class TestInputValidation(unittest.TestCase):
    """Test input validation for security."""
    
    def test_malformed_json_input(self):
        """Test handling of malformed JSON input."""
        malformed_inputs = [
            '{"unclosed": "value"',  # Unclosed object
            '{"valid": "value"} extra text',  # Extra text after JSON
            '{"depth": {"nested": {"too": {"deep": {}}}}}',  # Very deep nesting
            '{"array": [' + '0,' * 10000 + '0]}',  # Very large array
            '{"key": "' + 'A' * 1000000 + '"}',  # Very large string
        ]
        
        for json_str in malformed_inputs:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(json_str)
                temp_path = f.name
            
            try:
                algorithm = SelectKBest()
                
                # Should raise appropriate error
                with self.assertRaises((json.JSONDecodeError, ValueError, MemoryError)):
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    algorithm.load_tenders(data)
            finally:
                os.unlink(temp_path)
    
    def test_deeply_nested_structures(self):
        """Test handling of deeply nested structures (stack overflow prevention)."""
        # Create deeply nested dictionary
        def create_deep_nest(depth):
            d = {}
            current = d
            for i in range(depth):
                current['nested'] = {}
                current = current['nested']
            return d
        
        # Test with reasonable depth (should work)
        deep_dict = create_deep_nest(100)
        tender = {"data": deep_dict, "Наименование объекта закупки": "test"}
        
        algorithm = SelectKBest()
        try:
            algorithm.load_tenders([tender])
            # Should handle without stack overflow
        except RecursionError:
            self.fail("RecursionError with deeply nested structure")
    
    def test_circular_references(self):
        """Test handling of circular references in input."""
        # Create circular reference
        import gc
        
        circular_dict = {"name": "circular"}
        circular_dict["self"] = circular_dict  # Circular reference
        
        tender = {"data": circular_dict, "Наименование объекта закупки": "test"}
        
        algorithm = SelectKBest()
        try:
            algorithm.load_tenders([tender])
            # Should handle without infinite recursion
        except (RecursionError, ValueError):
            pass  # Expected to fail gracefully
    
    def test_binary_data_in_text_fields(self):
        """Test handling of binary data in text fields."""
        binary_inputs = [
            b'\x00\x01\x02\x03',  # Raw bytes
            'Text with null \x00 character',
            'UTF-8 with BOM \xef\xbb\xbf',
            'Text with control characters \x07\x08',
        ]
        
        for binary in binary_inputs:
            if isinstance(binary, bytes):
                text = binary.decode('latin-1')  # Force decode
            else:
                text = binary
            
            tender = {"Наименование объекта закупки": text}
            
            # Should handle without crashing
            from core.preprocessing import preprocess_tender
            processed = preprocess_tender(tender)
            self.assertIsInstance(processed, dict)


class TestResourceLimits(unittest.TestCase):
    """Test resource limit enforcement."""
    
    def test_cpu_time_limit(self):
        """Test that operations don't use excessive CPU time."""
        import time
        import signal
        
        # This test is tricky to implement without affecting test runner
        # We'll do a simple performance test instead
        algorithm = SelectKBest()
        
        # Create moderate dataset
        tenders = [{"Наименование объекта закупки": f"Тендер {i}"} for i in range(1000)]
        algorithm.load_tenders(tenders)
        
        target = {"Наименование объекта закупки": "Запрос"}
        
        # Time the operation
        start = time.time()
        results = algorithm.find_similar(target, k=10)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        self.assertLess(elapsed, 5.0)  # Less than 5 seconds
    
    def test_memory_limit(self):
        """Test memory usage limits."""
        import psutil
        import os
        
        # Note: This test may be flaky in CI environments
        process = psutil.Process(os.getpid())
        
        # Create large dataset
        n_tenders = 2000
        large_tenders = [
            {
                "Наименование объекта закупки": "A" * 10000,  # 10KB string
                "Описание": "B" * 50000,  # 50KB string
                "Дополнительно": "C" * 100000  # 100KB string
            }
            for i in range(n_tenders)
        ]
        
        # Measure memory before
        initial_memory = process.memory_info().rss
        
        algorithm = SelectKBest()
        
        # Load tenders (this should use significant memory)
        algorithm.load_tenders(large_tenders[:100])  # Only 100 to avoid OOM
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        # 100 tenders * ~160KB each = ~16MB
        self.assertLess(memory_increase, 200 * 1024 * 1024)  # 200 MB upper bound


# Helper for time module
import time
from datetime import datetime

if __name__ == '__main__':
    unittest.main(verbosity=2)