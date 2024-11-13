# test_init.py
# python -m unittest discover tests

import unittest
from _package import _function, _Class  # import functions/classes from __init__.py

class TestInit(unittest.TestCase):

    def test_your_function(self):
        result = _function()
        self.assertEqual(result, expected_value)

    def test_your_class(self):
        obj = _Class()
        self.assertTrue(obj.some_attribute)
        self.assertEqual(obj.some_method(), expected_output)

if __name__ == '__main__':
    unittest.main()
