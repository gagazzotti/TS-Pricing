"""TBD"""

import unittest

from tests.complete.test_bg import TestMellinBG

# Importer les modules de tests dans le même dossier
from tests.complete.test_ts import TestMellinTS


# Créer une suite de tests
def test_suite():
    """TBD"""
    suite = unittest.TestSuite()
    for test in [TestMellinBG, TestMellinTS]:
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(test))

    return suite


if __name__ == "__main__":
    # Créer un runner pour exécuter la suite
    runner = unittest.TextTestRunner()
    runner.run(test_suite())
