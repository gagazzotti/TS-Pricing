"""TBD"""

import unittest

# Importer les modules de tests dans le même dossier
from tests.quick.test_ts import TestMellinTS
from tests.quick.test_ts_p import TestMellinTSp


# Créer une suite de tests
def test_suite():
    """TBD"""
    suite = unittest.TestSuite()
    for test in (TestMellinTS, TestMellinTSp):
        suite.addTest(unittest.TestLoader().loadTestsFromTestCase(test))

    return suite


if __name__ == "__main__":
    # Créer un runner pour exécuter la suite
    runner = unittest.TextTestRunner()
    runner.run(test_suite())
