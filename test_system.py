from system import ML_System_Regression
import unittest

class test(unittest.TestCase):
    def test_prueba(self):
        sistema = ML_System_Regression()
        resultado = sistema.ML_Flow_regression()
        self.assertTrue(resultado["success"],"Modelo ejecutado correctamente")
        self.assertGreaterEqual(resultado["accuracy"],70,"The model accuracy be above 0.7")

if __name__ == "__main__":
    unittest.main()