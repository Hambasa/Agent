import unittest
from main import TaskAnalyzer, TaskComplexity


class StubMetaCognition:
    def assess_capability_for_task(self, task: str) -> float:
        # Always return full confidence to simplify tests
        return 1.0


class StubAgent:
    def __init__(self):
        self.meta_cognition = StubMetaCognition()


class TaskAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = TaskAnalyzer(StubAgent())

    def test_simple_task(self):
        analysis = self.analyzer.analyze_task("open a file")
        self.assertEqual(analysis.complexity, TaskComplexity.SIMPLE)

    def test_moderate_task(self):
        analysis = self.analyzer.analyze_task("create and configure project")
        self.assertEqual(analysis.complexity, TaskComplexity.MODERATE)

    def test_complex_task(self):
        analysis = self.analyzer.analyze_task(
            "integrate multiple systems and orchestrate workflows"
        )
        self.assertEqual(analysis.complexity, TaskComplexity.COMPLEX)


if __name__ == "__main__":
    unittest.main()
