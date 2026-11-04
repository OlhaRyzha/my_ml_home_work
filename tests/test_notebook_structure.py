import ast
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
COURSE_REFERENCE_NOTEBOOKS = {
    Path("13_nlp/Bert_Embeddings.ipynb"),
    Path("13_nlp/Bert_Fine_Tuning.ipynb"),
    Path("13_nlp/HW_NLP_Classification.ipynb"),
    Path("13_nlp/NLP_Classification.ipynb"),
    Path("13_nlp/Word_Embeddings.ipynb"),
}


def test_notebook_imports_are_only_in_first_code_cell() -> None:
    violations: list[str] = []

    for notebook_path in NOTEBOOKS_DIR.rglob("*.ipynb"):
        if ".ipynb_checkpoints" in notebook_path.parts:
            continue
        # Course examples introduce imports as each topic is taught.
        if notebook_path.relative_to(NOTEBOOKS_DIR) in COURSE_REFERENCE_NOTEBOOKS:
            continue

        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

        for cell_number, cell in enumerate(code_cells[1:], start=2):
            source = "".join(cell.get("source", []))
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            if any(
                isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body
            ):
                violations.append(f"{notebook_path}: code cell {cell_number}")

    assert not violations, "Imports outside the first code cell:\n" + "\n".join(
        violations
    )
