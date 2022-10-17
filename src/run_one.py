import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError

notebook = "Final_table.ipynb"

with open(notebook) as f:
    nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=6000, kernel_name='python3')
    try:
        ep.preprocess(nb, {'metadata': {'path': ''}})
    except CellExecutionError:
        out = None
        msg = 'Error executing the notebook "%s".\n\n' % notebook
        msg += 'See notebook "%s" for the traceback.' % notebook
        print(msg)
        raise
    finally:
        with open(notebook, mode='w', encoding='utf-8') as f:
            nbformat.write(nb, f)
            with open(notebook, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)
