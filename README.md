# QuantEcon.cheatsheet (KDB+ Addition)
A cheatsheet for MATLAB, Python, Julia, and KDB+

* The html file is in: _build/html/index.html

## Instructions for Building the Project

To build the project, follow these steps:

1. **Install Sphinx**

   First, make sure you have Sphinx installed. You can upgrade it to the latest version using pip:

   ```shell
   pip install -U sphinx
   ```

2. **Building from Makefile**

    Clean the project and make html files:
    ```shell
    make clean
    make html
    ```

3. **Modify code cells**

    In order to modify KDB+ code cells, make sure to open index.rst file in the root directory and change the text accordingly.