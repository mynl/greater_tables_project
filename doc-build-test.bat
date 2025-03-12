@echo off
setlocal

:: Define paths
set REPO_URL=https://github.com/mynl/greater_tables_project
set BUILD_DIR=C:\tmp\greater_tables_docs
set VENV_DIR=%BUILD_DIR%\venv

:: Remove existing directory if it exists
if exist "%BUILD_DIR%" rd /s /q "%BUILD_DIR%"

:: Clone the latest development repo
git clone --depth 1 %REPO_URL% "%BUILD_DIR%"
if %errorlevel% neq 0 exit /b %errorlevel%

pushd "%BUILD_DIR%"

:: Create virtual environment
python -m venv "%VENV_DIR%"
if %errorlevel% neq 0 exit /b %errorlevel%

:: Activate virtual environment
call "%VENV_DIR%\Scripts\activate"

:: Upgrade pip and install dependencies from pyproject.toml
python -m pip install --upgrade pip
pip install --upgrade build setuptools
pip install .
pip install ".[doc]"  || pip install sphinx  # Ensure Sphinx is installed

:: Build the documentation
sphinx-build -b html docs docs/_build/html
if %errorlevel% neq 0 exit /b %errorlevel%

:: Deactivate virtual environment
deactivate

echo Documentation build complete: %BUILD_DIR%\docs\_build\html
endlocal
