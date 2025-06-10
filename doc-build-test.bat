REM USE doc-test instead!!
echo use doc test...
rem @echo off
rem setlocal

rem :: Define paths
rem set REPO_URL=https://github.com/mynl/greater_tables_project
rem set BUILD_DIR=C:\tmp\greater_tables_docs
rem set VENV_DIR=%BUILD_DIR%\venv

rem :: Remove existing directory if it exists
rem if exist "%BUILD_DIR%" rd /s /q "%BUILD_DIR%"

rem :: Clone the latest development repo
rem git clone --depth 1 %REPO_URL% "%BUILD_DIR%"
rem if %errorlevel% neq 0 exit /b %errorlevel%

rem pushd "%BUILD_DIR%"

rem :: Create virtual environment
rem python -m venv "%VENV_DIR%"
rem if %errorlevel% neq 0 exit /b %errorlevel%

rem :: Activate virtual environment
rem call "%VENV_DIR%\Scripts\activate"

rem :: Upgrade pip and install dependencies from pyproject.toml
rem python -m pip install --upgrade pip
rem pip install --upgrade build setuptools
rem pip install .
rem pip install ".[doc]"  || pip install sphinx  # Ensure Sphinx is installed

rem :: Build the documentation
rem sphinx-build -b html docs docs/_build/html
rem if %errorlevel% neq 0 exit /b %errorlevel%

rem :: Deactivate virtual environment
rem deactivate

rem echo Documentation build complete: %BUILD_DIR%\docs\_build\html
rem endlocal
