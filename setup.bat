@echo off
echo Installing Python dependencies...
pip install -r requirements.txt

REM Check if pdfinfo (from Poppler) is available in PATH
where pdfinfo >nul 2>nul
IF ERRORLEVEL 1 (
    echo Poppler not found. Installing Poppler for Windows...
    curl -L https://github.com/oschwartz10612/poppler-windows/releases/download/v23.05.0/Release-23.05.0-0.zip --output poppler.zip
    powershell -Command "Expand-Archive -Path poppler.zip -DestinationPath poppler -Force"
    setx PATH "%CD%\poppler\Release-23.05.0-0\poppler-23.05.0\Library\bin;%PATH%"
    echo Poppler installed and added to PATH.
) ELSE (
    echo Poppler is already installed.
)

echo Setup complete. You can now run the script using:
echo python ocr_cv.py
pause
